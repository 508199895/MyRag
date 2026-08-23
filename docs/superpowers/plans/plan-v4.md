# 本地 Markdown 文档问答工具 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 `spec-v4.md` 实现第二版本地 Markdown RAG 主链路：递归加载 Markdown 菜谱、构建或加载 FAISS、执行向量 + BM25 + RRF 混合检索、按意图生成回答，并通过 `python -m src` 进入连续问答。

**Architecture:** 项目采用“CLI 入口 + `RagService` 生命周期编排 + 文档准备/索引构建/检索/生成模块协作”。`RagService` 只负责启动顺序、知识库构建和单轮问答编排；文档 metadata、FAISS 生命周期、混合检索、Prompt 与 LLM 调用分别收敛到独立模块，避免后续 API/Web 或评测扩展污染第二版主链路。

**Tech Stack:** Python 3.13、LangChain、LangChain Community、LangChain OpenAI-compatible Chat model、FAISS、HuggingFace/sentence-transformers、python-dotenv、PyYAML、Pydantic、jieba、pytest、pytest-cov、ruff。

**Spec:** `docs/superpowers/specs/spec-v4.md`

## Global Constraints

- 始终使用简体中文沟通与编写 Markdown 正文。
- 第二版只支持本地 Markdown 文档，不支持 TXT、PDF、网页抓取、图片、多模态文档。
- 第二版不支持 parquet/jsonl/tsv 数据集加载。
- 第二版不实现评测数据集加载、检索评测、生成评测、Web/API 服务、多轮对话记忆、索引过期检测、增量索引、debug 模式或日志系统。
- 启动命令只有 `python -m src`；CLI 不提供 `-s`、`--debug`、`--rebuild` 等启动参数。
- 流式输出由 `config.yaml` 的 `generation.stream` 控制。
- 自动化测试不得真实调用 LLM API；查询路由、查询优化和最终回答生成都应使用 fake/mock client。
- 文档 ID 与 chunk ID 使用 Python 标准库 `hashlib` 中的 MD5 哈希算法。
- 配置加载必须使用 `yaml.safe_load`、`load_dotenv` 和 `AppConfig.model_validate(raw_config)`。
- 所有配置段都允许未声明字段；只校验字段类型，不校验字段值范围。
- 新增直接依赖必须写入 `requirements.txt`，记录实际安装和验证使用的版本号，不使用 `pip freeze` 生成整棵依赖树。
- 实施阶段如需实际执行依赖安装命令、联网下载模型，或由 agent 主动触发模型下载，必须先向用户请求权限。
- Git 提交前必须明确向用户请求权限，并在用户同意后再执行 `git commit`。
- 不提交或泄露 `.env`、API Key、FAISS 索引产物、虚拟环境、缓存目录。

---

## File Structure

- Modify: `src/config.py`。更新配置模型，支持 `retrieval.vector/bm25/hybrid`、`generation.stream` 和四个 Prompt 模板路径；启动阶段检查 Prompt 路径存在。
- Modify: `config.example.yaml`。同步 `spec-v4.md` 推荐配置结构。
- Create/Modify: `.env.example`。保留 `DEEPSEEK_API_KEY=your-api-key`。
- Modify: `requirements.txt`。记录实际验证过的直接依赖版本。
- Use existing/create: `docs/prompts/query_router.md`、`docs/prompts/query_rewrite.md`、`docs/prompts/generate_step_by_step_answer.md`、`docs/prompts/generate_basic_answer.md`。提供查询路由、查询改写、详细回答和一般回答四类 Prompt 模板；`list` 意图不使用回答 Prompt。
- Create: `src/document_preparation/__init__.py`、`src/document_preparation/module.py`。实现 `DocumentPreparationModule`。
- Create: `src/indexing/__init__.py`、`src/indexing/module.py`。实现 `IndexConstructionModule`。
- Create: `src/retrieval/__init__.py`、`src/retrieval/module.py`。实现 `RetrievalModule`。
- Create: `src/generation/module.py`。实现 `GenerationModule`；旧 `src/generation/prompts.py` 只作为历史兼容辅助，不作为主链路入口。
- Create: `src/service.py`。实现 `RagService`。
- Create: `src/__main__.py`。实现 `python -m src` 入口。
- Create/Modify: `tests/unit/`、`tests/integration/`、`tests/e2e/`、`tests/regression/`、`tests/fixtures/` 下的对应测试与 fixture。
- Modify: `.gitignore`。补齐 `.env`、`storage/`、`*.faiss`、`*.pkl`、`__pycache__/`、`.pytest_cache/`、`RAG_2026/`、`ENV/RAG_2026/`。
- Modify: `README.md`。补充第二版运行与验收说明。

---

### Task 1: 配置契约、模板路径校验和依赖清单

**Files:**
- Modify: `src/config.py`
- Modify: `config.example.yaml`
- Create/Modify: `.env.example`
- Modify: `requirements.txt`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: `load_config(config_path: str | Path = "config.yaml", env_path: str | Path = ".env") -> AppConfig`
- Produces:
  - `AppConfig.data_path: str`
  - `AppConfig.index_save_path: str`
  - `AppConfig.splitter: MarkdownHeaderSplitterConfig`
  - `AppConfig.embedding: EmbeddingConfig`
  - `AppConfig.retrieval.vector.search_type: str`
  - `AppConfig.retrieval.vector.top_k: int`
  - `AppConfig.retrieval.bm25.top_k: int`
  - `AppConfig.retrieval.hybrid.top_k: int`
  - `AppConfig.generation.stream: bool`
  - `AppConfig.generation.query_router_prompt_template_path: str`
  - `AppConfig.generation.query_rewrite_prompt_template_path: str`
  - `AppConfig.generation.step_by_step_answer_prompt_template_path: str`
  - `AppConfig.generation.basic_answer_prompt_template_path: str`

- [ ] **Step 1: Write failing tests for the v4 config shape**

Replace the valid config fixture in `tests/unit/test_config.py` with:

```python
VALID_CONFIG = """
data_path: data/cook
index_save_path: storage/faiss/default

splitter:
  type: markdown_header
  headers_to_split_on:
    - ["#", "h1"]
    - ["##", "h2"]
    - ["###", "h3"]
  strip_headers: false

embedding:
  model_name: BAAI/bge-small-zh-v1.5

retrieval:
  vector:
    search_type: similarity
    top_k: 5
  bm25:
    top_k: 5
  hybrid:
    top_k: 3

generation:
  provider: openai_compatible
  base_url: https://api.deepseek.com
  model_name: deepseek-v4-flash
  api_key: $DEEPSEEK_API_KEY
  stream: true
  temperature: 0.2
  max_tokens: 1024
  query_router_prompt_template_path: docs/prompts/query_router.md
  query_rewrite_prompt_template_path: docs/prompts/query_rewrite.md
  step_by_step_answer_prompt_template_path: docs/prompts/generate_step_by_step_answer.md
  basic_answer_prompt_template_path: docs/prompts/generate_basic_answer.md
"""
```

Add:

```python
def test_load_config_returns_v4_typed_app_config(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path, env_path = write_valid_files(tmp_path)

    config = load_config(config_path=config_path, env_path=env_path)

    assert config.retrieval.vector.search_type == "similarity"
    assert config.retrieval.vector.top_k == 5
    assert config.retrieval.bm25.top_k == 5
    assert config.retrieval.hybrid.top_k == 3
    assert config.generation.stream is True
    assert config.generation.query_router_prompt_template_path == "docs/prompts/query_router.md"
    assert config.generation.query_rewrite_prompt_template_path == "docs/prompts/query_rewrite.md"
    assert config.generation.step_by_step_answer_prompt_template_path == "docs/prompts/generate_step_by_step_answer.md"
    assert config.generation.basic_answer_prompt_template_path == "docs/prompts/generate_basic_answer.md"
```

- [ ] **Step 2: Run config tests and verify they fail**

Run:

```bash
E:/007.agent/007.project/RAG/ENV/RAG_2026/python.exe -m pytest tests/unit/test_config.py -v
```

Expected: FAIL because current `src/config.py` still has the older retrieval and generation shape.

- [ ] **Step 3: Update `src/config.py` models**

Implement these model classes and keep `extra="allow"` on all config sections:

```python
class RetrievalVectorConfig(ConfigModel):
    model_config = ConfigDict(extra="allow")

    search_type: str = Field(description="传给 LangChain FAISS as_retriever 的 search_type。")
    top_k: int = Field(description="向量检索返回的 chunk 数量。")


class RetrievalBm25Config(ConfigModel):
    model_config = ConfigDict(extra="allow")

    top_k: int = Field(description="BM25 检索返回的 chunk 数量。")


class RetrievalHybridConfig(ConfigModel):
    model_config = ConfigDict(extra="allow")

    top_k: int = Field(description="RRF 重排后返回的候选 chunk 数量。")


class RetrievalConfig(ConfigModel):
    model_config = ConfigDict(extra="allow")

    vector: RetrievalVectorConfig
    bm25: RetrievalBm25Config
    hybrid: RetrievalHybridConfig


class GenerationConfig(ConfigModel):
    model_config = ConfigDict(extra="allow")

    provider: str = Field(description="LLM provider 名称。")
    base_url: str = Field(description="OpenAI-compatible API base URL。")
    model_name: str = Field(description="LLM 模型名。")
    api_key: str = Field(description="LLM API key。")
    stream: bool = Field(description="是否流式输出。")
    temperature: float = Field(description="生成温度。")
    max_tokens: int = Field(description="最大输出 token 数。")
    query_router_prompt_template_path: str
    query_rewrite_prompt_template_path: str
    step_by_step_answer_prompt_template_path: str
    basic_answer_prompt_template_path: str
```

- [ ] **Step 4: Add Prompt path existence validation during config load**

After `AppConfig.model_validate(resolved_config_data)`, check the five configured prompt paths relative to the current process working directory unless they are absolute. Raise `ConfigError` with every missing path:

```python
PROMPT_PATH_FIELDS = (
    "query_router_prompt_template_path",
    "query_rewrite_prompt_template_path",
    "step_by_step_answer_prompt_template_path",
    "basic_answer_prompt_template_path",
)
```

- [ ] **Step 5: Sync `config.example.yaml` and `.env.example`**

`config.example.yaml` must use the exact v4 generation prompt fields and include `generation.stream: true`. `.env.example` must contain:

```env
DEEPSEEK_API_KEY=your-api-key
```

- [ ] **Step 6: Update dependency list only after verification**

If dependencies must be installed, ask the user first. After installation/version verification, pin direct dependencies in `requirements.txt` for:

```text
python-dotenv
PyYAML
pydantic
pytest
pytest-cov
ruff
langchain
langchain-community
langchain-openai
faiss-cpu
sentence-transformers
jieba
```

- [ ] **Step 7: Run config tests and verify they pass**

Run:

```bash
E:/007.agent/007.project/RAG/ENV/RAG_2026/python.exe -m pytest tests/unit/test_config.py -v
```

Expected: PASS.

- [ ] **Step 8: Request commit permission**

Ask the user before committing. If approved, run:

```bash
git add src/config.py config.example.yaml .env.example requirements.txt tests/unit/test_config.py
git commit -m "chore: align config contract with spec v4"
```

---

### Task 2: Prompt 模板文件与 GenerationModule 初始化

**Files:**
- Use existing/create: `docs/prompts/query_router.md`
- Use existing/create: `docs/prompts/query_rewrite.md`
- Use existing/create: `docs/prompts/generate_step_by_step_answer.md`
- Use existing/create: `docs/prompts/generate_basic_answer.md`
- Create: `src/generation/module.py`
- Modify: `src/generation/__init__.py`
- Test: `tests/unit/test_generation_module.py`
- Test: `tests/unit/test_prompts.py`

**Interfaces:**
- Consumes: `GenerationConfig`
- Produces:
  - `GenerationModule.__init__(config: GenerationConfig) -> None`
  - `GenerationModule.setup_llm() -> None`
  - `GenerationModule.prompts: dict[str, ChatPromptTemplate]`
  - `GenerationModule.llm`

- [ ] **Step 1: Write failing tests for prompt files**

Update `tests/unit/test_prompts.py` so it checks the four v4 prompt files:

```python
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("path", "required_variables"),
    [
        ("docs/prompts/query_router.md", ("{query}",)),
        ("docs/prompts/query_rewrite.md", ("{query}",)),
        ("docs/prompts/generate_step_by_step_answer.md", ("{question}", "{context}")),
        ("docs/prompts/generate_basic_answer.md", ("{question}", "{context}")),
    ],
)
def test_v4_prompt_template_exists_and_contains_required_variables(path, required_variables):
    template_path = Path(path)

    assert template_path.is_file()
    content = template_path.read_text(encoding="utf-8")
    for variable in required_variables:
        assert variable in content
```

- [ ] **Step 2: Write failing tests for module initialization**

Create `tests/unit/test_generation_module.py`:

```python
from pathlib import Path

import pytest

from src.config import GenerationConfig
from src.generation.module import GenerationModule, GenerationModuleError


def make_generation_config(tmp_path: Path, **overrides) -> GenerationConfig:
    paths = {}
    prompts = {
        "query_router": "用户问题：{query}",
        "query_rewrite": "用户问题：{query}",
        "step_by_step_answer": "用户问题：{question}\n检索内容：{context}",
        "basic_answer": "用户问题：{question}\n检索内容：{context}",
    }
    for name, content in prompts.items():
        prompt_path = tmp_path / f"{name}.md"
        prompt_path.write_text(content, encoding="utf-8")
        paths[name] = str(prompt_path)
    data = {
        "provider": "openai_compatible",
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-v4-flash",
        "api_key": "test-key",
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 1024,
        "query_router_prompt_template_path": paths["query_router"],
        "query_rewrite_prompt_template_path": paths["query_rewrite"],
        "step_by_step_answer_prompt_template_path": paths["step_by_step_answer"],
        "basic_answer_prompt_template_path": paths["basic_answer"],
    }
    data.update(overrides)
    return GenerationConfig.model_validate(data)


def test_generation_module_checks_prompt_paths(tmp_path):
    missing = tmp_path / "missing.md"
    config = make_generation_config(tmp_path, basic_answer_prompt_template_path=str(missing))

    with pytest.raises(GenerationModuleError) as exc_info:
        GenerationModule(config)

    assert str(missing) in str(exc_info.value)
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
E:/007.agent/007.project/RAG/ENV/RAG_2026/python.exe -m pytest tests/unit/test_prompts.py tests/unit/test_generation_module.py -v
```

Expected: FAIL because v4 prompt files and `GenerationModule` are not implemented.

- [ ] **Step 4: Verify the existing four Prompt templates**

The user has already downloaded the four Prompt templates into `docs/prompts/`. Do not overwrite their content. Verify these files exist and keep their required variables:

```text
docs/prompts/query_router.md: {query}
docs/prompts/query_rewrite.md: {query}
docs/prompts/generate_step_by_step_answer.md: {question}, {context}
docs/prompts/generate_basic_answer.md: {question}, {context}
```

If one file is missing, create only the missing file with the same required variables and concise Chinese instructions matching its filename. If a file exists but lacks a required variable, stop with `NEEDS_CONTEXT` instead of replacing user-provided prompt wording.

- [ ] **Step 5: Implement initialization**

In `src/generation/module.py`, implement:

```python
class GenerationModuleError(Exception):
    """Raised when generation module setup or execution fails."""
```

`GenerationModule.__init__` stores config, checks every prompt path, loads each as `ChatPromptTemplate.from_template(path.read_text(encoding="utf-8"))`, and stores them under keys `query_router`、`query_rewrite`、`step_by_step_answer`、`basic_answer`.

- [ ] **Step 6: Implement `setup_llm()` with LangChain Chat model**

Use `langchain_openai.ChatOpenAI` for OpenAI-compatible DeepSeek:

```python
from langchain_openai import ChatOpenAI

self.llm = ChatOpenAI(
    api_key=self.config.api_key,
    base_url=self.config.base_url,
    model=self.config.model_name,
    temperature=self.config.temperature,
    max_tokens=self.config.max_tokens,
    streaming=self.config.stream,
)
```

If initialization fails, raise `GenerationModuleError` with provider, base_url, model_name, and the original failure reason.

- [ ] **Step 7: Run prompt and generation initialization tests**

Run:

```bash
E:/007.agent/007.project/RAG/ENV/RAG_2026/python.exe -m pytest tests/unit/test_prompts.py tests/unit/test_generation_module.py -v
```

Expected: PASS.

- [ ] **Step 8: Request commit permission**

Ask the user before committing. If approved, run:

```bash
git add docs/prompts src/generation tests/unit/test_prompts.py tests/unit/test_generation_module.py
git commit -m "feat: add v4 prompts and generation module setup"
```

---

### Task 3: 文档准备模块

**Files:**
- Create: `src/document_preparation/__init__.py`
- Create: `src/document_preparation/module.py`
- Test: `tests/unit/test_document_preparation.py`

**Interfaces:**
- Consumes: `data_path: str | Path`, `splitter_config: MarkdownHeaderSplitterConfig`
- Produces:
  - `DocumentPreparationModule.documents: list[Document]`
  - `DocumentPreparationModule.chunks: list[Document]`
  - `DocumentPreparationModule.child_parent_map: dict[str, str]`
  - `load_documents() -> list[Document]`
  - `_enhance_metadata() -> None`
  - `split_documents() -> list[Document]`
  - `get_ranked_parent_docs(chunks: list[Document]) -> list[Document]`

- [ ] **Step 1: Write failing loading and metadata tests**

Create `tests/unit/test_document_preparation.py`:

```python
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.config import MarkdownHeaderSplitterConfig
from src.document_preparation.module import (
    DocumentPreparationError,
    DocumentPreparationModule,
)


def make_splitter_config() -> MarkdownHeaderSplitterConfig:
    return MarkdownHeaderSplitterConfig.model_validate(
        {
            "type": "markdown_header",
            "headers_to_split_on": [("#", "h1"), ("##", "h2"), ("###", "h3")],
            "strip_headers": False,
        }
    )


def test_load_documents_recurses_md_only_and_enhances_parent_metadata(tmp_path):
    doc_path = tmp_path / "meat_dish" / "红烧肉.md"
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text("# 红烧肉\n难度：★★★\n哨兵：糖色", encoding="utf-8")
    (tmp_path / "meat_dish" / "ignore.txt").write_text("txt", encoding="utf-8")
    root_doc = tmp_path / "根目录菜.md"
    root_doc.write_text("# 根目录菜\n无星级", encoding="utf-8")

    module = DocumentPreparationModule(tmp_path, make_splitter_config())
    docs = module.load_documents()

    assert len(docs) == 2
    by_source = {doc.metadata["source"]: doc for doc in docs}
    meat = by_source["meat_dish/红烧肉.md"].metadata
    assert meat["doc_type"] == "parent"
    assert meat["category"] == "荤菜"
    assert meat["dish_name"] == "红烧肉"
    assert meat["difficulty"] == "中等"
    assert len(meat["parent_id"]) == 32
    assert by_source["根目录菜.md"].metadata["category"] == "未知"


def test_load_documents_rejects_missing_data_path(tmp_path):
    module = DocumentPreparationModule(tmp_path / "missing", make_splitter_config())

    with pytest.raises(DocumentPreparationError, match="路径不存在"):
        module.load_documents()
```

- [ ] **Step 2: Run loading tests and verify they fail**

Run:

```bash
E:/007.agent/007.project/RAG/ENV/RAG_2026/python.exe -m pytest tests/unit/test_document_preparation.py -v
```

Expected: FAIL because the module is not implemented.

- [ ] **Step 3: Implement Markdown loading and parent metadata**

Implementation requirements:

- Recursively load only `.md` files using `Path.rglob("*.md")`.
- Use UTF-8.
- Skip empty or whitespace-only Markdown files and print a warning containing `文档内容为空` and the path.
- Raise `DocumentPreparationError(f"路径不存在：{self.data_path}")` when `data_path` does not exist.
- Raise `DocumentPreparationError("文档库为空")` when no Markdown files exist.
- If all Markdown files are empty, raise `DocumentPreparationError("文档库为空")`.
- Compute `source` with `path.relative_to(data_path).as_posix()`.
- Compute `parent_id` with `hashlib.md5(source.encode("utf-8")).hexdigest()`.
- Infer `category` from the first source path segment and `CATEGORY_MAPPING`; root documents and unmapped categories become `未知`.
- Infer `dish_name` from `Path(source).stem`.
- Infer `difficulty` from the first consecutive `★+` match; 1-5 map to `DIFFICULTY_MAPPING`, anything else becomes `未知`.

- [ ] **Step 4: Write failing chunk and parent ranking tests**

Add:

```python
def test_split_documents_generates_child_metadata_and_map(tmp_path):
    doc_path = tmp_path / "vegetable_dish" / "清炒菜心.md"
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text("# 清炒菜心\n## 食材\n菜心\n## 做法\n快炒", encoding="utf-8")
    module = DocumentPreparationModule(tmp_path, make_splitter_config())
    parents = module.load_documents()

    chunks = module.split_documents()

    assert parents
    assert chunks
    first = chunks[0].metadata
    assert first["doc_type"] == "child"
    assert first["parent_id"] == parents[0].metadata["parent_id"]
    assert first["chunk_index"] == 0
    assert len(first["chunk_id"]) == 32
    assert module.child_parent_map[first["chunk_id"]] == first["parent_id"]
    assert all(chunk.page_content.strip() for chunk in chunks)


def test_get_ranked_parent_docs_orders_by_hit_count_then_first_position(tmp_path, capsys):
    module = DocumentPreparationModule(tmp_path, make_splitter_config())
    parent_a = Document(page_content="A", metadata={"parent_id": "a", "dish_name": "A"})
    parent_b = Document(page_content="B", metadata={"parent_id": "b", "dish_name": "B"})
    module.documents = [parent_a, parent_b]
    chunks = [
        Document(page_content="b1", metadata={"parent_id": "b", "chunk_id": "b1"}),
        Document(page_content="a1", metadata={"parent_id": "a", "chunk_id": "a1"}),
        Document(page_content="a2", metadata={"parent_id": "a", "chunk_id": "a2"}),
        Document(page_content="missing", metadata={"parent_id": "x", "chunk_id": "x1"}),
    ]

    ranked = module.get_ranked_parent_docs(chunks)

    assert ranked == [parent_a, parent_b]
    assert "无法回溯父文档" in capsys.readouterr().out
```

- [ ] **Step 5: Implement splitting and ranking**

Use `langchain_text_splitters.MarkdownHeaderTextSplitter`. Defaults when config omits optional fields:

```python
headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
strip_headers = False
```

For each chunk, inherit parent metadata, set `doc_type="child"`, set `chunk_index`, compute `chunk_id` with:

```python
hashlib.md5(f"{parent_id}{chunk_index}".encode("utf-8")).hexdigest()
```

`get_ranked_parent_docs()` must count valid parent hits, remember the first chunk position, ignore missing parents with a warning, and return parents sorted by `(-hit_count, first_position)`.

- [ ] **Step 6: Run document preparation tests**

Run:

```bash
E:/007.agent/007.project/RAG/ENV/RAG_2026/python.exe -m pytest tests/unit/test_document_preparation.py -v
```

Expected: PASS.

- [ ] **Step 7: Request commit permission**

Ask the user before committing. If approved, run:

```bash
git add src/document_preparation tests/unit/test_document_preparation.py
git commit -m "feat: add markdown document preparation module"
```

---

### Task 4: FAISS 索引构建模块

**Files:**
- Create: `src/indexing/__init__.py`
- Create: `src/indexing/module.py`
- Test: `tests/unit/test_indexing.py`
- Test: `tests/integration/test_indexing_integration.py`

**Interfaces:**
- Consumes: `index_save_path: str | Path`, `embedding_config: EmbeddingConfig`, `chunks: list[Document]`
- Produces:
  - `IndexConstructionModule.vectorstore`
  - `IndexConstructionModule._load_embedding_model()`
  - `IndexConstructionModule.load_index() -> bool`
  - `IndexConstructionModule.build_index(chunks: list[Document]) -> None`
  - `IndexConstructionModule.save_index() -> None`

- [ ] **Step 1: Write failing unit tests for missing and incomplete index**

Create `tests/unit/test_indexing.py`:

```python
from pathlib import Path

from langchain_core.documents import Document

from src.config import EmbeddingConfig
from src.indexing.module import IndexConstructionModule


class DeterministicEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0, 0.0]


def make_embedding_config():
    return EmbeddingConfig.model_validate({"model_name": "fake-model"})


def test_load_index_returns_false_when_index_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(IndexConstructionModule, "_load_embedding_model", lambda self: DeterministicEmbeddings())
    module = IndexConstructionModule(tmp_path / "missing", make_embedding_config())

    assert module.load_index() is False


def test_load_index_returns_false_when_one_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(IndexConstructionModule, "_load_embedding_model", lambda self: DeterministicEmbeddings())
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "index.faiss").write_bytes(b"not-real")
    module = IndexConstructionModule(index_dir, make_embedding_config())

    assert module.load_index() is False
```

- [ ] **Step 2: Run indexing unit tests and verify they fail**

Run:

```bash
E:/007.agent/007.project/RAG/ENV/RAG_2026/python.exe -m pytest tests/unit/test_indexing.py -v
```

Expected: FAIL because indexing module is not implemented.

- [ ] **Step 3: Implement `IndexConstructionModule`**

Implementation requirements:

- `__init__` stores `Path(index_save_path)`, loads embeddings through `_load_embedding_model()`, and sets `self.vectorstore = None`.
- `_load_embedding_model()` uses `langchain_community.embeddings.HuggingFaceEmbeddings(model_name=self.embedding_config.model_name)`.
- If embedding loading fails, raise `IndexConstructionError` containing the model name and failure reason.
- `load_index()` checks both `index.faiss` and `index.pkl`; if either is missing, return `False`.
- Use LangChain FAISS with `FAISS.load_local(str(self.index_save_path), self.embedding_model, allow_dangerous_deserialization=True)` only for local index files created by this app.
- On load failure, print `索引加载失败，将重新构建索引。` and return `False`.
- `build_index(chunks)` calls `FAISS.from_documents(chunks, self.embedding_model)`.
- `save_index()` calls `self.vectorstore.save_local(str(self.index_save_path))`; save failure raises `IndexConstructionError` containing `index_save_path`.

- [ ] **Step 4: Add integration test for real FAISS persistence with fake embeddings**

Create `tests/integration/test_indexing_integration.py`:

```python
from langchain_core.documents import Document

from src.config import EmbeddingConfig
from src.indexing.module import IndexConstructionModule
from tests.unit.test_indexing import DeterministicEmbeddings


def test_build_save_and_load_faiss_index(tmp_path, monkeypatch):
    monkeypatch.setattr(IndexConstructionModule, "_load_embedding_model", lambda self: DeterministicEmbeddings())
    config = EmbeddingConfig.model_validate({"model_name": "fake-model"})
    chunks = [Document(page_content="红烧肉", metadata={"chunk_id": "c1", "parent_id": "p1"})]

    first = IndexConstructionModule(tmp_path / "faiss", config)
    first.build_index(chunks)
    first.save_index()

    assert (tmp_path / "faiss" / "index.faiss").is_file()
    assert (tmp_path / "faiss" / "index.pkl").is_file()

    second = IndexConstructionModule(tmp_path / "faiss", config)
    assert second.load_index() is True
    assert second.vectorstore is not None
```

- [ ] **Step 5: Run indexing tests**

Run:

```bash
E:/007.agent/007.project/RAG/ENV/RAG_2026/python.exe -m pytest tests/unit/test_indexing.py tests/integration/test_indexing_integration.py -v
```

Expected: PASS.

- [ ] **Step 6: Request commit permission**

Ask the user before committing. If approved, run:

```bash
git add src/indexing tests/unit/test_indexing.py tests/integration/test_indexing_integration.py
git commit -m "feat: add faiss index construction module"
```

---

### Task 5: 混合检索模块

**Files:**
- Create: `src/retrieval/__init__.py`
- Create: `src/retrieval/module.py`
- Test: `tests/unit/test_retrieval.py`

**Interfaces:**
- Consumes: `chunks: list[Document]`, `vectorstore`, `config: RetrievalConfig`
- Produces:
  - `RetrievalModule._init_retrievers() -> None`
  - `RetrievalModule.vector_search(query: str) -> list[Document]`
  - `RetrievalModule.bm25_search(query: str) -> list[Document]`
  - `RetrievalModule.hybrid_search(query: str) -> list[Document]`
  - `RetrievalModule._rrf_rerank(ranked_results: list[list[Document]]) -> list[Document]`

- [ ] **Step 1: Write failing RRF and hybrid tests**

Create `tests/unit/test_retrieval.py`:

```python
from langchain_core.documents import Document

from src.config import RetrievalConfig
from src.retrieval.module import RetrievalModule


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, query):
        return self.docs


class FakeVectorstore:
    def __init__(self, docs):
        self.docs = docs
        self.kwargs = None

    def as_retriever(self, search_type, search_kwargs):
        self.kwargs = {"search_type": search_type, "search_kwargs": search_kwargs}
        return FakeRetriever(self.docs)


def make_retrieval_config(vector_top_k=5, bm25_top_k=5, hybrid_top_k=3):
    return RetrievalConfig.model_validate(
        {
            "vector": {"search_type": "similarity", "top_k": vector_top_k},
            "bm25": {"top_k": bm25_top_k},
            "hybrid": {"top_k": hybrid_top_k},
        }
    )


def test_vector_search_uses_configured_top_k():
    docs = [Document(page_content="红烧肉", metadata={"chunk_id": "a"})]
    vectorstore = FakeVectorstore(docs)
    module = RetrievalModule(docs, vectorstore, make_retrieval_config(vector_top_k=5))

    assert module.vector_search("红烧肉") == docs
    assert vectorstore.kwargs["search_type"] == "similarity"
    assert vectorstore.kwargs["search_kwargs"] == {"k": 5}


def test_rrf_deduplicates_and_orders_by_score():
    a = Document(page_content="a", metadata={"chunk_id": "a"})
    b = Document(page_content="b", metadata={"chunk_id": "b"})
    c = Document(page_content="c", metadata={"chunk_id": "c"})
    module = RetrievalModule([a, b, c], FakeVectorstore([a, b]), make_retrieval_config())

    ranked = module._rrf_rerank([[a, b], [b, c]])

    assert [doc.metadata["chunk_id"] for doc in ranked] == ["b", "a", "c"]
```

- [ ] **Step 2: Run retrieval tests and verify they fail**

Run:

```bash
E:/007.agent/007.project/RAG/ENV/RAG_2026/python.exe -m pytest tests/unit/test_retrieval.py -v
```

Expected: FAIL because retrieval module is not implemented.

- [ ] **Step 3: Implement retrieval module**

Implementation requirements:

- Use `vectorstore.as_retriever(search_type=config.vector.search_type, search_kwargs={"k": config.vector.top_k})`.
- Use `langchain_community.retrievers.BM25Retriever.from_documents(chunks, preprocess_func=jieba.lcut)`.
- Set `self.bm25_retriever.k = config.bm25.top_k`.
- `hybrid_search()` calls vector search and BM25 search, passes both ranked lists to `_rrf_rerank()`, and returns `config.hybrid.top_k` documents.
- RRF uses a code constant `RRF_K = 60`, rank starts at 1, and document identity is `metadata["chunk_id"]` when present, otherwise fallback to `page_content`.
- Preserve first seen document object for each deduplicated key.

- [ ] **Step 4: Add tests for empty single route and top-k truncation**

Add:

```python
def test_hybrid_search_uses_available_route_when_other_is_empty(monkeypatch):
    a = Document(page_content="红烧肉", metadata={"chunk_id": "a"})
    module = RetrievalModule([a], FakeVectorstore([]), make_retrieval_config(hybrid_top_k=1))
    monkeypatch.setattr(module, "bm25_search", lambda query: [a])

    assert module.hybrid_search("红烧肉") == [a]


def test_hybrid_search_returns_empty_when_both_routes_empty(monkeypatch):
    module = RetrievalModule([], FakeVectorstore([]), make_retrieval_config())
    monkeypatch.setattr(module, "bm25_search", lambda query: [])

    assert module.hybrid_search("什么都没有") == []
```

- [ ] **Step 5: Run retrieval tests**

Run:

```bash
E:/007.agent/007.project/RAG/ENV/RAG_2026/python.exe -m pytest tests/unit/test_retrieval.py -v
```

Expected: PASS.

- [ ] **Step 6: Request commit permission**

Ask the user before committing. If approved, run:

```bash
git add src/retrieval tests/unit/test_retrieval.py
git commit -m "feat: add hybrid retrieval module"
```

---

### Task 6: GenerationModule 路由、改写、Context 与回答生成

**Files:**
- Modify: `src/generation/module.py`
- Test: `tests/unit/test_generation_module.py`

**Interfaces:**
- Consumes: fake or real LangChain chat model compatible with `.invoke(messages)` and `.stream(messages)`
- Produces:
  - `route_query(question: str) -> Literal["list", "detail", "general"]`
  - `rewrite_query(question: str) -> str`
  - `build_context(docs: list[Document]) -> str`
  - `generate_answer(question: str, intent: Literal["list", "detail", "general"], context: str, stream: bool, docs: list[Document] | None = None) -> str`

- [ ] **Step 1: Write failing behavior tests with fake LangChain chat model**

Add to `tests/unit/test_generation_module.py`:

```python
from langchain_core.documents import Document


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeLlm:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def invoke(self, messages):
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return FakeMessage(value)

    def stream(self, messages):
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        for part in value:
            yield FakeMessage(part)


def test_route_query_reads_json_intent(tmp_path):
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm(['{"intent": "list", "reason": "推荐"}'])

    assert module.route_query("推荐几个菜") == "list"


def test_route_query_defaults_general_on_invalid_json(tmp_path):
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm(["不是 JSON"])

    assert module.route_query("随便问") == "general"


def test_rewrite_query_falls_back_to_original_question(tmp_path, capsys):
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm([RuntimeError("network")])

    assert module.rewrite_query("原问题") == "原问题"
    assert "查询优化失败" in capsys.readouterr().out


def test_build_context_uses_ranked_parent_documents(tmp_path):
    module = GenerationModule(make_generation_config(tmp_path))
    docs = [
        Document(
            page_content="制作步骤",
            metadata={
                "dish_name": "红烧肉",
                "category": "荤菜",
                "difficulty": "中等",
                "source": "meat_dish/红烧肉.md",
            },
        )
    ]

    context = module.build_context(docs)

    assert "[文档1]" in context
    assert "菜名：红烧肉" in context
    assert "类别：荤菜" in context
    assert "难度：中等" in context
    assert "来源：meat_dish/红烧肉.md" in context
    assert "内容：\n制作步骤" in context


def test_generate_answer_selects_prompt_and_returns_content(tmp_path):
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm(["回答"])

    assert module.generate_answer("问题", "detail", "上下文", stream=False) == "回答"


def test_generate_answer_formats_list_without_calling_llm(tmp_path):
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm([RuntimeError("list should not call llm")])
    docs = [
        Document(page_content="a", metadata={"dish_name": "西红柿炒鸡蛋"}),
        Document(page_content="b", metadata={"dish_name": "凉拌黄瓜"}),
        Document(page_content="c", metadata={"dish_name": "紫菜蛋花汤"}),
    ]

    assert module.generate_answer("推荐几个菜", "list", "上下文", stream=False, docs=docs) == (
        "为您推荐以下菜品：\n"
        "1. 西红柿炒鸡蛋\n"
        "2. 凉拌黄瓜\n"
        "3. 紫菜蛋花汤"
    )
```

- [ ] **Step 2: Run generation behavior tests and verify they fail**

Run:

```bash
E:/007.agent/007.project/RAG/ENV/RAG_2026/python.exe -m pytest tests/unit/test_generation_module.py -v
```

Expected: FAIL until behavior is implemented.

- [ ] **Step 3: Implement route and rewrite**

Implementation requirements:

- Use `self.prompts["query_router"].format_messages(query=question)` and `self.llm.invoke(messages)`.
- Parse `json.loads(message.content)` and read only `intent`.
- Valid intents are `{"list", "detail", "general"}`.
- JSON parse failure, empty intent, non-string intent, illegal intent, or LLM failure returns `"general"`; LLM failure prints a short message.
- `rewrite_query()` uses `query_rewrite` prompt with `format_messages(query=question)` and returns raw `message.content`.
- `rewrite_query()` failure prints `查询优化失败` and returns the original question.

- [ ] **Step 4: Implement context construction**

`build_context(docs)` builds parent-document context in this exact shape and caps total text to `MAX_CONTEXT_CHARS = 6000`:

```text
[文档1]
菜名：红烧肉
类别：荤菜
难度：中等
来源：meat_dish/红烧肉.md
内容：
制作步骤正文
```

Use metadata defaults `未知` for missing dish/category/difficulty and empty string for missing source. Preserve input doc order and truncate only after assembling in order.

- [ ] **Step 5: Implement answer generation**

Implementation requirements:

- `list` intent does not call LLM and does not use any answer Prompt. It formats parent docs directly with `dish_name = doc.metadata.get("dish_name", "未知菜品")` and returns:

```text
为您推荐以下菜品：
1. 西红柿炒鸡蛋
2. 凉拌黄瓜
3. 紫菜蛋花汤
```

- Prompt selection for LLM-backed answers: `detail` -> `step_by_step_answer` prompt, all other non-list values -> `basic_answer` prompt.
- Non-stream mode uses `self.llm.invoke(messages)` and returns `message.content`.
- Stream mode uses `self.llm.stream(messages)`, prints each `chunk.content` with `end=""` and `flush=True`, concatenates all pieces, prints a newline after success, and returns the concatenated answer.
- Any non-stream LLM failure returns `LLM API 调用失败，请检查 API Key、base_url、模型名或网络连接`.
- Stream failure preserves already printed fragments, appends and prints `回答生成失败，请重试或检查配置/网络。`, and returns accumulated text plus that message.
- Prompt variable errors are not swallowed; let LangChain raise them.

- [ ] **Step 6: Run generation tests**

Run:

```bash
E:/007.agent/007.project/RAG/ENV/RAG_2026/python.exe -m pytest tests/unit/test_generation_module.py -v
```

Expected: PASS.

- [ ] **Step 7: Request commit permission**

Ask the user before committing. If approved, run:

```bash
git add src/generation/module.py tests/unit/test_generation_module.py
git commit -m "feat: implement generation module behavior"
```

---

### Task 7: RagService 生命周期与 CLI

**Files:**
- Create: `src/service.py`
- Create: `src/__main__.py`
- Test: `tests/unit/test_service.py`
- Test: `tests/e2e/test_cli.py`

**Interfaces:**
- Consumes:
  - `load_config() -> AppConfig`
  - `DocumentPreparationModule`
  - `IndexConstructionModule`
  - `GenerationModule`
  - `RetrievalModule`
- Produces:
  - `RagService.startup() -> None`
  - `RagService.build_knowledge_base() -> None`
  - `RagService.run_interactive() -> None`
  - `RagService.ask(question: str) -> str`
  - `RagService.shutdown() -> None`

- [ ] **Step 1: Write failing service orchestration tests**

Create `tests/unit/test_service.py`:

```python
from langchain_core.documents import Document

from src.service import RagService


class FakeDocs:
    def __init__(self):
        self.documents = []
        self.chunks = [Document(page_content="chunk", metadata={"parent_id": "p", "chunk_id": "c"})]
        self.parent = Document(page_content="parent", metadata={"parent_id": "p"})

    def load_documents(self):
        self.documents = [self.parent]
        return self.documents

    def split_documents(self):
        return self.chunks

    def get_ranked_parent_docs(self, chunks):
        return [self.parent] if chunks else []


class FakeIndex:
    def __init__(self, load_result=False):
        self.vectorstore = object()
        self.load_result = load_result
        self.built = False
        self.saved = False

    def load_index(self):
        return self.load_result

    def build_index(self, chunks):
        self.built = True

    def save_index(self):
        self.saved = True


class FakeRetrieval:
    def hybrid_search(self, query):
        return [Document(page_content="chunk", metadata={"parent_id": "p", "chunk_id": "c"})]


class FakeGeneration:
    def __init__(self):
        self.rewritten = []

    def route_query(self, question):
        return "detail"

    def rewrite_query(self, question):
        self.rewritten.append(question)
        return "优化问题"

    def build_context(self, docs):
        return "上下文"

    def generate_answer(self, question, intent, context, stream, docs=None):
        return f"{intent}:{context}:{stream}:{len(docs or [])}"


def test_build_knowledge_base_rebuilds_when_index_missing(capsys):
    service = RagService.__new__(RagService)
    service.document_module = FakeDocs()
    service.index_module = FakeIndex(load_result=False)

    service.build_knowledge_base()

    assert service.index_module.built is True
    assert service.index_module.saved is True
    assert "构建知识库完成" in capsys.readouterr().out


def test_ask_runs_single_turn_without_terminal_input():
    service = RagService.__new__(RagService)
    service.generation_module = FakeGeneration()
    service.retrieval_module = FakeRetrieval()
    service.document_module = FakeDocs()
    service.config = type("Config", (), {"generation": type("Gen", (), {"stream": False})()})()

    assert service.ask("怎么做红烧肉") == "detail:上下文:False:1"
```

- [ ] **Step 2: Run service tests and verify they fail**

Run:

```bash
E:/007.agent/007.project/RAG/ENV/RAG_2026/python.exe -m pytest tests/unit/test_service.py -v
```

Expected: FAIL because `src/service.py` is not implemented.

- [ ] **Step 3: Implement `RagService`**

Implementation requirements:

- `startup()` calls `load_config()`, initializes `DocumentPreparationModule(config.data_path, config.splitter)`, `IndexConstructionModule(config.index_save_path, config.embedding)`, `GenerationModule(config.generation)`, calls `generation_module.setup_llm()`, calls `build_knowledge_base()`, then initializes `RetrievalModule(document_module.chunks, index_module.vectorstore, config.retrieval)`.
- `build_knowledge_base()` calls document load, split, index load; if index load returns `False`, calls build and save; prints `构建知识库完成`.
- `ask(question)` handles one question and never calls `input()`.
- `ask()` route flow: `route_query(question)`; for `list` use original question; for `detail` and `general` use `rewrite_query(question)`.
- If retrieval returns no chunks, return `未检索到相关内容。`.
- If parent docs are empty after ranking, return `未检索到相关内容。`.
- After parent docs are ranked, call `generation_module.build_context(parent_docs)`, then call `generation_module.generate_answer(question, intent, context, config.generation.stream, docs=parent_docs)`. `RagService` must not implement list-result formatting itself.
- `run_interactive()` prints `您的问题是：`, reads input, treats empty input as retry, exits on `exit`、`quit`、`q`, prints answer, and continues.
- `shutdown()` exists and returns `None`.

- [ ] **Step 4: Implement CLI entry**

Create `src/__main__.py`:

```python
from src.service import RagService


def main() -> None:
    service = RagService()
    service.startup()
    service.run_interactive()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Add E2E CLI tests with monkeypatch**

Create `tests/e2e/test_cli.py`:

```python
from src.service import RagService


def test_run_interactive_reprompts_empty_input_then_exits(monkeypatch, capsys):
    service = RagService.__new__(RagService)
    service.ask = lambda question: f"answer:{question}"
    inputs = iter(["", "红烧肉怎么做", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    service.run_interactive()

    output = capsys.readouterr().out
    assert "您的问题是：" in output
    assert "请输入问题" in output
    assert "answer:红烧肉怎么做" in output
```

- [ ] **Step 6: Run service and CLI tests**

Run:

```bash
E:/007.agent/007.project/RAG/ENV/RAG_2026/python.exe -m pytest tests/unit/test_service.py tests/e2e/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 7: Request commit permission**

Ask the user before committing. If approved, run:

```bash
git add src/service.py src/__main__.py tests/unit/test_service.py tests/e2e/test_cli.py
git commit -m "feat: add rag service and cli entry"
```

---

### Task 8: 主链路集成测试与回归测试

**Files:**
- Create/Modify: `tests/fixtures/cook_docs/`
- Create: `tests/integration/test_rag_service_integration.py`
- Create/Modify: `tests/regression/test_regressions.py`

**Interfaces:**
- Consumes: all modules from Tasks 1-7
- Produces: proof that startup, index rebuild/load, BM25 rebuild, retrieval-empty short-circuit, parent traceback, and mocked LLM calls work without real API access.

- [ ] **Step 1: Create stable Markdown fixtures**

Create:

```text
tests/fixtures/cook_docs/meat_dish/红烧肉.md
tests/fixtures/cook_docs/vegetable_dish/清炒菜心.md
tests/fixtures/cook_docs/soup/番茄蛋汤.md
```

Use this for `红烧肉.md`:

```markdown
# 红烧肉

难度：★★★

## 食材

五花肉、冰糖、生抽、老抽。

## 做法

先煸出油脂，再炒糖色，加水炖煮至软糯。
```

- [ ] **Step 2: Write integration test for knowledge base build and reload**

Use a temporary config, temporary `.env`, deterministic embeddings, and fake LLM. Assert first startup creates `index.faiss` and `index.pkl`; second startup loads existing index; `ask("红烧肉怎么做")` returns fake answer; no real LLM API call is made.

- [ ] **Step 3: Write regression tests**

Cover these fixed behaviors:

```python
def test_invalid_route_json_defaults_to_general(tmp_path):
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm(["不是 JSON"])

    assert module.route_query("随便问") == "general"


def test_illegal_route_intent_defaults_to_general(tmp_path):
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm(['{"intent": "unknown"}'])

    assert module.route_query("推荐几个菜") == "general"


def test_empty_retrieval_returns_not_found_without_generation():
    service = RagService.__new__(RagService)
    service.generation_module = FakeGeneration()
    service.retrieval_module = type("Retrieval", (), {"hybrid_search": lambda self, query: []})()
    service.document_module = FakeDocs()
    service.config = type("Config", (), {"generation": type("Gen", (), {"stream": False})()})()

    assert service.ask("不存在的菜") == "未检索到相关内容。"


def test_missing_parent_chunk_is_ignored_and_warns(tmp_path, capsys):
    module = DocumentPreparationModule(tmp_path, make_splitter_config())
    module.documents = []

    ranked = module.get_ranked_parent_docs(
        [Document(page_content="orphan", metadata={"parent_id": "missing", "chunk_id": "c1"})]
    )

    assert ranked == []
    assert "无法回溯父文档" in capsys.readouterr().out


def test_existing_faiss_index_load_skips_rebuild_even_if_source_changes():
    service = RagService.__new__(RagService)
    service.document_module = FakeDocs()
    service.index_module = FakeIndex(load_result=True)

    service.build_knowledge_base()

    assert service.index_module.built is False
    assert service.index_module.saved is False
```

Each regression test should include a concrete fixture or fake object and one behavioral assertion matching the function name.

- [ ] **Step 4: Run integration and regression tests**

Run:

```bash
E:/007.agent/007.project/RAG/ENV/RAG_2026/python.exe -m pytest tests/integration tests/regression -v
```

Expected: PASS.

- [ ] **Step 5: Request commit permission**

Ask the user before committing. If approved, run:

```bash
git add tests/fixtures tests/integration tests/regression
git commit -m "test: cover rag service integration and regressions"
```

---

### Task 9: 本地质量门禁与人工验收准备

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`

**Interfaces:**
- Consumes: all implementation tasks
- Produces: runnable local verification and user-facing run instructions.

- [ ] **Step 1: Check `.gitignore` against spec**

Verify these patterns exist:

```gitignore
.env
storage/
*.faiss
*.pkl
__pycache__/
.pytest_cache/
RAG_2026/
ENV/RAG_2026/
```

Only modify `.gitignore` for missing entries.

- [ ] **Step 2: Add or update README run section**

Add:

```markdown
## 运行

1. 复制 `config.example.yaml` 为 `config.yaml`。
2. 复制 `.env.example` 为 `.env`，填写 `DEEPSEEK_API_KEY`。
3. 安装依赖：`pip install -r requirements.txt`。
4. 启动：`python -m src`。

第二版只支持本地 Markdown 文档。是否流式输出由 `config.yaml` 的 `generation.stream` 控制。
```

- [ ] **Step 3: Run full local test command**

Run:

```bash
E:/007.agent/007.project/RAG/ENV/RAG_2026/python.exe -m pytest tests -v --tb=short --cov=src --cov-report=term-missing --basetemp=.pytest_tmp
```

Expected: PASS.

- [ ] **Step 4: Run lint and format checks**

Run:

```bash
make lint
make format-check
```

Expected: PASS for both commands.

- [ ] **Step 5: Manual CLI smoke test**

After dependencies are installed and `.env` contains a valid key, run:

```bash
E:/007.agent/007.project/RAG/ENV/RAG_2026/python.exe -m src
```

Expected:

- startup prints `构建知识库完成`
- CLI displays `您的问题是：`
- empty input prompts again
- `exit`、`quit`、`q` exit
- LLM API failure returns `LLM API 调用失败，请检查 API Key、base_url、模型名或网络连接` and the program continues

- [ ] **Step 6: Request commit permission**

Ask the user before committing. If approved, run:

```bash
git add .gitignore README.md
git commit -m "docs: add v4 run and verification notes"
```

---

## Self-Review

- Spec coverage: Task 1 covers config, `.env`, template path validation, and dependency constraints. Task 2 covers four Prompt templates and LangChain Chat model setup. Task 3 covers Markdown-only loading, metadata enhancement, MD5 IDs, chunking, child-parent map, and parent traceback ordering. Task 4 covers FAISS index load/build/save lifecycle. Task 5 covers vector search, BM25 with jieba, hybrid search, and RRF. Task 6 covers route defaults, query rewriting fallback, parent-doc context, list intent fixed-format output without LLM, stream and non-stream answer generation, and LLM failure messages. Task 7 covers `RagService` startup, `ask(question)`, interactive CLI, no startup args, and empty/exit input behavior. Task 8 covers integration and regression cases without real LLM calls. Task 9 covers `.gitignore`, README, full pytest, lint, format, and manual CLI smoke testing.
- Placeholder scan: no unresolved implementation markers are left; every task has files, interfaces, concrete tests, commands, expected outcomes, and commit protocol.
- Type consistency: method and property names match `spec-v4.md`: `RagService.startup()`、`build_knowledge_base()`、`run_interactive()`、`ask(question)`、`shutdown()`；`DocumentPreparationModule.documents/chunks/child_parent_map`；`IndexConstructionModule.vectorstore/load_index/build_index/save_index`；`RetrievalModule.vector_search/bm25_search/hybrid_search/_rrf_rerank`；`GenerationModule.route_query/rewrite_query/build_context/generate_answer`。
