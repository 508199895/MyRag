# 本地 Markdown 文档问答工具 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 `spec-v3.md` 实现第二版本地 Markdown RAG 主链路：加载 Markdown、构建/加载 FAISS、混合检索、按意图生成回答，并通过 `python -m src` 进入连续问答。

**Architecture:** 项目采用“CLI 入口 + `RagService` 生命周期编排 + 文档准备/索引构建/检索/生成模块协作”。`RagService` 只编排生命周期和单轮问答，具体业务逻辑下沉到各模块；外层交互循环与 `ask(question)` 单轮问答保持分离，方便测试和后续 API/Web 复用。

**Tech Stack:** Python 3.13、LangChain、LangChain Community、FAISS、HuggingFace/sentence-transformers、OpenAI-compatible Chat API 的 `openai` 包接口、python-dotenv、PyYAML、pydantic、jieba、pytest、ruff。

**Spec:** `docs/superpowers/specs/spec-v3.md`

## Global Constraints

- 始终使用简体中文沟通与编写 Markdown 正文。
- 第二版只支持本地 Markdown 文档，不支持 TXT、PDF、网页抓取、图片、多模态文档。
- 第二版不实现评测数据集加载、检索评测、生成评测、Web/API 服务、多轮对话记忆、索引过期检测、增量索引、debug 模式或日志系统。
- 启动命令只有 `python -m src`；CLI 不提供 `-s`、`--debug`、`--rebuild` 等启动参数。
- 流式输出由 `config.yaml` 的 `generation.stream` 控制。
- 自动化测试不得真实调用 LLM API；查询路由、查询优化和最终回答生成都应使用 fake/mock client。
- 新增直接依赖必须写入 `requirements.txt`，记录实际安装和验证使用的版本号，不使用 `pip freeze` 生成整棵依赖树。
- 实施阶段如需实际执行依赖安装命令，必须先向用户请求权限。
- Git 提交前必须明确向用户请求权限，并在用户同意后再执行 `git commit`。
- 不提交或泄露 `.env`、API Key、FAISS 索引产物、虚拟环境、缓存目录。
- 文档 ID 与 chunk ID 使用 Python 标准库 `hashlib` 中的 MD5 哈希算法。
- OpenAI-compatible LLM 客户端使用 `openai` 包提供的接口初始化。

---

## File Structure

- Modify: `src/config.py`。更新配置模型，使其匹配 `spec-v3.md` 的 `retrieval.vector.search_type`、`generation.stream` 和五个 Prompt 模板路径。
- Modify: `config.example.yaml`。同步 `spec-v3.md` 推荐配置结构。
- Create/Modify: `.env.example`。提供 `DEEPSEEK_API_KEY=your-api-key`。
- Modify: `requirements.txt`。记录项目直接依赖及验证版本。
- Create: `docs/prompts/query_router.md`、`docs/prompts/query_rewrite.md`、`docs/prompts/list_answer.md`、`docs/prompts/detail_answer.md`、`docs/prompts/general_answer.md`。提供五类 Prompt 模板。
- Create: `src/document_preparation/__init__.py`、`src/document_preparation/module.py`。实现 `DocumentPreparationModule`。
- Create: `src/indexing/__init__.py`、`src/indexing/module.py`。实现 `IndexConstructionModule`。
- Create: `src/retrieval/__init__.py`、`src/retrieval/module.py`。实现 `RetrievalModule`。
- Create: `src/generation/module.py`。实现 `GenerationModule`。
- Keep or deprecate internally: `src/generation/prompts.py`。若继续保留，只作为兼容旧测试的辅助；新主链路使用 `GenerationModule`。
- Create: `src/service.py`。实现 `RagService`。
- Create: `src/__main__.py`。实现 `python -m src` 入口。
- Create/Modify tests under `tests/unit/`、`tests/integration/`、`tests/e2e/`、`tests/regression/`、`tests/fixtures/`。

---

### Task 1: 配置契约、模板和依赖清单

**Files:**
- Modify: `src/config.py`
- Modify: `config.example.yaml`
- Create/Modify: `.env.example`
- Modify: `requirements.txt`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: existing `load_config(config_path: str | Path = "config.yaml", env_path: str | Path = ".env") -> AppConfig`
- Produces:
  - `AppConfig.data_path: str`
  - `AppConfig.index_save_path: str`
  - `AppConfig.splitter: MarkdownHeaderSplitterConfig`
  - `AppConfig.embedding: EmbeddingConfig`
  - `AppConfig.retrieval: RetrievalConfig`
  - `AppConfig.retrieval.vector.search_type: str`
  - `AppConfig.retrieval.vector.top_k: int`
  - `AppConfig.retrieval.bm25.top_k: int`
  - `AppConfig.retrieval.hybrid.top_k: int`
  - `AppConfig.generation.stream: bool`
  - `AppConfig.generation.query_router_prompt_template_path: str`
  - `AppConfig.generation.query_rewrite_prompt_template_path: str`
  - `AppConfig.generation.list_prompt_template_path: str`
  - `AppConfig.generation.detail_prompt_template_path: str`
  - `AppConfig.generation.general_prompt_template_path: str`

- [ ] **Step 1: Write failing tests for the v3 config shape**

Add or replace test data in `tests/unit/test_config.py`:

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
  list_prompt_template_path: docs/prompts/list_answer.md
  detail_prompt_template_path: docs/prompts/detail_answer.md
  general_prompt_template_path: docs/prompts/general_answer.md
"""
```

Add assertions:

```python
def test_load_config_returns_v3_typed_app_config(tmp_path, monkeypatch):
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
    assert config.generation.list_prompt_template_path == "docs/prompts/list_answer.md"
    assert config.generation.detail_prompt_template_path == "docs/prompts/detail_answer.md"
    assert config.generation.general_prompt_template_path == "docs/prompts/general_answer.md"
```

- [ ] **Step 2: Run config tests and verify they fail for the old shape**

Run:

```bash
python -m pytest tests/unit/test_config.py -v
```

Expected: FAIL because current `RetrievalConfig` still expects `type`, and `GenerationConfig` still expects `prompt_template_path` instead of the v3 Prompt fields.

- [ ] **Step 3: Update `src/config.py` models**

Implement these model classes:

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
    list_prompt_template_path: str
    detail_prompt_template_path: str
    general_prompt_template_path: str
```

Keep top-level `AppConfig` as the single returned object. Keep required-field and type-only validation behavior. Keep unknown fields allowed in config sections.

- [ ] **Step 4: Sync templates and dependency list**

Update `config.example.yaml` to exactly match the v3 recommended structure. Create `.env.example` if absent:

```env
DEEPSEEK_API_KEY=your-api-key
```

Update `requirements.txt` after dependency installation/version verification. The direct dependencies expected by v3 are:

```text
python-dotenv
PyYAML
pydantic
pytest
pytest-cov
ruff
langchain
langchain-community
faiss-cpu
sentence-transformers
openai
jieba
```

Before installing or downloading packages, ask the user for permission. After installation, pin the verified direct dependency versions in `requirements.txt`.

- [ ] **Step 5: Run config tests and verify they pass**

Run:

```bash
python -m pytest tests/unit/test_config.py -v
```

Expected: PASS.

- [ ] **Step 6: Request commit permission**

Ask the user before committing. If approved, run:

```bash
git add src/config.py config.example.yaml .env.example requirements.txt tests/unit/test_config.py
git commit -m "chore: align config contract with spec v3"
```

---

### Task 2: Prompt 模板文件与生成模块基础契约

**Files:**
- Create: `docs/prompts/query_router.md`
- Create: `docs/prompts/query_rewrite.md`
- Create: `docs/prompts/list_answer.md`
- Create: `docs/prompts/detail_answer.md`
- Create: `docs/prompts/general_answer.md`
- Create: `src/generation/module.py`
- Modify: `src/generation/__init__.py`
- Test: `tests/unit/test_generation_module.py`

**Interfaces:**
- Consumes: `GenerationConfig`
- Produces:
  - `GenerationModule.__init__(config) -> None`
  - `GenerationModule.setup_llm() -> None`
  - `GenerationModule.route_query(question: str) -> Literal["list", "detail", "general"]`
  - `GenerationModule.rewrite_query(question: str) -> str`
  - `GenerationModule.build_context(docs: list[Document]) -> str`
  - `GenerationModule.generate_answer(question: str, intent: Literal["list", "detail", "general"], context: str, stream: bool) -> str`

- [ ] **Step 1: Write failing tests for prompt existence and path checking**

Create `tests/unit/test_generation_module.py`:

```python
from pathlib import Path

import pytest

from src.config import GenerationConfig
from src.generation.module import GenerationModule, GenerationModuleError


def make_generation_config(tmp_path: Path, **overrides) -> GenerationConfig:
    paths = {}
    for name in ["query_router", "query_rewrite", "list", "detail", "general"]:
        prompt_path = tmp_path / f"{name}.md"
        prompt_path.write_text("问题：{question}\n资料：{context}", encoding="utf-8")
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
        "list_prompt_template_path": paths["list"],
        "detail_prompt_template_path": paths["detail"],
        "general_prompt_template_path": paths["general"],
    }
    data.update(overrides)
    return GenerationConfig.model_validate(data)


def test_generation_module_checks_prompt_paths(tmp_path):
    missing = tmp_path / "missing.md"
    config = make_generation_config(
        tmp_path,
        list_prompt_template_path=str(missing),
    )

    with pytest.raises(GenerationModuleError) as exc_info:
        GenerationModule(config)

    assert str(missing) in str(exc_info.value)
```

- [ ] **Step 2: Run the new generation test and verify it fails**

Run:

```bash
python -m pytest tests/unit/test_generation_module.py::test_generation_module_checks_prompt_paths -v
```

Expected: FAIL because `src/generation/module.py` does not exist.

- [ ] **Step 3: Create the five Prompt templates**

Use these required variables:

```text
query_router.md: {question}
query_rewrite.md: {question}
list_answer.md: {question}, {context}
detail_answer.md: {question}, {context}
general_answer.md: {question}, {context}
```

`query_router.md` must instruct the model to output JSON containing only an `intent` value from `list`、`detail`、`general`.

- [ ] **Step 4: Implement `GenerationModule` initialization and prompt loading**

Implement a `GenerationModuleError` and load all configured prompt paths during `__init__`. For this task, `setup_llm()` can initialize the client later, but the method must exist and be callable.

```python
class GenerationModuleError(Exception):
    """Raised when generation module setup or execution fails."""
```

- [ ] **Step 5: Run prompt/generation tests**

Run:

```bash
python -m pytest tests/unit/test_generation_module.py tests/unit/test_prompts.py -v
```

Expected: PASS. If legacy `tests/unit/test_prompts.py` conflicts with v3, update it to check the five v3 prompt files instead of only `docs/prompts/llm_generator.md`.

- [ ] **Step 6: Request commit permission**

Ask the user before committing. If approved, run:

```bash
git add docs/prompts src/generation tests/unit/test_generation_module.py tests/unit/test_prompts.py
git commit -m "feat: add v3 prompt templates and generation module shell"
```

---

### Task 3: 文档准备模块

**Files:**
- Create: `src/document_preparation/__init__.py`
- Create: `src/document_preparation/module.py`
- Test: `tests/unit/test_document_preparation.py`
- Test fixtures: `tests/fixtures/cook_docs/`

**Interfaces:**
- Consumes: `data_path: str | Path`, `splitter_config: MarkdownHeaderSplitterConfig`
- Produces:
  - `DocumentPreparationModule.documents: list[Document]`
  - `DocumentPreparationModule.chunks: list[Document]`
  - `DocumentPreparationModule.child_parent_map: dict[str, str]`
  - `load_documents() -> list[Document]`
  - `split_documents() -> list[Document]`
  - `get_ranked_parent_docs(chunks: list[Document]) -> list[Document]`

- [ ] **Step 1: Write failing metadata and loading tests**

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


def test_load_documents_enhances_parent_metadata(tmp_path):
    doc_path = tmp_path / "meat_dish" / "红烧肉.md"
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text("# 红烧肉\n难度：★★★\n做法内容", encoding="utf-8")
    module = DocumentPreparationModule(tmp_path, make_splitter_config())

    docs = module.load_documents()

    assert len(docs) == 1
    metadata = docs[0].metadata
    assert metadata["doc_type"] == "parent"
    assert metadata["source"] == "meat_dish/红烧肉.md"
    assert metadata["category"] == "荤菜"
    assert metadata["dish_name"] == "红烧肉"
    assert metadata["difficulty"] == "中等"
    assert len(metadata["parent_id"]) == 32


def test_load_documents_rejects_missing_data_path(tmp_path):
    module = DocumentPreparationModule(tmp_path / "missing", make_splitter_config())

    with pytest.raises(DocumentPreparationError, match="路径不存在"):
        module.load_documents()
```

- [ ] **Step 2: Run loading tests and verify they fail**

Run:

```bash
python -m pytest tests/unit/test_document_preparation.py -v
```

Expected: FAIL because the module is not implemented.

- [ ] **Step 3: Implement Markdown loading and parent metadata**

Implementation requirements:

- Recursively load only `.md` files.
- Use UTF-8.
- Skip empty or whitespace-only Markdown files and print a warning containing the path.
- Raise `DocumentPreparationError("路径不存在：...")` if `data_path` does not exist.
- Raise `DocumentPreparationError("文档库为空")` if no Markdown files exist.
- Compute `parent_id` with `hashlib.md5(source.encode("utf-8")).hexdigest()`.
- Normalize `source` as POSIX relative path.

- [ ] **Step 4: Write failing chunk and parent ranking tests**

Add tests:

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


def test_get_ranked_parent_docs_orders_by_hit_count_then_first_position(tmp_path):
    module = DocumentPreparationModule(tmp_path, make_splitter_config())
    parent_a = Document(page_content="A", metadata={"parent_id": "a", "dish_name": "A"})
    parent_b = Document(page_content="B", metadata={"parent_id": "b", "dish_name": "B"})
    module.documents = [parent_a, parent_b]
    chunks = [
        Document(page_content="b1", metadata={"parent_id": "b", "chunk_id": "b1"}),
        Document(page_content="a1", metadata={"parent_id": "a", "chunk_id": "a1"}),
        Document(page_content="a2", metadata={"parent_id": "a", "chunk_id": "a2"}),
    ]

    ranked = module.get_ranked_parent_docs(chunks)

    assert ranked == [parent_a, parent_b]
```

- [ ] **Step 5: Implement splitting and ranking**

Use LangChain Markdown header splitter. Use parent metadata inheritance. Compute `chunk_id` with `hashlib.md5(f"{parent_id}{chunk_index}".encode("utf-8")).hexdigest()`.

- [ ] **Step 6: Run document preparation tests**

Run:

```bash
python -m pytest tests/unit/test_document_preparation.py -v
```

Expected: PASS.

- [ ] **Step 7: Request commit permission**

Ask the user before committing. If approved, run:

```bash
git add src/document_preparation tests/unit/test_document_preparation.py tests/fixtures
git commit -m "feat: add markdown document preparation module"
```

---

### Task 4: 索引构建模块

**Files:**
- Create: `src/indexing/__init__.py`
- Create: `src/indexing/module.py`
- Test: `tests/unit/test_indexing.py`
- Test: `tests/integration/test_indexing_integration.py`

**Interfaces:**
- Consumes: `index_save_path: str | Path`, `embedding_config: EmbeddingConfig`, `chunks: list[Document]`
- Produces:
  - `IndexConstructionModule.vectorstore`
  - `IndexConstructionModule.load_index() -> bool`
  - `IndexConstructionModule.build_index(chunks: list[Document]) -> None`
  - `IndexConstructionModule.save_index() -> None`

- [ ] **Step 1: Write failing unit tests with fake vectorstore**

Create `tests/unit/test_indexing.py` and monkeypatch FAISS/embedding loading:

```python
from langchain_core.documents import Document

from src.config import EmbeddingConfig
from src.indexing.module import IndexConstructionModule


class FakeVectorstore:
    def __init__(self):
        self.saved_path = None

    def save_local(self, path):
        self.saved_path = path


def test_load_index_returns_false_when_files_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(IndexConstructionModule, "_load_embedding_model", lambda self: object())
    module = IndexConstructionModule(tmp_path / "index", EmbeddingConfig(model_name="fake"))

    assert module.load_index() is False


def test_build_index_uses_passed_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(IndexConstructionModule, "_load_embedding_model", lambda self: object())
    captured = {}

    class FakeFAISS:
        @staticmethod
        def from_documents(chunks, embedding):
            captured["chunks"] = chunks
            captured["embedding"] = embedding
            return FakeVectorstore()

    monkeypatch.setattr("src.indexing.module.FAISS", FakeFAISS)
    module = IndexConstructionModule(tmp_path / "index", EmbeddingConfig(model_name="fake"))
    chunks = [Document(page_content="内容", metadata={"chunk_id": "c1"})]

    module.build_index(chunks)

    assert captured["chunks"] == chunks
    assert module.vectorstore is not None
```

- [ ] **Step 2: Run indexing unit tests and verify they fail**

Run:

```bash
python -m pytest tests/unit/test_indexing.py -v
```

Expected: FAIL because indexing module is not implemented.

- [ ] **Step 3: Implement `IndexConstructionModule`**

Implementation requirements:

- `__init__` stores `Path(index_save_path)`, embedding config, loads embedding model, sets `self.vectorstore = None`.
- `_load_embedding_model()` uses a LangChain HuggingFace embedding implementation backed by sentence-transformers.
- `load_index()` checks both `index.faiss` and `index.pkl`; missing files return `False`.
- Load failures print `索引加载失败，将重新构建索引。` and return `False`.
- `build_index(chunks)` calls `FAISS.from_documents(chunks, self.embedding_model)`.
- `save_index()` calls `self.vectorstore.save_local(str(self.index_save_path))`.
- Save failure raises an indexing error mentioning the index directory path.

- [ ] **Step 4: Add integration test for real FAISS persistence**

Use a tiny fake embedding class if sentence-transformers download would make the test brittle:

```python
class DeterministicEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0, 0.0]
```

Patch `_load_embedding_model` to return this class, then assert `index.faiss` and `index.pkl` are created and a second module can load them.

- [ ] **Step 5: Run indexing tests**

Run:

```bash
python -m pytest tests/unit/test_indexing.py tests/integration/test_indexing_integration.py -v
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
    docs = [Document(page_content="a", metadata={"chunk_id": "a"})]
    vectorstore = FakeVectorstore(docs)
    module = RetrievalModule(docs, vectorstore, make_retrieval_config(vector_top_k=5))

    assert module.vector_search("问题") == docs
    assert vectorstore.kwargs["search_type"] == "similarity"
    assert vectorstore.kwargs["search_kwargs"] == {"k": 5}


def test_rrf_deduplicates_and_orders_by_score():
    a = Document(page_content="a", metadata={"chunk_id": "a"})
    b = Document(page_content="b", metadata={"chunk_id": "b"})
    module = RetrievalModule([a, b], FakeVectorstore([a, b]), make_retrieval_config())

    ranked = module._rrf_rerank([[a, b], [b, a]])

    assert [doc.metadata["chunk_id"] for doc in ranked] == ["a", "b"]
```

- [ ] **Step 2: Run retrieval tests and verify they fail**

Run:

```bash
python -m pytest tests/unit/test_retrieval.py -v
```

Expected: FAIL because retrieval module is not implemented.

- [ ] **Step 3: Implement retrieval module**

Implementation requirements:

- Use `vectorstore.as_retriever(search_type=config.vector.search_type, search_kwargs={"k": config.vector.top_k})`.
- Use `langchain_community.retrievers.BM25Retriever`.
- Set BM25 `k` to `config.bm25.top_k`.
- Use `jieba.lcut` as BM25 preprocessing.
- `hybrid_search()` calls vector search and BM25 search, runs RRF, returns at most `config.hybrid.top_k`.
- RRF uses `score(d) = sum(1 / (RRF_K + rank_i(d)))`, where rank starts at 1 and `RRF_K` is a code constant.

- [ ] **Step 4: Run retrieval tests**

Run:

```bash
python -m pytest tests/unit/test_retrieval.py -v
```

Expected: PASS.

- [ ] **Step 5: Request commit permission**

Ask the user before committing. If approved, run:

```bash
git add src/retrieval tests/unit/test_retrieval.py
git commit -m "feat: add hybrid retrieval module"
```

---

### Task 6: 生成模块完整行为

**Files:**
- Modify: `src/generation/module.py`
- Test: `tests/unit/test_generation_module.py`

**Interfaces:**
- Consumes: fake or real `openai.OpenAI`-compatible client, prompt files, `Document` parents
- Produces:
  - `route_query()` defaults to `general` on invalid JSON, empty intent, illegal intent, or client failure.
  - `rewrite_query()` returns model text, or original question after printing `查询优化失败`.
  - `build_context()` formats parent docs and truncates to about 6000 Chinese characters.
  - `generate_answer()` returns LLM answer or configured failure message.

- [ ] **Step 1: Write failing route/rewrite/context tests**

Add tests:

```python
from langchain_core.documents import Document


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def create(self, **kwargs):
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return FakeResponse(value)


class FakeChat:
    def __init__(self, outputs):
        self.completions = FakeCompletions(outputs)


class FakeClient:
    def __init__(self, outputs):
        self.chat = FakeChat(outputs)


def test_route_query_reads_json_intent(tmp_path):
    module = GenerationModule(make_generation_config(tmp_path))
    module.client = FakeClient(['{"intent": "list", "reason": "推荐"}'])

    assert module.route_query("推荐几个菜") == "list"


def test_route_query_defaults_general_on_invalid_json(tmp_path):
    module = GenerationModule(make_generation_config(tmp_path))
    module.client = FakeClient(["不是 JSON"])

    assert module.route_query("随便问") == "general"


def test_rewrite_query_falls_back_to_original_question(tmp_path, capsys):
    module = GenerationModule(make_generation_config(tmp_path))
    module.client = FakeClient([RuntimeError("network")])

    assert module.rewrite_query("原问题") == "原问题"
    assert "查询优化失败" in capsys.readouterr().out


def test_build_context_uses_parent_documents(tmp_path):
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
    assert "内容：\n制作步骤" in context
```

- [ ] **Step 2: Run generation behavior tests and verify they fail**

Run:

```bash
python -m pytest tests/unit/test_generation_module.py -v
```

Expected: FAIL until `GenerationModule` has full behavior.

- [ ] **Step 3: Implement OpenAI-compatible client setup**

Use the `openai` package:

```python
from openai import OpenAI

self.client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
```

Use `client.chat.completions.create(...)` for routing, rewriting, and answer generation. Pass `model`、`temperature`、`max_tokens` from config as relevant.

- [ ] **Step 4: Implement route, rewrite, context, and answer methods**

Requirements:

- `route_query()` parses JSON and reads only `intent`.
- Any route parse or LLM failure returns `"general"` and prints information.
- `rewrite_query()` returns raw model content, no extra JSON parsing.
- `generate_answer()` chooses prompt by `intent`.
- Non-stream generation returns full message content.
- Stream generation prints chunks as they arrive and returns the concatenated answer.
- LLM API failure returns `LLM API 调用失败，请检查 API Key、base_url、模型名或网络连接`.
- Stream interruption preserves already printed fragments, appends `回答生成失败，请重试或检查配置/网络。`, then returns the accumulated text plus that message.

- [ ] **Step 5: Run generation tests**

Run:

```bash
python -m pytest tests/unit/test_generation_module.py -v
```

Expected: PASS.

- [ ] **Step 6: Request commit permission**

Ask the user before committing. If approved, run:

```bash
git add src/generation/module.py tests/unit/test_generation_module.py
git commit -m "feat: implement generation module behavior"
```

---

### Task 7: RagService 生命周期和 CLI

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
  - `RetrievalModule`
  - `GenerationModule`
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
    def route_query(self, question):
        return "detail"

    def rewrite_query(self, question):
        return "优化问题"

    def build_context(self, docs):
        return "上下文"

    def generate_answer(self, question, intent, context, stream):
        return f"{intent}:{context}"


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

    assert service.ask("怎么做红烧肉") == "detail:上下文"
```

- [ ] **Step 2: Run service tests and verify they fail**

Run:

```bash
python -m pytest tests/unit/test_service.py -v
```

Expected: FAIL because `src/service.py` is not implemented.

- [ ] **Step 3: Implement `RagService`**

Implementation requirements:

- `startup()` loads config and initializes document, index, generation, knowledge base, and retrieval modules in the order specified by `spec-v3.md`.
- `build_knowledge_base()` calls load, split, load index, build/save if needed, then prints `构建知识库完成`.
- `ask(question)` handles one question and never calls `input()`.
- If retrieval returns no chunks, return `未检索到相关内容。`.
- If parent docs are empty after ranking, return `未检索到相关内容。`.
- For `list`, search with original question.
- For `detail` and `general`, search with rewritten query.
- `run_interactive()` handles prompt, empty input, exit commands, and output.

- [ ] **Step 4: Implement CLI entry**

`src/__main__.py` should only create `RagService`, call `startup()`, then `run_interactive()`. Do not parse CLI business flags.

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

Use monkeypatch to replace `RagService.startup` and `RagService.ask`, then verify empty input does not exit and `exit` exits.

- [ ] **Step 6: Run service and CLI tests**

Run:

```bash
python -m pytest tests/unit/test_service.py tests/e2e/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 7: Request commit permission**

Ask the user before committing. If approved, run:

```bash
git add src/service.py src/__main__.py tests/unit/test_service.py tests/e2e/test_cli.py
git commit -m "feat: add rag service and cli entry"
```

---

### Task 8: 主链路集成测试和回归测试

**Files:**
- Create/Modify: `tests/fixtures/cook_docs/`
- Create: `tests/integration/test_rag_service_integration.py`
- Create/Modify: `tests/regression/test_regressions.py`

**Interfaces:**
- Consumes: all modules from Tasks 1-7
- Produces: source-backed proof that startup, index rebuild/load, retrieval-empty short-circuit, and LLM mocking work end-to-end without real API calls.

- [ ] **Step 1: Create stable Markdown fixtures**

Create fixture files:

```text
tests/fixtures/cook_docs/meat_dish/红烧肉.md
tests/fixtures/cook_docs/vegetable_dish/清炒菜心.md
tests/fixtures/cook_docs/soup/番茄蛋汤.md
```

Example content for `红烧肉.md`:

```markdown
# 红烧肉

难度：★★★

## 食材

五花肉、冰糖、生抽、老抽。

## 做法

先煸出油脂，再炒糖色，加水炖煮至软糯。
```

- [ ] **Step 2: Write integration test for complete service startup**

Use temp config, temp `.env`, deterministic embeddings, and fake generation client. Assert:

- first startup creates `index.faiss` and `index.pkl`
- second startup loads existing index
- `ask("红烧肉怎么做")` returns the fake answer
- no real LLM API call is made

- [ ] **Step 3: Write regression tests**

Cover these fixed behaviors:

- invalid route JSON defaults to `general`
- illegal intent defaults to `general`
- empty retrieval returns `未检索到相关内容。`
- chunk with missing parent is ignored and warns
- source Markdown changes do not force FAISS rebuild if existing index can load

- [ ] **Step 4: Run integration and regression tests**

Run:

```bash
python -m pytest tests/integration tests/regression -v
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
- Modify: `README.md` if it exists
- Create: `README.md` if absent
- Modify: `.gitignore` only if required entries are missing

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
```

Only modify `.gitignore` if one of these entries is absent.

- [ ] **Step 2: Add or update README run section**

Document:

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
python -m pytest tests -v --tb=short --cov=src --cov-report=term-missing --basetemp=.pytest_tmp
```

Expected: PASS.

- [ ] **Step 4: Run lint and format checks**

Run:

```bash
ruff check src tests
ruff format --check src tests
```

Expected: PASS for both commands.

- [ ] **Step 5: Manual CLI smoke test**

After dependencies are installed and `.env` contains a valid key, run:

```bash
python -m src
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
git add README.md .gitignore
git commit -m "docs: add v3 run and verification notes"
```

---

## Self-Review

- Spec coverage: Tasks 1-2 cover configuration, env, templates, dependency constraints, and OpenAI-compatible setup. Task 3 covers Markdown loading, metadata, MD5 IDs, splitting, and parent ranking. Task 4 covers FAISS lifecycle. Task 5 covers vector/BM25/RRF retrieval. Task 6 covers routing, query rewriting, context construction, and answer generation. Task 7 covers `RagService` and CLI behavior. Task 8 covers integration, E2E-adjacent behavior, and regressions. Task 9 covers local verification, README, and ignore rules.
- Placeholder scan: no placeholder sections are left for implementers; each task lists files, interfaces, test snippets, commands, expected results, and commit protocol.
- Type consistency: method and property names match `spec-v3.md`: `RagService.startup()`、`build_knowledge_base()`、`run_interactive()`、`ask(question)`、`shutdown()`；`DocumentPreparationModule.documents/chunks/child_parent_map`；`IndexConstructionModule.vectorstore/load_index/build_index/save_index`；`RetrievalModule.vector_search/bm25_search/hybrid_search/_rrf_rerank`；`GenerationModule.route_query/rewrite_query/build_context/generate_answer`。
