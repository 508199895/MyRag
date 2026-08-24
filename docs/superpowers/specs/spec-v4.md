# 本地 Markdown 文档问答工具 Spec

## 1. 背景与目标

本项目第二版要实现一个面向个人使用的本地 Markdown 文档问答工具。项目基于 LangChain、FAISS、本地 Embedding 与 OpenAI-compatible Chat API，围绕本地菜谱 Markdown 文档完成加载、索引、混合检索、意图路由和回答生成。

长期方向可以扩展到评测数据集、API 服务和 Web 界面，但第二版目标只覆盖本地 Markdown 文档问答。后续扩展方向不得反向污染第二版的模块职责和验收范围。

第二版完成后必须具备以下可验收能力：

- 执行 `python -m src` 后进入连续交互式问答。
- 启动时读取 `config.yaml` 与 `.env`，初始化文档准备、索引构建、生成和检索模块。
- 能递归加载 `data_path` 下的 Markdown 文档，enhance 父文档元数据，切分子 chunks，并建立父子映射。
- 能加载已有 LangChain FAISS 索引；索引不存在或加载失败时，基于当前 chunks 构建并保存索引。
- 每轮问题独立处理：路由意图、必要时优化查询、混合检索、回溯父文档、构建 context，并调用 OpenAI-compatible Chat API 生成回答。

功能目标：

- 只支持本地 Markdown 文档，不支持 TXT。
- 使用本地 Embedding 模型完成文本向量化，默认模型为 `BAAI/bge-small-zh-v1.5`。
- 使用 LangChain FAISS vectorstore 保存和加载向量索引。
- 使用向量检索 + BM25 检索 + RRF 的混合检索链路。
- 使用 OpenAI-compatible Chat API，默认配置指向 DeepSeek。
- 提供可复用的 `RagService`，让 CLI、后续 API/Web、测试和脚本都能复用同一套核心流程。

第二版非目标：

- 不支持 TXT、PDF、网页抓取、图片、多模态文档。
- 不支持 parquet/jsonl/tsv 数据集加载。
- 不实现评测数据集加载、检索评测或生成评测。
- 不提供检索评测或生成评测 CLI。
- 不实现 Web/API 服务。
- 不实现多轮对话记忆。
- 不做索引过期检测，不做增量索引，不提供 CLI 重建参数。
- 不实现 debug 模式。
- 不实现日志系统，仅保留普通终端提示。

## 2. 第二版实现范围

第二版只实现本地 Markdown 文档问答主链路。评测相关能力只在后续扩展方向中描述，第二版不创建可用的评测模块接口、不实现评测数据 loader，也不在 CLI 主流程中接入评测。

模块设计要优先服务当前本地菜谱 Markdown 场景，同时为后续数据格式差异较大的评测数据集保留清晰边界。文档准备模块需要使用类组织加载、元数据 enhance、切分和父文档回溯能力，避免同模块函数散乱，方便后期扩展和维护。

## 3. 整体架构

项目采用“交互入口 + `RagService` 生命周期编排 + 分模块协作”的结构。

启动命令只有一个：

```bash
python -m src
```

启动后进入连续交互式问答。用户不在命令行传入问题，CLI 也不提供 `-s`、`--debug`、`--rebuild` 等启动参数。流式输出由配置文件控制。

初始化阶段：

```text
系统启动
→ 加载配置
→ 初始化文件准备模块
   - 只加载文档库路径和 splitter 配置
→ 初始化索引构建模块
   - 加载 FAISS 保存路径和 embedding 配置
   - 加载 embedding 模型
   - 设置空 vectorstore
→ 初始化生成模块
   - 读取并保存 generation 配置
   - 调用 setup_llm()
   - 初始化可调用的 OpenAI-compatible LLM 客户端
→ RagService.build_knowledge_base()
   - 调用文档准备模块递归加载 Markdown
   - load_documents() 内部调用元数据 enhance
   - 切分 chunks
   - 加载索引：调用索引构建模块加载已有索引,索引不存在或加载失败时，基于 chunks 构建并保存索引
   - 打印“构建知识库完成”
→ 初始化检索模块
   - 获取文档准备模块的 chunks
   - 获取索引构建模块的 vectorstore
   - 启动向量检索器和 BM25 检索器
→ 系统就绪
```

单轮查询阶段：

```text
用户输入问题
→ 查询路由：LLM 输出 intent
→ 查询优化：detail/general 使用优化查询，list 使用原始问题
→ 混合检索：向量检索 + BM25 检索 + RRF 重排
→ 返回候选 chunks
→ 文档准备模块根据 chunk_id/parent_id 回溯父文档并去重排序
→ 生成模块 build_context(docs)
→ 根据 intent 选择 Prompt 模板
→ 调用 LLM 生成回答
→ 返回结果
```

每轮查询独立处理，不保留历史对话。

`RagService` 是长期运行入口，只负责生命周期和模块调用编排；具体业务细节下沉到文档准备、索引构建、检索和生成模块。
交互式问答循环和实际问答链路必须分离：外层循环只负责提示、读取输入、退出判断和输出结果；实际问答只负责处理单个问题。

## 4. 项目目录结构

目标结构：

```text
RAG/
├─ src/
│  ├─ __main__.py                     # 入口：python -m src
│  ├─ service.py                      # RagService 生命周期与查询编排
│  ├─ config.py                       # config.yaml + .env 读取与类型校验
│  ├─ document_preparation/
│  │  ├─ __init__.py
│  │  └─ module.py                    # DocumentPreparationModule
│  ├─ indexing/
│  │  ├─ __init__.py
│  │  └─ module.py                    # IndexConstructionModule
│  ├─ retrieval/
│  │  ├─ __init__.py
│  │  └─ module.py                    # RetrievalModule：混合检索与 RRF
│  └─ generation/
│     ├─ __init__.py
│     └─ module.py                    # GenerationModule
├─ data/
│  ├─ cook/                           # 第二版 Markdown 文档库来源
│  ├─ BEIR-NQ/                        # 后续评测数据集
│  ├─ hotpot_qa/                      # 后续评测数据集
│  ├─ rag-qa-arena/                   # 后续评测数据集
│  └─ squad2/                         # 后续评测数据集
├─ docs/
│  ├─ note/
│  ├─ prompts/
│  │  ├─ query_router.md
│  │  ├─ query_rewrite.md
│  │  ├─ list_answer.md
│  │  ├─ detail_answer.md
│  │  └─ general_answer.md
│  └─ superpowers/
│     ├─ specs/
│     │  └─ spec-v4.md
│     └─ plans/
│        └─ plan-v4.md
├─ experiments/                       # 探索性 notebook 和临时实验材料
├─ tests/
│  ├─ unit/
│  ├─ integration/
│  ├─ e2e/
│  ├─ regression/
│  └─ fixtures/
├─ config.example.yaml
├─ .env.example
├─ config.yaml
├─ .env
├─ .gitignore
├─ requirements.txt
└─ AGENTS.md
```

第二版不创建 `knowledge_base/` 目录。知识库构建只是 `RagService.build_knowledge_base()` 的生命周期方法，运行期数据仍分别保存在文档准备模块和索引构建模块中。

第二版不创建可用的 `src/evaluation/` 模块。评测能力留到后续扩展阶段。

## 5. 模块职责

### 5.1 `src/__main__.py`

职责：

- 启动 `RagService`。
- 启动连续交互式问答循环。
- 每轮展示问题提示、读取用户输入、输出回答。
- 处理退出命令。

CLI 不解析业务启动参数。支持命令只有：

```bash
python -m src
```

### 5.2 `src/service.py`

职责：

- 提供 `RagService` 作为长期运行入口。
- 启动时加载配置与环境变量。
- 初始化 `DocumentPreparationModule`、`IndexConstructionModule`、`GenerationModule`。
- 通过 `build_knowledge_base()` 编排文档加载、chunk 切分、索引加载或构建。
- 初始化 `RetrievalModule`。
- `run_interactive()` 负责外层交互式问答循环，包括提示、读取输入、退出判断和输出结果。
- `ask(question)` 负责单次实际问答，包括路由、查询优化、检索、父文档回溯、context 构建和回答生成。
- 外层交互循环不得直接实现检索或生成细节，单次问答方法不得直接读取终端输入。

核心方法：

```text
RagService.startup()
RagService.build_knowledge_base()
RagService.run_interactive()
RagService.ask(question)
RagService.shutdown()
```

`build_knowledge_base()` 不返回值，只完成模块状态准备，并在完成时打印：

```text
构建知识库完成
```

### 5.3 `src/config.py`

职责：

- 使用 `yaml.safe_load` 读取项目根目录下的 `config.yaml`。
- 使用 `load_dotenv` 加载项目根目录下的 `.env`。
- 使用 `AppConfig.model_validate` 校验必需配置项存在。
- 只校验字段类型，不校验字段值范围。
- 允许所有配置段出现未声明字段。
- 将配置转换为应用内部可使用的配置对象。

配置加载实现约束：

- YAML 解析必须使用 PyYAML 的 `yaml.safe_load`，不使用不安全的 YAML loader。
- `.env` 加载必须使用 `python-dotenv` 提供的 `load_dotenv`。
- 字段校验与配置对象构建必须通过 Pydantic 模型 `AppConfig.model_validate(raw_config)` 完成，不手写分散的字段校验逻辑。
- `AppConfig` 及其嵌套配置模型负责表达必需字段、字段类型、默认值和允许额外字段的策略。

如果根目录没有 `config.yaml`，启动失败，并提示用户复制 `config.example.yaml` 为 `config.yaml` 并填写配置。

`.env` 必须存在。若 `.env` 或系统环境变量中都没有 `DEEPSEEK_API_KEY`，启动失败，并提示填写 `.env` 或环境变量。

### 5.4 `src/document_preparation/module.py`

文档准备模块负责准备父文档、子 chunks 和父子映射，也负责根据候选 chunks 回溯并排序父文档。

第二版只使用 `DocumentPreparationModule` 一个主类管理文档加载、元数据 enhance、切分和父文档回溯，不再拆分 loader、metadata、splitter 等多个文件。后续模块复杂度真实上升时，再按职责拆分。

#### 5.4.1 `DocumentPreparationModule`

建议接口：

```python
class DocumentPreparationModule:
    CATEGORY_MAPPING = {
        "meat_dish": "荤菜",
        "vegetable_dish": "素菜",
        "soup": "汤品",
        "dessert": "甜品",
        "breakfast": "早餐",
        "staple": "主食",
        "aquatic": "水产",
        "condiment": "调料",
        "drink": "饮品",
    }

    DIFFICULTY_MAPPING = {
        5: "非常困难",
        4: "困难",
        3: "中等",
        2: "简单",
        1: "非常简单",
    }

    def __init__(self, data_path, splitter_config): ...
    def load_documents(self) -> list[Document]: ...
    def _enhance_metadata(self) -> None: ...
    def split_documents(self) -> list[Document]: ...
    def get_ranked_parent_docs(self, chunks: list[Document]) -> list[Document]: ...
```

维护属性：

```python
documents: list[Document]        # 父文档，doc_type="parent"
chunks: list[Document]           # 子文档，doc_type="child"
child_parent_map: dict[str, str] # {chunk_id: parent_id}
```

不维护 `parent_doc_map`；根据 `parent_id` 回溯父文档时允许线性查找 `documents`。

#### 5.4.2 Markdown 加载

- 根据 `data_path` 递归加载 `.md` 文件。
- 不加载 TXT 或其他格式。
- 文件内容为空或只包含空白时跳过，并打印文档内容为空警告和文档路径。
- 如果 `data_path` 不存在，启动失败并提示路径不存在。
- 如果 `data_path` 存在但没有 `.md` 文件，启动失败并打印：

```text
文档库为空
```

#### 5.4.3 父文档 metadata

`load_documents()` 负责加载父文档，并调用 `_enhance_metadata()` 直接更新 `self.documents` 中每个父文档的 metadata。`_enhance_metadata()` 不返回值。父文档 metadata 固定字段：

```python
{
    "parent_id": "...",
    "doc_type": "parent",
    "source": "...",
    "category": "...",
    "dish_name": "...",
    "difficulty": "..."
}
```

字段规则：

- `source`：相对 `data_path` 的规范化 POSIX 路径。
- `parent_id`：对 `source` 使用 Python 标准库 `hashlib` 中的 MD5 哈希算法生成稳定父文档 ID。
- `category`：检查 `source` 的相对路径片段中是否包含 `CATEGORY_MAPPING` 的 key；如果包含，则取该 key 对应的标准中文类别。比如 `dishes/aquatic/白灼虾.md` 和 `dishes/aquatic/红烧鲤鱼.md` 的相对路径片段包含 `aquatic`，因此 `category="水产"`。如果没有任何路径片段能匹配 `CATEGORY_MAPPING`，则设为 `未知`。
- `dish_name`：读取不带 `.md` 后缀的文件名。
- `difficulty`：在全文中查找第一次连续出现的实心星号 `★`，根据星号数量映射难度。1-5 个星号使用 `DIFFICULTY_MAPPING`；超过 5 个星号或未匹配到星号时设为 `未知`。

#### 5.4.4 文档切分与子 chunk metadata

使用配置指定的 Markdown splitter 切分父文档。第二版不提供 fallback splitter。

子 chunk metadata 继承父文档 metadata，但覆盖 `doc_type="child"`，并加入 chunk 字段：

```python
{
    "parent_id": "...",
    "chunk_id": "...",
    "doc_type": "child",
    "chunk_index": 0,
    "source": "...",
    "category": "...",
    "dish_name": "...",
    "difficulty": "..."
}
```

字段规则：

- `chunk_index`：chunk 在原父文档切分结果中的位置。
- `chunk_id`：对 `parent_id + chunk_index` 使用 Python 标准库 `hashlib` 中的 MD5 哈希算法生成稳定子 chunk ID。
- `child_parent_map`：维护 `{chunk_id: parent_id}`。

#### 5.4.5 父文档去重排序

`get_ranked_parent_docs(chunks)` 根据候选 chunks 回溯父文档，并返回排序后的父文档列表。

排序规则：

1. 先按父文档命中 chunk 数量降序。
2. 命中数量相同时，按该父文档第一次出现在候选 chunks 中的位置升序。

如果某个 chunk 无法根据 `parent_id` 回溯到父文档，则忽略该 chunk，继续处理其他 chunks，并打印警告。

### 5.5 `src/indexing/module.py`

第二版只使用 `IndexConstructionModule` 一个主类管理 embedding 模型加载、FAISS vectorstore 加载、构建和保存。

职责：

- 在构造函数 `__init__` 中接收 `index_save_path` 和 embedding 配置。
- 在构造函数 `__init__` 中调用 `_load_embedding_model()`，加载 HuggingFace/sentence-transformers embedding 模型。
- 允许 sentence-transformers 按其默认行为自动下载模型。
- 初始化空 vectorstore。
- 加载 LangChain FAISS vectorstore。
- 基于 chunks 构建 LangChain FAISS vectorstore。
- 保存 LangChain FAISS vectorstore。

建议接口：

```python
class IndexConstructionModule:
    def __init__(self, index_save_path, embedding_config): ...
    def _load_embedding_model(self): ...
    def load_index(self) -> bool: ...
    def build_index(self, chunks: list[Document]) -> None: ...
    def save_index(self) -> None: ...
```

约定：

- `load_index()` 成功返回 `True`。
- 索引文件不存在、索引文件不完整或加载失败时返回 `False`。
- 索引加载失败时打印：

```text
索引加载失败，将重新构建索引。
```

- `load_index()` 返回 `False` 后，由 `RagService.build_knowledge_base()` 调用 `build_index(chunks)` 和 `save_index()`。
- 索引模块只接收 chunks，不重新加载文档，不关心文档路径和切分规则。

LangChain FAISS 默认持久化文件：

```text
index.faiss
index.pkl
```

第二版不做索引过期检测。只要已有索引能加载，就直接使用。即使 FAISS 索引与当前 Markdown chunks 不一致，第二版也接受这个风险；用户需要手动删除索引目录来触发重建。

### 5.6 `src/retrieval/module.py`

职责：

- 初始化向量检索器。
- 初始化 BM25 检索器。
- 执行混合检索。
- 使用 RRF 重排。
- 返回候选 chunks。

建议接口：

```python
class RetrievalModule:
    def __init__(self, chunks, vectorstore, config): ...
    def _init_retrievers(self): ...
    def vector_search(self, query: str) -> list[Document]: ...
    def bm25_search(self, query: str) -> list[Document]: ...
    def hybrid_search(self, query: str) -> list[Document]: ...
    def _rrf_rerank(self, ranked_results: list[list[Document]]) -> list[Document]: ...
```

`_init_retrievers()` 负责统一初始化向量检索器和 BM25 检索器。`hybrid_search()` 是检索模块对外入口，负责调用 `vector_search()`、`bm25_search()` 和 `_rrf_rerank()`，最终返回候选 chunks。具体功能必须按方法拆开：`vector_search()` 负责向量检索，`bm25_search()` 负责 BM25 检索，`_rrf_rerank()` 负责 RRF 重排。

向量检索使用：

```python
vectorstore.as_retriever(
    search_type=config.retrieval.vector.search_type,
    search_kwargs={"k": config.retrieval.vector.top_k},
)
```

BM25 使用：

```python
from langchain_community.retrievers import BM25Retriever
```

BM25 基于当前启动时文档准备模块生成的 chunks 每次在内存中重建，不持久化。中文分词使用 `jieba`。

BM25 检索数量使用 `config.retrieval.bm25.top_k`。

混合检索规则：

- 向量检索取 `config.retrieval.vector.top_k` 个 chunks，默认 5。
- BM25 检索取 `config.retrieval.bm25.top_k` 个 chunks，默认 5。
- RRF 重排后再截断为 `config.retrieval.hybrid.top_k` 个候选 chunks，默认 3。
- RRF 参数写死在代码中，第二版不暴露配置。

RRF 公式：

```text
score(d) = sum(1 / (RRF_K + rank_i(d)))
```

其中 `RRF_K` 为代码常量。

### 5.7 `src/generation/module.py`

生成模块负责查询路由、查询优化、context 构建和最终回答生成。

第二版只使用 `GenerationModule` 一个主类管理 Prompt 模板加载、路由、查询优化、context 构建和回答生成。

生成模块使用 LangChain 框架实现。实现特定功能时，优先使用 LangChain 已有接口，例如 Prompt 模板、Chat model、Runnable/chain 组合、输出解析器等，避免重复造轮子。

建议方法：

```python
GenerationModule.__init__(config)
GenerationModule.setup_llm()
GenerationModule.route_query(question) -> Literal["list", "detail", "general"]
GenerationModule.rewrite_query(question) -> str
GenerationModule.build_context(docs) -> str
GenerationModule.generate_answer(
    question,
    intent: Literal["list", "detail", "general"],
    context,
    stream,
) -> str
```

`__init__(config)` 负责读取并保存生成配置，包括 provider、base_url、model_name、api_key、stream、temperature、max_tokens 和各 Prompt 模板路径。`setup_llm()` 负责使用 LangChain Chat model 初始化 OpenAI-compatible LLM 可调用对象，得到后续路由、查询优化和回答生成可复用的模型对象。

#### 5.7.1 查询路由

LLM 根据用户输入识别 intent。intent 只允许：

- `list`：用户想获取菜品列表或推荐，只需要菜名。
- `detail`：用户想获取具体制作方法或详细信息。
- `general`：不属于 `list` 或 `detail` 的其他一般性问题。

查询路由 Prompt 输出必须是 JSON，只要求包含 `intent` 字段，允许忽略额外字段。实现只读取 `intent`。

如果 JSON 解析失败、`intent` 为空或 `intent` 不在允许值内，默认使用 `general`。

如果查询路由 LLM 调用失败，默认使用 `general`，并打印信息。

#### 5.7.2 查询优化

- `list` 意图不做查询优化，直接使用原始问题检索。
- `detail` 和 `general` 意图调用查询优化。
- 查询优化返回值不需要 JSON，也不做额外清洗。
- 查询优化失败时显示：

```text
查询优化失败
```

并使用原始问题继续检索。

#### 5.7.3 Context 构建

最终回答的 context 由排序后的父文档 docs 构建，不直接使用候选 chunks 作为最终 context。

`build_context(docs)` 内部写死最大长度，第二版约为 6000 中文字符。超过长度限制时按父文档排序顺序截断。

context 格式固定：

```text
[文档1]
菜名：...
类别：...
难度：...
来源：...
内容：
...

[文档2]
菜名：...
类别：...
难度：...
来源：...
内容：
...
```

#### 5.7.4 回答生成

根据 intent 选择对应 Prompt 模板：

- `list` 使用 `docs/prompts/list_answer.md`
- `detail` 使用 `docs/prompts/detail_answer.md`
- `general` 使用 `docs/prompts/general_answer.md`

LLM API 调用失败时，当前轮返回：

```text
LLM API 调用失败，请检查 API Key、base_url、模型名或网络连接
```

程序不退出，回到下一轮提问。

## 6. 配置设计

### 6.1 `config.yaml`

`config.yaml` 位于项目根目录，是运行配置，可以提交 git，但不得包含 API Key、个人本地路径或其他敏感信息。`.env` 文件必须存在，密钥默认从 `.env` 读取；如果系统环境变量中已经存在同名密钥，则系统环境变量可以覆盖 `.env` 中的值。

推荐结构：

```yaml
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
    # 传给 LangChain FAISS vectorstore.as_retriever 的 search_type。
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
```

配置校验规则：

- 校验必需字段存在。
- 只校验字段类型，不校验字段值范围。
- 所有配置段都允许未声明字段。
- `splitter.headers_to_split_on` 与 `splitter.strip_headers` 非必填，有默认值。

### 6.2 `.env`

`.env` 位于项目根目录，只保存密钥，不提交 git。`.env` 必须存在。

推荐结构：

```env
DEEPSEEK_API_KEY=your-api-key
```

密钥可以来自 `.env` 或系统环境变量；若两处都没有 `DEEPSEEK_API_KEY`，启动失败，并提示填写 `.env` 或环境变量。

### 6.3 模板文件

需要提供可提交模板：

```text
config.example.yaml
.env.example
```

真实密钥文件不提交：

```text
.env
```

`config.example.yaml` 应与本节推荐结构保持同步。实际安装依赖时再更新 `requirements.txt`；只修改 spec 时不要为了预告依赖而改动依赖文件。

## 7. Prompt 模板

Prompt 模板文件：

```text
docs/prompts/query_router.md
docs/prompts/query_rewrite.md
docs/prompts/list_answer.md
docs/prompts/detail_answer.md
docs/prompts/general_answer.md
```

模板必需变量：

- `query_router.md`：`{question}`
- `query_rewrite.md`：`{question}`
- `list_answer.md`：`{question}`、`{context}`
- `detail_answer.md`：`{question}`、`{context}`
- `general_answer.md`：`{question}`、`{context}`

启动阶段只检查配置中声明的 Prompt 模板文件路径是否存在，不做自定义变量校验。Prompt 渲染直接使用 LangChain `ChatPromptTemplate` 实现，执行时报错交给框架。生成模块中涉及 Prompt 组合、LLM 调用、链式编排和输出解析的能力，优先使用 LangChain 对应接口。

## 8. 版本控制与忽略规则

当前项目目录已经是 Git 仓库。项目应通过 Git 管理源码、文档、测试、配置模板和 Prompt 模板，但必须避免提交本地敏感配置、索引产物、虚拟环境和缓存。

应提交到 Git 的内容：

- `src/` 应用源码。
- `tests/` 自动化测试。
- `docs/superpowers/specs/spec-v4.md`、`docs/superpowers/plans/plan-v4.md`、`docs/prompts/` 下的 Prompt 模板。
- `config.example.yaml`。
- `config.yaml`，前提是不包含密钥或个人本地路径。
- `.env.example`。
- `requirements.txt`。
- `AGENTS.md`。

不应提交到 Git 的内容：

- `.env`。
- `index_save_path` 指向的 FAISS 索引目录，例如 `storage/faiss/`。
- LangChain FAISS 产物：`index.faiss`、`index.pkl`。
- 虚拟环境目录，包括项目内残留 venv 或其他本地环境目录。
- Python 缓存目录，例如 `__pycache__/`、`.pytest_cache/`。
- 大型临时数据、运行缓存和本地实验输出。

需要提供并维护 `.gitignore`，至少覆盖：

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

## 9. 运行行为

启动命令：

```bash
python -m src
```

交互行为：

- 启动后进入连续问答循环。
- 每轮显示：

```text
您的问题是：
```

- 用户输入 `exit`、`quit`、`q` 时退出。
- 用户输入空行时不退出，提示重新输入。
- 每轮问题独立处理，不保留历史对话。
- 是否流式输出从 `generation.stream` 读取。

## 10. 索引生命周期

启动阶段：

```text
python -m src
→ 读取 config.yaml
→ 加载 .env
→ 初始化 RagService
→ 初始化 DocumentPreparationModule
→ 初始化 IndexConstructionModule
→ RagService.build_knowledge_base()
→ 尝试加载 index_save_path/index.faiss 与 index_save_path/index.pkl
```

索引策略：

- 如果 `index.faiss` 和 `index.pkl` 都存在且能成功加载，直接使用旧索引。
- 如果索引目录不存在、任一索引文件缺失、索引文件不完整或加载失败，视为索引不可用。
- 索引加载失败时打印 `索引加载失败，将重新构建索引。`
- 索引不可用时，基于当前启动时生成的 chunks 构建并保存新索引。
- 第二版不做 mtime、manifest 或 hash 过期检测。
- 第二版不提供 CLI 重建参数。
- 如果源 Markdown 变化但旧 FAISS 索引仍能加载，程序不会自动感知；此时 BM25 基于当前 chunks，FAISS 可能仍基于旧 chunks。第二版接受这个风险。

构建索引流程：

```text
文档准备模块提供 chunks
→ 加载 Embedding 模型
→ 构建 LangChain FAISS vectorstore
→ 保存到 index_save_path
→ 生成 index.faiss + index.pkl
```

## 11. 查询数据流

每轮查询流程：

```text
用户输入问题
→ RagService.run_interactive() 读取输入并调用 RagService.ask(question)
→ RagService.ask(question)
→ GenerationModule.route_query(question)
→ 如果 intent 为 detail/general，调用 GenerationModule.rewrite_query(question)
→ 如果 intent 为 list，使用原始问题
→ RetrievalModule.hybrid_search(query)
→ 如果没有候选 chunks，直接返回“未检索到相关内容。”
→ DocumentPreparationModule.get_ranked_parent_docs(chunks)
→ 如果父文档 docs 为空，直接返回“未检索到相关内容。”
→ GenerationModule.build_context(docs)
→ GenerationModule.generate_answer(question, intent, context, stream)
→ 输出回答
→ 回到下一轮“您的问题是：”
```

检索为空时：

```text
未检索到相关内容。
```

此时不进入父文档排序，也不调用最终回答生成。

候选 chunks 有结果但父文档回溯失败时，忽略失败 chunk，继续处理其他 chunk，并打印警告。如果最终 docs 为空，返回 `未检索到相关内容。`，不调用最终生成。

## 12. 错误处理与边界情况

配置缺失：

- 如果根目录没有 `config.yaml`，启动失败。
- 提示用户复制 `config.example.yaml` 为 `config.yaml` 并填写配置。

密钥缺失：

- `.env` 必须存在。
- 如果 `.env` 和系统环境变量中都没有 `DEEPSEEK_API_KEY`，启动失败。
- 提示用户填写 `.env` 或环境变量。

Prompt 模板缺失：

- 如果任一配置的 Prompt 模板路径不存在，启动失败。
- 输出缺失路径。

Prompt 变量问题：

- 启动阶段不做自定义变量校验。
- 渲染时使用 LangChain `ChatPromptTemplate`，执行时报错交给框架。

文档库路径不存在：

- 启动失败。
- 提示路径不存在。

文档库为空：

- 如果 `data_path` 存在但没有 `.md` 文件，启动失败。
- 打印 `文档库为空`。

文档内容为空：

- 如果 `.md` 文件内容为空或只包含空白，构建时跳过该文件。
- 打印文档内容为空警告和文档路径。
- 如果跳过空白文档后没有有效父文档或 chunks，启动失败。

Embedding 模型加载失败：

- 启动失败。
- 提示模型名和失败原因。

FAISS 保存或加载失败：

- 加载失败时视为索引不存在，并打印 `索引加载失败，将重新构建索引。`
- 保存失败时启动失败，并输出索引目录路径。

查询路由失败：

- JSON 解析失败、intent 非法或空值时，默认 `general`。
- LLM 调用失败时，默认 `general`，并打印信息。

查询优化失败：

- 显示 `查询优化失败`。
- 使用原始问题继续检索。

检索为空：

- 返回 `未检索到相关内容。`
- 不调用最终回答生成。

LLM API 调用失败：

- 当前轮返回失败信息。
- 程序继续运行。

失败信息：

```text
LLM API 调用失败，请检查 API Key、base_url、模型名或网络连接
```

用户输入：

- 空输入不退出，提示重新输入。
- `exit`、`quit`、`q` 退出。

流式输出中断：

- 保留已经输出的片段。
- 追加提示：`回答生成失败，请重试或检查配置/网络。`
- 回到下一轮提问。

## 13. 测试与验收标准

正式自动化测试统一放在 `tests/` 目录中，使用 pytest 组织。

测试目录约定：

```text
tests/
├─ unit/
├─ integration/
├─ e2e/
├─ regression/
└─ fixtures/
```

`tests/fixtures/` 用于保存无法直接复用项目文件的最小测试样本，不预设具体文件结构。测试应优先复用项目模板和小型稳定样本，只有当真实数据太大、不稳定、依赖私密配置，或需要特殊边界输入时，才新增 fixture。

`experiments/` 仅保存探索性 notebook 和临时实验材料，不作为 pytest 自动化测试目录。

自动化测试不得真实调用 LLM API。查询路由、查询优化和最终回答生成都应使用 fake/mock client。手工验收才使用真实 API。

### 13.1 单元测试

配置读取：

- 能读取 `config.yaml` 与 `.env`。
- 使用临时项目根目录、最小合法 `config.yaml` 与 `.env`，断言配置对象字段可正确读取。
- YAML 文件读取使用 `yaml.safe_load`。
- `.env` 文件加载使用 `load_dotenv`。
- 当 `.env` 中设置 `DEEPSEEK_API_KEY` 时，断言配置中的 `api_key=$DEEPSEEK_API_KEY` 能解析为对应环境变量值。
- 配置字段校验与配置对象构建使用 `AppConfig.model_validate`。
- 缺少必需字段时给出清晰错误。
- 字段类型错误时给出清晰错误。
- 允许未声明字段。
- `config.yaml` 中包含未声明字段时，断言加载成功且不影响必需字段解析。
- 删除 `config.yaml` 时，断言启动失败错误提示复制 `config.example.yaml`。
- 删除 `.env` 且清空系统环境变量时，断言启动失败并提示填写 `.env` 或环境变量。
- 删除 `DEEPSEEK_API_KEY` 时，断言启动失败且错误信息包含密钥缺失原因。
- 配置不存在的 Prompt 模板路径时，断言启动失败并输出缺失路径。

文档准备：

- 能递归扫描 `.md`。
- 能忽略非 Markdown 文件。
- 使用包含根目录、一级目录和嵌套目录的 Markdown fixture，断言递归加载所有 `.md` 文件。
- `data_path` 不存在时报错。
- 没有 `.md` 文件时打印 `文档库为空`。
- 空 `.md` 被跳过，并打印文档内容为空警告和路径。
- 空文件和只包含空白的文件不会出现在 `self.documents` 中。
- 如果所有 `.md` 都为空或只包含空白，断言最终无有效父文档或 chunks 时启动失败。
- 父文档 `page_content` 与源 Markdown 内容一致。
- 能 enhance 父文档 metadata：`parent_id`、`doc_type`、`source`、`category`、`dish_name`、`difficulty`。
- `CATEGORY_MAPPING` 能根据相对路径片段中的固定类别 key 映射为标准中文类别；例如 `dishes/aquatic/白灼虾.md` 包含 `aquatic`，映射为 `水产`；无法映射时为 `未知`。
- 根目录文档的 `category` 为 `未知`。
- `difficulty` 能根据第一次连续 `★` 数量映射，超过 5 个或无匹配为 `未知`。
- 同一 fixture 重复调用 `load_documents()` 时，`parent_id` 保持稳定。
- 能切分 chunks，并生成 `chunk_id`、`chunk_index`、`doc_type="child"`、`child_parent_map`。
- 每个父文档 metadata 至少包含 `parent_id`、`doc_type`、`source`、`category`、`dish_name`、`difficulty`。
- 每个 chunk metadata 至少包含 `parent_id`、`chunk_id`、`doc_type`、`chunk_index`、`source`、`category`、`dish_name`、`difficulty`。
- `len(child_parent_map) == len(chunks)`，且每个 `chunk.metadata["chunk_id"]` 都是 `child_parent_map` 的键。
- 每个 chunk 的 `parent_id` 与 `child_parent_map[chunk_id]` 完全一致。
- 对同一父文档切分出的 chunks，`chunk_index` 从 0 开始连续且不重复，chunk 内容非空。
- 使用包含稳定哨兵文本的 Markdown fixture，验证关键内容片段均出现在合并后的切分结果中，且相对顺序与父文档一致。
- `get_ranked_parent_docs()` 能按命中数量和首次出现顺序排序。

索引构建：

- `load_index()` 在索引缺失时返回 `False`。
- 索引目录不存在时，`load_index()` 返回 `False`。
- 仅存在 `index.faiss` 或仅存在 `index.pkl` 时，`load_index()` 返回 `False`。
- `load_index()` 在索引加载失败时返回 `False`。
- `build_index(chunks)` 基于传入 chunks 构建索引，不重新加载文档。
- `save_index()` 能生成 `index.faiss` 与 `index.pkl`。
- Embedding 模型加载失败时，断言启动失败且错误包含模型名和失败原因。
- FAISS 保存失败时，断言启动失败且错误包含 `index_save_path`。

Prompt 与生成：

- 启动阶段能检查所有 Prompt 模板路径存在。
- 使用 fake/mock LLM 客户端模拟初始化成功，断言 `setup_llm()` 完成大模型初始化。
- 使用 fake/mock LLM 客户端模拟初始化失败，断言 `setup_llm()` 或启动流程失败，并给出包含 provider、base_url、model_name 或底层异常原因的清晰错误信息。
- 查询路由 JSON 解析失败默认 `general`。
- intent 为空、未知值或非字符串时默认 `general`。
- 查询路由 LLM 抛异常时默认 `general`，并打印信息。
- 查询优化失败时打印 `查询优化失败`，并回退原始问题。
- `build_context()` 使用父文档 docs 构建固定格式 context，并写死最大长度截断。
- `build_context()` 生成的 context 包含菜名、类别、难度、来源和内容。
- `build_context()` 保持输入父文档 docs 的排序。
- 超长父文档被截断后，context 仍保持可读格式。
- `generate_answer()` 根据 `list`、`detail`、`general` 选择对应 Prompt 模板。
- 最终 LLM 调用失败时，断言返回固定失败文案。
- Prompt 变量错误时，断言错误交给 LangChain 抛出或传播。

检索：

- 能初始化向量检索器。
- 能初始化 `BM25Retriever`。
- 使用 fake vector retriever 返回固定 chunks，断言 `vector_search()` 返回结果数量与 `retrieval.vector.top_k` 约束一致。
- 使用小型中文 chunks 初始化 BM25，断言 `bm25_search()` 返回结果数量与 `retrieval.bm25.top_k` 约束一致。
- `hybrid_search()` 能合并向量检索和 BM25 检索结果，并返回 `retrieval.hybrid.top_k` 个候选 chunks。
- 单一路检索返回空列表时，`hybrid_search()` 仍使用另一路结果参与 RRF 重排和截断。
- 两路检索均为空时，`hybrid_search()` 返回空列表。
- 重复 chunk 同时出现在两路结果时，最终结果不重复。
- `retrieval.hybrid.top_k` 小于可用候选数时，最终结果截断为 `retrieval.hybrid.top_k`。

CLI：

- `python -m src` 启动交互循环。
- `run_interactive()` 只负责外层循环，不直接实现问答细节。
- `ask(question)` 可被单元测试直接调用，不读取终端输入。
- 不支持 `-s`、`--debug`、`--rebuild` 等 CLI 参数。

### 13.2 集成测试

- 使用小型 Markdown fixture 文档库构建父文档和 chunks。
- 真实构建 FAISS 索引，并在配置目录看到 `index.faiss` 和 `index.pkl`。
- 第二次启动时能加载已有索引。
- FAISS 加载失败时能打印重建提示并自动重建保存。
- BM25 每次启动基于当前 chunks 在内存中重建。
- 混合检索能返回候选 chunks。
- 文档准备模块能将候选 chunks 排序回溯为父文档 docs。
- 检索为空时不调用最终生成，并返回 `未检索到相关内容。`
- LLM 路由、查询优化和回答生成全部 mock。

### 13.3 E2E 测试

- 使用测试配置和小型 Markdown fixture 启动完整 `RagService`。
- mock LLM 客户端返回固定路由、优化查询和回答。
- 输入普通问题后能完成完整链路并返回预期回答。
- 输入 `exit`、`quit`、`q` 能退出。
- 输入空行不会退出，会提示重新输入。

### 13.4 回归测试

- 固化曾经出现过的配置解析、文档 metadata、chunk 映射、索引加载、路由默认值和检索为空等问题。
- 每次修复缺陷后补充对应 regression case。

### 13.5 CLI 人工验收

- 执行 `python -m src` 后进入交互式循环。
- 显示：

```text
您的问题是：
```

- 输入普通问题后根据 `generation.stream` 输出回答。
- 输入 `exit`、`quit`、`q` 能退出。
- 输入空行不会退出，会提示重新输入。
- LLM API 失败时返回失败信息并继续下一轮。

### 13.6 后续评测扩展验收

以下内容不属于第二版验收范围：

- 加载 BEIR-NQ、SQuAD2、HotpotQA、rag-qa-arena 等数据集。
- 支持 Recall@k、MRR、Hit Rate。
- 支持端到端 QA 评测。

## 14. 依赖与运行环境

第二版使用 Python 3.13 作为默认运行版本。

依赖需要安装到项目专用虚拟环境 `RAG_2026` 中。如果该虚拟环境不存在，实施阶段需要先创建；如果已存在，则直接复用。

第二版使用传统 Python 依赖管理：

```bash
pip install -r requirements.txt
```

不使用 uv 或 poetry。

依赖维护规范：

- 新增 Python 依赖时，必须同步记录到 `requirements.txt`。
- 记录依赖时必须带上实际安装和验证使用的版本号，不写裸包名。
- 不允许只在本地虚拟环境中安装依赖但不更新 `requirements.txt`。
- 实施阶段如需实际执行依赖安装命令，必须先向用户请求权限。
- 只在实际安装或实际使用依赖时修改 `requirements.txt`；仅修改 spec 时不要改动依赖文件。
- `requirements.txt` 优先记录项目直接依赖，即代码中直接使用、项目运行明确需要的包，并为这些直接依赖标注版本。
- 第二版不要求使用 `pip freeze > requirements.txt` 生成完整间接依赖清单，避免把虚拟环境中的无关包写入项目依赖。
- 如果后续需要严格锁定完整依赖树，可再引入 `requirements.lock.txt` 或 pip-tools；第二版不做。

核心依赖方向：

- LangChain
- LangChain Community
- FAISS
- HuggingFace / sentence-transformers
- OpenAI-compatible Chat API 客户端
- python-dotenv
- PyYAML
- jieba

## 15. 后续扩展方向

数据源扩展：

- TXT 文档加载。
- parquet/jsonl/tsv 数据集加载。
- PDF、HTML、网页抓取。

评测扩展：

- 评测数据集 loader。
- 检索评测：Recall@k、MRR、Hit Rate。
- 端到端 QA 评测：Exact Match、F1、LLM-as-judge。

服务扩展：

- FastAPI 查询接口。
- Web 聊天页面。

索引扩展：

- manifest 记录源文件路径、mtime、size、hash。
- 自动检测索引与源文档是否一致。
- 按文件增量更新索引。
- 多索引管理。
- CLI 手动重建参数。

检索扩展：

- 可配置 BM25 开关。
- 可配置 RRF 参数。
- 更精细的父文档排序策略。

配置扩展：

- 多 Embedding provider。
- 多 LLM provider。
- 日志配置。
