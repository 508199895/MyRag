# 通用 RAG 框架项目 Spec

## 1. 背景与目标

本项目要实现一个基于 LangChain 与 FAISS 的通用 RAG 框架。第一版聚焦本地 Markdown/TXT 文档问答，后续可扩展到标准 RAG/QA 数据集评测、API 服务和 Web 界面。

第一版目标：

- 支持从本地 Markdown/TXT 文件库构建向量索引。
- 使用本地 Embedding 模型完成文本向量化。
- 使用 LangChain FAISS vectorstore 保存和加载索引。
- 启动项目时自动检查索引是否存在、是否过期，并在需要时自动构建或重建。
- 对外只暴露交互式查询入口。
- 通过 DeepSeek OpenAI-compatible API 生成回答。
- 提供可复用的 `RagService`，让 CLI、后续 API/Web、测试和脚本都能复用同一套核心流程。
- 为后续数据集加载、检索评测、端到端 QA 评测预留模块边界，但不作为第一版可用功能。

第一版非目标：

- 不支持 PDF、网页抓取、图片、多模态文档。
- 不支持 parquet/jsonl/tsv 数据集加载。
- 不提供检索评测或生成评测 CLI。
- 不实现 Web/API 服务。
- 不实现多轮对话记忆。
- 不实现增量索引；源文件变化时整体重建索引。
- 不实现日志系统，仅预留后续扩展空间。

## 2. 推荐方案

采用“本地文档 RAG + 评测骨架”的 B 版方案。

第一版实现 Markdown/TXT 本地文档 RAG 查询。数据集加载、检索评测、生成评测只保留清晰模块位置，不进入主查询链路。这样可以先把文档处理、索引、检索、生成这条主链路做稳，同时避免后续加入评测功能时推倒重来。

## 3. 整体架构

项目采用“交互入口 + RagService 核心服务 + 分阶段模块”的结构。

用户通过以下命令启动：

```bash
python -m src
```

启动后进入连续交互式问答。用户不在命令行传入问题。

核心链路：

```text
config.yaml / .env
→ 初始化 RagService
→ 检查 FAISS 持久化目录与索引文件
→ 必要时扫描 Markdown/TXT 文档库
→ 必要时构建或重建 LangChain FAISS vectorstore
→ 进入连续问答循环
→ 每轮独立检索，不保留历史
→ 渲染 docs/prompts/llm_generator.md
→ 调用 DeepSeek OpenAI-compatible API
→ 输出回答
```

`RagService` 是长期运行入口，负责协调配置、索引、检索和生成。CLI 入口只负责参数解析、交互循环和终端输出。

## 4. 项目目录结构

现有项目目录包括：

```text
data/
  BEIR-NQ/
  cook/
  hotpot_qa/
  rag-qa-arena/
  squad2/
docs/
  note/
  spec.md
  prompt_plan.md
  eval_plan.md
experiments/
  readHfData.ipynb
  readJson.ipynb
  Untitled-1.ipynb
AGENTS.md
```

说明：当前仓库中的 `test/` 目录实际存放探索性 notebook，不属于正式自动化测试目录。目标结构中将其改名为 `experiments/`，正式测试统一放入 `tests/`。

第一版新增和整理后的目标结构：

```text
RAG/
├─ src/
│  ├─ __main__.py                  # 入口：python -m src
│  ├─ service.py                   # RagService 主流程
│  ├─ config.py                    # config.yaml + .env 读取与校验
│  ├─ document_processing/
│  │  ├─ __init__.py
│  │  └─ processor.py              # 扫描 .md/.txt、读取文本、切分 chunks
│  ├─ indexing/
│  │  ├─ __init__.py
│  │  ├─ embeddings.py             # 本地 BGE Embedding
│  │  └─ vectorstore.py            # FAISS 保存、加载、过期检测、重建
│  ├─ retrieval/
│  │  ├─ __init__.py
│  │  └─ retriever.py              # top_k 检索
│  ├─ generation/
│  │  ├─ __init__.py
│  │  ├─ generator.py              # DeepSeek OpenAI-compatible API 调用
│  │  └─ prompts.py                # Prompt 模板加载与渲染
│  └─ evaluation/
│     ├─ __init__.py
│     ├─ dataset_loader.py         # 后续扩展：数据集加载
│     ├─ retrieval_eval.py         # 后续扩展：检索评测
│     └─ qa_eval.py                # 后续扩展：端到端 QA 评测
├─ data/
│  ├─ cook/                        # 第一版可作为 Markdown 文档库来源
│  ├─ BEIR-NQ/                     # 后续评测数据集
│  ├─ hotpot_qa/                   # 后续评测数据集
│  ├─ rag-qa-arena/                # 后续评测数据集
│  └─ squad2/                      # 后续评测数据集
├─ docs/
│  ├─ note/                        # 现有笔记
│  ├─ prompts/
│  │  └─ llm_generator.md          # 生成回答的 Prompt 模板
│  ├─ spec.md                      # 本文件：整个项目 spec
│  ├─ prompt_plan.md               # 现有文档
│  └─ eval_plan.md                 # 现有文档
├─ experiments/
│  ├─ readHfData.ipynb             # 从现有 test/ 迁移而来，作为探索性 notebook
│  ├─ readJson.ipynb
│  └─ Untitled-1.ipynb
├─ tests/
│  ├─ unit/                        # pytest 单元测试
│  ├─ integration/                 # pytest 集成测试
│  └─ fixtures/                    # 测试用文档、配置、Prompt 等小样本
├─ config.example.yaml             # 可提交配置模板
├─ .env.example                    # 可提交环境变量模板
├─ config.yaml                     # 本地真实配置，不提交
├─ .env                            # 本地真实密钥，不提交
├─ requirements.txt
└─ AGENTS.md
```


## 5. 模块职责

### 5.1 `src/__main__.py`

职责：

- 解析 CLI 参数。
- 初始化 `RagService`。
- 启动连续交互式问答循环。
- 每轮展示问题提示、读取用户输入、输出回答。
- 处理退出命令。

支持参数：

```bash
python -m src
python -m src -s y
python -m src -s n
python -m src --debug
```

`-s y|n` 控制是否流式输出。不传时使用配置文件中的默认值。

### 5.2 `src/service.py`

职责：

- 提供 `RagService` 作为长期运行入口。
- 启动时加载配置与环境变量。
- 检查并准备向量索引。
- 对每个问题执行检索、Prompt 渲染和生成。
- 对外提供查询接口，供 CLI、后续 API/Web、脚本和测试复用。

建议核心方法：

```text
RagService.startup()
RagService.ask(question, stream=True)
RagService.shutdown()
```

可选便捷函数：

```text
ask_once(question, config_path="config.yaml")
load_config(config_path="config.yaml")
```

### 5.3 `src/config.py`

职责：

- 读取项目根目录下的 `config.yaml`。
- 加载 `.env` 中的密钥。
- 校验必需配置项。
- 将配置转换为应用内部可使用的配置对象。

### 5.4 `src/document_processing/processor.py`

职责：

- 根据 `documents.library_paths` 扫描源文档。
- 只处理 `.md` 和 `.txt`。
- 读取文件内容并转为 LangChain `Document`。
- 使用配置指定的 `MarkdownHeaderTextSplitter` 切分文本。
- 为 chunk 添加元数据。

第一版不提供 fallback splitter。

chunk metadata 至少包括：

- `source`：源文件路径。
- `chunk_id`：文档内 chunk 序号。
- `mtime`：源文件修改时间。
- header/title 信息：由 Markdown header splitter 产生或整理。

### 5.5 `src/indexing/embeddings.py`

职责：

- 加载本地 HuggingFace/sentence-transformers Embedding 模型。
- 默认模型为 `BAAI/bge-small-zh-v1.5`。
- 支持配置设备，例如 `cpu`。
- 支持向量归一化配置。
- 支持 query instruction 配置。

### 5.6 `src/indexing/vectorstore.py`

职责：

- 检查 FAISS 持久化目录。
- 检查 `index.faiss` 和 `index.pkl` 是否存在。
- 判断索引是否过期。
- 构建 LangChain FAISS vectorstore。
- 保存与加载 LangChain FAISS vectorstore。

LangChain FAISS 默认持久化文件：

```text
index.faiss
index.pkl
```

### 5.7 `src/retrieval/retriever.py`

职责：

- 基于 vectorstore 执行 top_k 检索。
- 返回检索到的 chunks 与必要的调试信息。
- debug 模式下提供来源、chunk id、相似度分数等信息。

### 5.8 `src/generation/prompts.py`

职责：

- 从 `generation.prompt_template_path` 读取 Prompt 模板。
- 校验模板至少包含 `{context}` 和 `{question}`。
- 将检索内容与用户问题渲染为最终 LLM 输入。

默认 Prompt 路径：

```text
docs/prompts/llm_generator.md
```

### 5.9 `src/generation/generator.py`

职责：

- 调用 DeepSeek OpenAI-compatible Chat API。
- 支持流式与非流式输出。
- 使用 `.env` 中的 `DEEPSEEK_API_KEY`。
- 使用 `config.yaml` 中的 `generation.base_url` 和 `generation.model_name`。

默认模型：

```text
deepseek-v4-flash
```

### 5.10 `src/evaluation/`

职责：

- 第一版仅保留扩展骨架。
- 后续支持 parquet/jsonl/tsv 数据集加载。
- 后续支持 Recall@k、MRR、Hit Rate。
- 后续支持端到端 QA 评测。

第一版不接入 CLI 主流程。

## 6. 配置设计

### 6.1 `config.yaml`

`config.yaml` 位于项目根目录，是本地真实运行配置，不提交 git。

推荐结构：

```yaml
documents:
  library_paths:
    - data/cook
  include_extensions:
    - .md
    - .txt

index:
  persist_dir: storage/faiss/default
  rebuild_on_source_change: true

splitter:
  type: markdown_header
  headers_to_split_on:
    - ["#", "h1"]
    - ["##", "h2"]
    - ["###", "h3"]

embedding:
  provider: huggingface
  model_name: BAAI/bge-small-zh-v1.5
  device: cpu
  normalize_embeddings: true
  query_instruction: "为这个句子生成表示以用于检索相关文章："

retrieval:
  top_k: 4

generation:
  provider: openai_compatible
  base_url: https://api.deepseek.com
  model_name: deepseek-v4-flash
  prompt_template_path: docs/prompts/llm_generator.md
  temperature: 0.2
  max_tokens: 1024

runtime:
  stream: true
  debug: false
```

### 6.2 `.env`

`.env` 位于项目根目录，只保存密钥，不提交 git。

推荐结构：

```env
DEEPSEEK_API_KEY=your-api-key
```

### 6.3 模板文件

需要提供可提交模板：

```text
config.example.yaml
.env.example
```

真实文件不提交：

```text
config.yaml
.env
```

## 7. 运行行为

启动命令：

```bash
python -m src
```

可选参数：

```bash
python -m src -s y
python -m src -s n
python -m src --debug
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
- 默认启用流式输出。
- `-s y` 启用流式输出。
- `-s n` 关闭流式输出。
- `--debug` 开启调试模式。

debug 模式行为：

- 显示检索来源。
- 显示 chunk id 和相似度分数。
- 索引过期时询问用户是否重建。
- LLM API 调用失败时显示更具体的异常信息。

非 debug 模式行为：

- 不显示来源。
- 索引缺失或过期时自动构建或重建。
- LLM API 调用失败时只显示面向用户的失败信息。

## 8. 索引生命周期

启动阶段：

```text
python -m src
→ 读取 config.yaml
→ 加载 .env
→ 初始化 RagService
→ 检查 index.persist_dir 是否存在
→ 检查 index.persist_dir/index.faiss 与 index.persist_dir/index.pkl 是否存在
```

索引判断：

- 如果 `index.persist_dir` 不存在，索引不可用，进入构建流程。
- 如果 `index.faiss` 或 `index.pkl` 任一缺失，索引不可用，进入构建流程。
- 如果两个索引文件都存在，再扫描源文档并执行过期检测。

过期检测：

```text
latest_source_mtime = 所有源文档中的最新修改时间
index_mtime = min(index.faiss.mtime, index.pkl.mtime)

if latest_source_mtime > index_mtime:
    索引过期
else:
    索引可用
```

过期策略：

- 非 debug 模式：自动重建索引。
- debug 模式：询问用户是否重建。
- 第一版不做增量索引，源文件变化时整体重建。

构建索引流程：

```text
扫描 documents.library_paths 下的 .md/.txt
→ 读取文档内容
→ 转为 LangChain Document
→ 使用 MarkdownHeaderTextSplitter 切分
→ 添加 chunk metadata
→ 使用 BAAI/bge-small-zh-v1.5 生成向量
→ 构建 LangChain FAISS vectorstore
→ 保存到 index.persist_dir
→ 生成 index.faiss + index.pkl
```

## 9. 查询数据流

每轮查询流程：

```text
用户输入问题
→ RagService 接收问题
→ retriever 按 top_k 检索 chunks
→ 如果没有检索结果，直接返回“未检索到相关内容。”
→ 如果有检索结果，将 chunks 拼接为 context
→ 读取并渲染 docs/prompts/llm_generator.md
→ 调用 DeepSeek OpenAI-compatible Chat API
→ 输出回答
→ debug 模式下额外显示来源
→ 回到下一轮 “您的问题是：”
```

检索为空时：

```text
未检索到相关内容。
```

此时不调用 LLM。

LLM API 调用失败时：

- 当前轮返回失败信息。
- 程序不退出。
- 回到下一轮提问。

建议普通模式提示：

```text
回答生成失败：DeepSeek API 调用失败，请检查 API Key、base_url、模型名或网络连接。
```

## 10. Prompt 模板

Prompt 模板路径由配置指定：

```yaml
generation:
  prompt_template_path: docs/prompts/llm_generator.md
```

模板至少支持以下变量：

```text
{context}
{question}
```

启动时需要校验模板文件存在，并校验必需变量存在。

回答风格由模板决定。推荐模板表达“严格基于检索内容回答；检索内容不足时说明无法从资料中确定”，但第一版不在代码中写死回答风格。

## 11. 错误处理与边界情况

配置缺失：

- 如果根目录没有 `config.yaml`，启动失败。
- 提示用户复制 `config.example.yaml` 为 `config.yaml` 并填写配置。

密钥缺失：

- 如果 `.env` 中没有 `DEEPSEEK_API_KEY`，启动失败。
- 提示用户填写 `.env`。

Prompt 模板缺失：

- 如果 `generation.prompt_template_path` 不存在，启动失败。
- 输出缺失路径。

Prompt 变量不完整：

- 如果模板缺少 `{context}` 或 `{question}`，启动失败。

索引目录不存在：

- 自动创建索引目录并构建索引。

索引文件不完整：

- 如果只存在 `index.faiss` 或只存在 `index.pkl`，视为索引损坏。
- 自动重建索引。

文档库路径不存在：

- 启动失败。
- 列出不存在的路径。

文档库为空：

- 如果没有可索引的 `.md` 或 `.txt` 文件，启动失败。

源文件变更：

- 按 mtime 检测。
- 非 debug 模式自动重建。
- debug 模式询问是否重建。

Embedding 模型加载失败：

- 启动失败。
- 提示模型名、设备配置，以及可能需要先下载模型。

FAISS 保存或加载失败：

- 输出索引目录路径。
- 提示可能需要删除损坏索引后重建。

检索为空：

- 返回“未检索到相关内容。”
- 不调用 LLM。
- debug 模式额外显示 top_k、索引路径、文档库路径。

LLM API 调用失败：

- 当前轮返回失败信息。
- 程序继续运行。
- debug 模式显示更具体的异常类型和响应状态码。

用户输入：

- 空输入不退出，提示重新输入。
- `exit`、`quit`、`q` 退出。

流式输出中断：

- 输出回答生成中断或失败提示。
- 回到下一轮提问。

## 12. 测试与验收标准

正式自动化测试统一放在 `tests/` 目录中，使用 pytest 组织。

测试目录约定：

```text
tests/
├─ unit/
│  ├─ test_config.py
│  ├─ test_document_processor.py
│  ├─ test_vectorstore.py
│  └─ test_prompts.py
├─ integration/
│  ├─ test_index_lifecycle.py
│  └─ test_rag_service.py
└─ fixtures/
```

`tests/fixtures/` 用于保存无法直接复用项目文件的最小测试样本，不预设具体文件结构。测试应优先复用项目模板和小型稳定样本，只有当真实数据太大、不稳定、依赖私密配置，或需要特殊边界输入时，才新增 fixture。

`experiments/` 仅保存探索性 notebook 和临时实验材料，不作为 pytest 自动化测试目录。

### 12.1 单元测试

配置读取：

- 能读取 `config.yaml` 与 `.env`。
- 缺少必需字段时给出清晰错误。

Prompt 模板：

- 能加载模板文件。
- 缺少 `{context}` 或 `{question}` 时失败。

文档处理：

- 能扫描 `.md` 和 `.txt`。
- 能忽略其他格式。
- 空目录报错。

索引判断：

- 能识别索引目录缺失。
- 能识别 `index.faiss` 或 `index.pkl` 缺失。
- 能识别源文件 mtime 晚于索引文件。

CLI 参数：

- `-s y` 解析为流式输出。
- `-s n` 解析为非流式输出。
- `--debug` 开启调试模式。
- 非法 `-s` 值报错。

### 12.2 轻量集成测试

- 使用小型 Markdown/TXT fixture 文档库构建 FAISS 索引。
- 构建后能在配置目录看到 `index.faiss` 和 `index.pkl`。
- 第二次启动时索引未过期则直接加载。
- 修改源文件 mtime 后能触发重建。
- 检索存在命中时能返回 context。
- 检索为空时不调用 LLM，并返回“未检索到相关内容。”
- LLM API 调用失败时返回失败信息，并继续下一轮。

### 12.3 CLI 人工验收

- 执行 `python -m src` 后进入交互式循环。
- 显示：

```text
您的问题是：
```

- 输入普通问题后默认流式输出回答。
- 输入 `exit`、`quit`、`q` 能退出。
- 输入空行不会退出，会提示重新输入。
- `python -m src -s n` 使用非流式输出。
- `python -m src --debug` 显示来源、相似度、必要时显示重建确认。
- LLM API 失败时返回失败信息并继续下一轮。

### 12.4 后续评测扩展验收

以下内容不属于第一版验收范围，但模块结构需为它们预留位置：

- 能加载 BEIR-NQ、SQuAD2、HotpotQA、rag-qa-arena 等数据集。
- 支持 Recall@k、MRR、Hit Rate。
- 支持端到端 QA 评测。

## 13. 依赖与运行环境

第一版使用 Python 3.13 作为默认运行版本。

依赖需要安装到项目专用虚拟环境 `RAG_2026` 中。如果该虚拟环境不存在，实施阶段需要先创建；如果已存在，则直接复用。

第一版使用传统 Python 依赖管理：

```bash
pip install -r requirements.txt
```

不使用 uv 或 poetry。

依赖维护规范：

- 新增 Python 依赖时，必须同步记录到 `requirements.txt`。
- 不允许只在本地虚拟环境中安装依赖但不更新 `requirements.txt`。
- 实施阶段如需实际执行依赖安装命令，必须先向用户请求权限。
- `requirements.txt` 优先记录项目直接依赖，即代码中直接使用、项目运行明确需要的包。
- 第一版不要求使用 `pip freeze > requirements.txt` 生成完整间接依赖清单，避免把虚拟环境中的无关包写入项目依赖。
- 如果后续需要严格锁定完整依赖树，可再引入 `requirements.lock.txt` 或 pip-tools；第一版不做。

核心依赖方向：

- LangChain
- FAISS
- HuggingFace / sentence-transformers
- OpenAI-compatible Chat API 客户端
- python-dotenv
- PyYAML

## 14. 后续扩展方向

数据源扩展：

- parquet/jsonl/tsv 数据集加载。
- PDF、HTML、网页抓取。

评测扩展：

- 检索评测：Recall@k、MRR、Hit Rate。
- 端到端 QA 评测：Exact Match、F1、LLM-as-judge。

服务扩展：

- FastAPI 查询接口。
- Web 聊天页面。

索引扩展：

- manifest 记录源文件路径、mtime、size、hash。
- 按文件增量更新索引。
- 多索引管理。

配置扩展：

- 多 Embedding provider。
- 多 LLM provider。
- 日志配置。
