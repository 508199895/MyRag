# RAG Framework V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个基于 LangChain 与 FAISS 的通用 RAG 框架第一版，支持本地 Markdown/TXT 文档库的交互式问答。

**Architecture:** 项目采用“交互入口 + RagService 核心服务 + 分阶段模块”的结构。CLI 只负责参数解析、交互循环和终端输出；`RagService` 负责配置、索引、检索、Prompt 渲染和生成编排。文档处理、索引、检索、生成、评测骨架分别放入独立模块，第一版只接入本地文档 RAG 主链路。

**Tech Stack:** Python 3.13、pytest、PyYAML、python-dotenv、LangChain、LangChain Community FAISS、HuggingFace / sentence-transformers、OpenAI-compatible Chat API 客户端、DeepSeek API。

## Global Constraints

- 第一版使用 Python 3.13 作为默认运行版本。
- 依赖需要安装到项目专用虚拟环境 `RAG_2026` 中。
- 第一版使用传统 Python 依赖管理：`pip install -r requirements.txt`。
- 不使用 uv 或 poetry。
- 新增 Python 直接依赖时，必须同步记录到 `requirements.txt`。
- 不允许只在本地虚拟环境中安装依赖但不更新 `requirements.txt`。
- `requirements.txt` 优先记录项目直接依赖，即代码中直接使用、项目运行明确需要的包。
- 第一版不要求使用 `pip freeze > requirements.txt` 生成完整间接依赖清单。
- 如果后续需要严格锁定完整依赖树，可再引入 `requirements.lock.txt` 或 pip-tools；第一版不做。
- 实施阶段如需实际执行依赖安装命令，必须先向用户请求权限。
- 第一版只支持本地 Markdown/TXT 文档问答。
- 第一版不支持 PDF、网页抓取、图片、多模态文档。
- 第一版不支持 parquet/jsonl/tsv 数据集加载。
- 第一版不提供检索评测或生成评测 CLI。
- 第一版不实现 Web/API 服务。
- 第一版不实现多轮对话记忆。
- 第一版不实现增量索引；源文件变化时整体重建索引。
- 第一版不实现日志系统，仅预留后续扩展空间。
- `config.yaml` 与 `.env` 是本地真实配置，不提交 git。
- `config.example.yaml` 与 `.env.example` 是可提交模板。
- 正式自动化测试统一放在 `tests/` 目录中，使用 pytest 组织。
- `experiments/` 仅保存探索性 notebook 和临时实验材料，不作为 pytest 自动化测试目录。

---

## Scope Check

`docs/spec.md` 同时描述第一版主链路与后续扩展方向。第一版可独立交付的软件是本地 Markdown/TXT 文档 RAG 查询，因此本计划只实现主链路、配置模板、测试体系、运行文档和评测模块骨架。后续数据集加载、检索评测、端到端 QA 评测、FastAPI、Web 页面、PDF/HTML 支持、增量索引、多 provider、日志系统不进入本次实现。

## File Structure

- Create: `src/__init__.py`，声明应用包。
- Create: `src/__main__.py`，提供 `python -m src` 入口，解析 CLI 参数并运行交互式问答循环。
- Create: `src/config.py`，读取 `config.yaml` 与 `.env`，校验必需配置，输出内部配置对象。
- Create: `src/service.py`，提供 `RagService` 长期运行入口，编排启动、索引准备、检索、生成、关闭。
- Create: `src/document_processing/__init__.py`，导出文档处理公共接口。
- Create: `src/document_processing/processor.py`，扫描 `.md/.txt` 文件，读取文本，使用 Markdown header splitter 切分 chunks，并写入 chunk metadata。
- Create: `src/indexing/__init__.py`，导出索引公共接口。
- Create: `src/indexing/embeddings.py`，加载本地 HuggingFace / sentence-transformers embedding 模型。
- Create: `src/indexing/vectorstore.py`，检查 FAISS 索引状态，构建、保存、加载 LangChain FAISS vectorstore。
- Create: `src/retrieval/__init__.py`，导出检索公共接口。
- Create: `src/retrieval/retriever.py`，执行 top-k 检索，返回 chunks、score 与 debug 所需来源信息。
- Create: `src/generation/__init__.py`，导出生成公共接口。
- Create: `src/generation/prompts.py`，加载、校验、渲染 Prompt 模板。
- Create: `src/generation/generator.py`，调用 DeepSeek OpenAI-compatible Chat API，支持流式与非流式输出。
- Create: `src/evaluation/__init__.py`，评测扩展包骨架。
- Create: `src/evaluation/dataset_loader.py`，保留数据集加载扩展边界，第一版调用时明确报不支持。
- Create: `src/evaluation/retrieval_eval.py`，保留检索评测扩展边界，第一版调用时明确报不支持。
- Create: `src/evaluation/qa_eval.py`，保留端到端 QA 评测扩展边界，第一版调用时明确报不支持。
- Create: `docs/prompts/llm_generator.md`，默认生成 Prompt 模板。
- Create: `config.example.yaml`，可提交配置模板。
- Create: `.env.example`，可提交环境变量模板。
- Create: `requirements.txt`，声明运行与测试直接依赖。
- Create: `tests/unit/`，pytest 单元测试。
- Create: `tests/integration/`，pytest 集成测试。
- Create: `tests/fixtures/`，测试用文档、配置、Prompt 等小样本。
- Rename or move: `test/*.ipynb` to `experiments/*.ipynb`，保留探索性 notebook，不混入正式测试目录。
- Create or modify: `.gitignore`，忽略 `.env`、`config.yaml`、`storage/`、虚拟环境和测试缓存。
- Create or modify: `README.md`，记录安装、配置、运行和人工验收步骤。

### Task 1: 项目目录、依赖与测试骨架

**Files:**
- Create: `src/__init__.py`
- Create: `tests/unit/`
- Create: `tests/integration/`
- Create: `tests/fixtures/`
- Create: `experiments/`
- Move: `test/readHfData.ipynb` to `experiments/readHfData.ipynb`
- Move: `test/readJson.ipynb` to `experiments/readJson.ipynb`
- Move: `test/Untitled-1.ipynb` to `experiments/Untitled-1.ipynb`
- Create: `requirements.txt`
- Create or modify: `.gitignore`

**Interfaces:**
- Produces: 项目基础目录结构。
- Produces: pytest 测试目录约定。
- Produces: `requirements.txt` 依赖入口。

**Local Constraints:**
- `test/` 不作为正式自动化测试目录；正式 pytest 测试必须放入 `tests/`。
- `experiments/` 仅保存探索性 notebook 和临时实验材料。

- [ ] Step 1: 检查当前目录是否存在 `test/`、`tests/`、`experiments/`，记录现状。
- [ ] Step 2: 创建 `src/` 包入口。
- [ ] Step 3: 创建 `tests/unit/`、`tests/integration/`、`tests/fixtures/`。
- [ ] Step 4: 将现有探索性 notebook 从 `test/` 移入 `experiments/`。
- [ ] Step 5: 确认迁移后 `test/` 不再作为 pytest 自动化测试入口。
- [ ] Step 6: 创建 `requirements.txt`，列出项目直接依赖方向：LangChain、FAISS、HuggingFace / sentence-transformers、OpenAI-compatible Chat API 客户端、python-dotenv、PyYAML、pytest。
- [ ] Step 7: 创建或更新 `.gitignore`，忽略 `.env`、`config.yaml`、`storage/`、`RAG_2026/`、`__pycache__/`、`.pytest_cache/`。
- [ ] Step 8: 运行 `pytest tests -v`，预期当前无测试或后续任务前暂无有效用例。
- [ ] Step 9: 如当前目录是 git 仓库，提交本任务变更，提交信息为 `chore: scaffold rag project structure`。

### Task 2: 配置读取与模板文件

**Files:**
- Create: `src/config.py`
- Create: `config.example.yaml`
- Create: `.env.example`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces: `ConfigError`
- Produces: `AppConfig`
- Produces: `load_config(config_path="config.yaml", env_path=".env")`

**Local Constraints:**
- `config.yaml` 位于项目根目录，是本地真实运行配置，不提交 git。
- `.env` 位于项目根目录，只保存密钥，不提交 git。
- `config.example.yaml` 必须包含 spec 第 6.1 节推荐结构。
- `.env.example` 必须包含 `DEEPSEEK_API_KEY=your-api-key`。
- 缺少 `config.yaml` 时，启动失败并提示复制 `config.example.yaml` 为 `config.yaml`。
- 缺少 `DEEPSEEK_API_KEY` 时，启动失败并提示填写 `.env`。

- [ ] Step 1: 编写配置读取测试，覆盖能读取 `config.yaml` 与 `.env`。
- [ ] Step 2: 编写配置缺失测试，覆盖根目录没有 `config.yaml` 时给出清晰错误。
- [ ] Step 3: 编写密钥缺失测试，覆盖 `.env` 中没有 `DEEPSEEK_API_KEY` 时给出清晰错误。
- [ ] Step 4: 编写必需字段测试，覆盖缺少 `documents`、`index`、`splitter`、`embedding`、`retrieval`、`generation`、`runtime` 任一 section 时失败。
- [ ] Step 5: 运行 `pytest tests/unit/test_config.py -v`，确认由于实现尚未完成而失败。
- [ ] Step 6: 实现配置读取、环境变量读取和必需字段校验。
- [ ] Step 7: 创建 `config.example.yaml`，字段覆盖文档库、索引、splitter、embedding、retrieval、generation、runtime。
- [ ] Step 8: 创建 `.env.example`。
- [ ] Step 9: 运行 `pytest tests/unit/test_config.py -v`，确认通过。
- [ ] Step 10: 如当前目录是 git 仓库，提交本任务变更，提交信息为 `feat: add configuration loading`。

### Task 3: Prompt 模板加载与渲染

**Files:**
- Create: `src/generation/__init__.py`
- Create: `src/generation/prompts.py`
- Create: `docs/prompts/llm_generator.md`
- Test: `tests/unit/test_prompts.py`

**Interfaces:**
- Produces: `PromptTemplateError`
- Produces: `load_prompt_template(path)`
- Produces: `render_prompt(template, context, question)`

**Local Constraints:**
- 默认 Prompt 路径为 `docs/prompts/llm_generator.md`。
- Prompt 模板必须包含 `{context}` 和 `{question}`。
- Prompt 模板缺失时启动失败，并输出缺失路径。
- Prompt 变量不完整时启动失败。
- 回答风格由模板决定；第一版不在代码中写死回答风格。

- [ ] Step 1: 编写模板存在性测试，覆盖默认模板文件存在。
- [ ] Step 2: 编写模板变量测试，覆盖缺少 `{context}` 或 `{question}` 时失败。
- [ ] Step 3: 编写模板渲染测试，覆盖 context 与 question 能被注入最终 LLM 输入。
- [ ] Step 4: 运行 `pytest tests/unit/test_prompts.py -v`，确认由于实现尚未完成而失败。
- [ ] Step 5: 实现 Prompt 模板读取、必需变量校验和渲染。
- [ ] Step 6: 创建默认 Prompt，表达“严格基于检索内容回答；检索内容不足时说明无法从资料中确定”。
- [ ] Step 7: 运行 `pytest tests/unit/test_prompts.py -v`，确认通过。
- [ ] Step 8: 如当前目录是 git 仓库，提交本任务变更，提交信息为 `feat: add prompt template handling`。

### Task 4: 文档扫描与切分

**Files:**
- Create: `src/document_processing/__init__.py`
- Create: `src/document_processing/processor.py`
- Test: `tests/unit/test_document_processor.py`
- Test fixture: `tests/fixtures/documents/`

**Interfaces:**
- Consumes: `SplitterConfig`
- Produces: `DocumentProcessingError`
- Produces: `scan_source_files(library_paths, include_extensions)`
- Produces: `load_and_split_documents(files, splitter_config)`

**Local Constraints:**
- 只处理 `.md` 和 `.txt`。
- 第一版不提供 fallback splitter。
- splitter 只支持配置指定的 `MarkdownHeaderTextSplitter`。
- chunk metadata 至少包括 `source`、`chunk_id`、`mtime`。
- header/title 信息由 Markdown header splitter 产生或整理后保留。
- 文档库路径不存在时启动失败，并列出不存在路径。
- 文档库为空时启动失败。

- [ ] Step 1: 在 `tests/fixtures/documents/` 准备最小 Markdown 与 TXT 测试样本。
- [ ] Step 2: 编写扫描测试，覆盖 `.md` 和 `.txt` 被返回。
- [ ] Step 3: 编写格式过滤测试，覆盖图片、parquet、jsonl、tsv 等非目标格式被忽略。
- [ ] Step 4: 编写路径不存在测试，覆盖列出缺失文档库路径。
- [ ] Step 5: 编写空文档库测试，覆盖没有可索引文件时报错。
- [ ] Step 6: 编写切分 metadata 测试，覆盖 `source`、`chunk_id`、`mtime` 和 header/title 信息。
- [ ] Step 7: 运行 `pytest tests/unit/test_document_processor.py -v`，确认由于实现尚未完成而失败。
- [ ] Step 8: 实现递归扫描、稳定排序、UTF-8 文本读取与 LangChain Document 转换。
- [ ] Step 9: 接入 Markdown header splitter，并写入 chunk metadata。
- [ ] Step 10: 运行 `pytest tests/unit/test_document_processor.py -v`，确认通过。
- [ ] Step 11: 如当前目录是 git 仓库，提交本任务变更，提交信息为 `feat: add markdown document processing`。

### Task 5: Embedding 加载与 FAISS 索引生命周期

**Files:**
- Create: `src/indexing/__init__.py`
- Create: `src/indexing/embeddings.py`
- Create: `src/indexing/vectorstore.py`
- Test: `tests/unit/test_vectorstore.py`
- Test: `tests/integration/test_index_lifecycle.py`

**Interfaces:**
- Consumes: `EmbeddingConfig`
- Consumes: LangChain `Document`
- Produces: `EmbeddingError`
- Produces: `create_embeddings(config)`
- Produces: `IndexStatus`
- Produces: `check_index_status(persist_dir, source_files)`
- Produces: `build_and_save_vectorstore(documents, embeddings, persist_dir)`
- Produces: `load_vectorstore(persist_dir, embeddings)`

**Local Constraints:**
- 默认 Embedding 模型为 `BAAI/bge-small-zh-v1.5`。
- 支持配置设备，例如 `cpu`。
- 支持向量归一化配置。
- 支持 query instruction 配置。
- LangChain FAISS 默认持久化文件为 `index.faiss` 和 `index.pkl`。
- `index.persist_dir` 不存在时，索引不可用并进入构建流程。
- `index.faiss` 或 `index.pkl` 任一缺失时，索引不可用并进入构建流程。
- 过期检测使用所有源文档最新 mtime 与两个索引文件较早 mtime 比较。
- 非 debug 模式下索引缺失或过期自动构建或重建。
- 第一版不做增量索引，源文件变化时整体重建。

- [ ] Step 1: 编写 embedding 配置传递测试，覆盖模型名、设备、归一化、query instruction。
- [ ] Step 2: 编写 embedding 加载失败测试，覆盖错误提示包含模型名和设备配置。
- [ ] Step 3: 编写索引目录缺失测试，覆盖索引不可用。
- [ ] Step 4: 编写索引文件不完整测试，覆盖任一 FAISS 文件缺失时索引不可用。
- [ ] Step 5: 编写索引过期测试，覆盖源文档 mtime 晚于索引文件时判定过期。
- [ ] Step 6: 编写索引可用测试，覆盖索引文件完整且未过期时可加载。
- [ ] Step 7: 运行 `pytest tests/unit/test_vectorstore.py -v`，确认由于实现尚未完成而失败。
- [ ] Step 8: 实现 embedding factory 和清晰失败错误。
- [ ] Step 9: 实现 FAISS 索引状态检查、构建保存和加载。
- [ ] Step 10: 编写轻量集成测试，使用小型 fixture 文档库构建索引并确认生成 `index.faiss` 与 `index.pkl`。
- [ ] Step 11: 编写第二次启动集成测试，验证索引未过期时直接加载。
- [ ] Step 12: 编写源文件 mtime 更新集成测试，验证触发重建。
- [ ] Step 13: 运行 `pytest tests/unit/test_vectorstore.py tests/integration/test_index_lifecycle.py -v`，确认通过。
- [ ] Step 14: 如当前目录是 git 仓库，提交本任务变更，提交信息为 `feat: add faiss index lifecycle`。

### Task 6: 检索适配

**Files:**
- Create: `src/retrieval/__init__.py`
- Create: `src/retrieval/retriever.py`
- Test: `tests/unit/test_retriever.py`

**Interfaces:**
- Consumes: LangChain vectorstore
- Produces: `RetrievedChunk`
- Produces: `retrieve(vectorstore, question, top_k)`
- Produces: `format_debug_sources(chunks)`

**Local Constraints:**
- 基于 vectorstore 执行 top-k 检索。
- `top_k` 来自 `retrieval.top_k` 配置。
- 返回检索到的 chunks 与必要 debug 信息。
- debug 模式需要能展示来源、chunk id 和相似度分数。
- 检索为空时由上层服务返回“未检索到相关内容。”，检索层只返回空结果。

- [ ] Step 1: 编写 top-k 检索测试，覆盖按配置的 `top_k` 调用 vectorstore。
- [ ] Step 2: 编写结果包装测试，覆盖返回 chunk 与相似度分数。
- [ ] Step 3: 编写 debug 来源格式测试，覆盖 source、chunk id、score。
- [ ] Step 4: 编写空结果测试，覆盖 vectorstore 无命中时返回空列表。
- [ ] Step 5: 运行 `pytest tests/unit/test_retriever.py -v`，确认由于实现尚未完成而失败。
- [ ] Step 6: 实现检索包装和 debug 来源格式化。
- [ ] Step 7: 运行 `pytest tests/unit/test_retriever.py -v`，确认通过。
- [ ] Step 8: 如当前目录是 git 仓库，提交本任务变更，提交信息为 `feat: add retriever adapter`。

### Task 7: DeepSeek OpenAI-compatible 生成适配

**Files:**
- Create: `src/generation/generator.py`
- Test: `tests/unit/test_generator.py`

**Interfaces:**
- Consumes: `GenerationConfig`
- Produces: `GenerationError`
- Produces: `DeepSeekGenerator`

**Local Constraints:**
- 使用 `.env` 中的 `DEEPSEEK_API_KEY`。
- 使用 `config.yaml` 中的 `generation.base_url` 和 `generation.model_name`。
- 默认模型为 `deepseek-v4-flash`。
- 支持流式与非流式输出。
- LLM API 调用失败时，当前轮返回失败信息，程序不退出。
- 普通模式失败提示为面向用户的统一信息。
- debug 模式需要能显示更具体的异常类型和响应状态码。

- [ ] Step 1: 编写非流式生成测试，覆盖返回完整回答文本。
- [ ] Step 2: 编写流式生成测试，覆盖逐段输出文本。
- [ ] Step 3: 编写 API 配置传递测试，覆盖 api key、base URL、模型名、temperature、max tokens。
- [ ] Step 4: 编写 API 失败测试，覆盖异常被包装为生成失败错误。
- [ ] Step 5: 编写 debug 错误信息测试，覆盖可读取具体异常类型或响应状态码。
- [ ] Step 6: 运行 `pytest tests/unit/test_generator.py -v`，确认由于实现尚未完成而失败。
- [ ] Step 7: 实现 DeepSeek OpenAI-compatible client 初始化。
- [ ] Step 8: 实现流式与非流式生成路径。
- [ ] Step 9: 实现失败包装和 debug 信息保留。
- [ ] Step 10: 运行 `pytest tests/unit/test_generator.py -v`，确认通过。
- [ ] Step 11: 如当前目录是 git 仓库，提交本任务变更，提交信息为 `feat: add deepseek generator adapter`。

### Task 8: RagService 主流程编排

**Files:**
- Create: `src/service.py`
- Test: `tests/integration/test_rag_service.py`

**Interfaces:**
- Consumes: `load_config(config_path="config.yaml", env_path=".env")`
- Consumes: `scan_source_files(library_paths, include_extensions)`
- Consumes: `load_and_split_documents(files, splitter_config)`
- Consumes: `create_embeddings(config)`
- Consumes: `check_index_status(persist_dir, source_files)`
- Consumes: `build_and_save_vectorstore(documents, embeddings, persist_dir)`
- Consumes: `load_vectorstore(persist_dir, embeddings)`
- Consumes: `retrieve(vectorstore, question, top_k)`
- Consumes: `load_prompt_template(path)`
- Consumes: `render_prompt(template, context, question)`
- Consumes: `DeepSeekGenerator`
- Produces: `RagService.startup()`
- Produces: `RagService.ask(question, stream=None)`
- Produces: `RagService.shutdown()`
- Produces: `ask_once(question, config_path="config.yaml")`

**Local Constraints:**
- `RagService` 是长期运行入口，负责协调配置、索引、检索和生成。
- CLI、后续 API/Web、测试和脚本都应复用同一套核心流程。
- 每轮问题独立处理，不保留历史对话。
- 检索为空时返回“未检索到相关内容。”。
- 检索为空时不调用 LLM。
- 非 debug 模式索引缺失或过期时自动构建或重建。
- debug 模式索引过期时询问用户是否重建。
- LLM API 调用失败时当前轮返回失败信息，程序继续运行。

- [ ] Step 1: 编写启动流程测试，覆盖配置、源文档、embedding、索引、prompt、generator 的初始化顺序。
- [ ] Step 2: 编写索引缺失服务测试，覆盖构建并保存索引。
- [ ] Step 3: 编写索引可用服务测试，覆盖直接加载索引。
- [ ] Step 4: 编写索引过期服务测试，覆盖非 debug 自动重建。
- [ ] Step 5: 编写 debug 过期确认测试，覆盖用户确认与拒绝两条路径。
- [ ] Step 6: 编写查询命中测试，覆盖检索、context 拼接、Prompt 渲染、生成调用。
- [ ] Step 7: 编写检索为空测试，覆盖不调用 LLM 并返回固定文案。
- [ ] Step 8: 编写生成失败测试，覆盖返回失败信息且下一轮仍可继续。
- [ ] Step 9: 运行 `pytest tests/integration/test_rag_service.py -v`，确认由于实现尚未完成而失败。
- [ ] Step 10: 实现 `RagService.startup()`。
- [ ] Step 11: 实现 `RagService.ask()`。
- [ ] Step 12: 实现 `RagService.shutdown()` 与 `ask_once()`。
- [ ] Step 13: 运行 `pytest tests/integration/test_rag_service.py -v`，确认通过。
- [ ] Step 14: 如当前目录是 git 仓库，提交本任务变更，提交信息为 `feat: add rag service orchestration`。

### Task 9: CLI 交互入口

**Files:**
- Create: `src/__main__.py`
- Test: `tests/unit/test_cli.py`

**Interfaces:**
- Consumes: `RagService`
- Produces: `build_parser()`
- Produces: `parse_args(argv=None)`
- Produces: `main(argv=None)`

**Local Constraints:**
- 启动命令为 `python -m src`。
- 用户不在命令行传入问题。
- 启动后进入连续交互式问答。
- 支持 `python -m src`、`python -m src -s y`、`python -m src -s n`、`python -m src --debug`。
- `-s y|n` 控制是否流式输出，不传时使用配置文件默认值。
- 每轮显示 `您的问题是：`。
- 用户输入 `exit`、`quit`、`q` 时退出。
- 用户输入空行时不退出，提示重新输入。
- debug 模式显示检索来源、chunk id、相似度分数和更具体失败信息。
- 非 debug 模式不显示来源。

- [ ] Step 1: 编写 CLI 参数测试，覆盖无参数、`-s y`、`-s n`、`--debug`。
- [ ] Step 2: 编写非法 `-s` 测试，覆盖非 `y|n` 值报错。
- [ ] Step 3: 编写交互循环测试，覆盖显示 `您的问题是：`。
- [ ] Step 4: 编写空输入测试，覆盖提示重新输入且不退出。
- [ ] Step 5: 编写退出命令测试，覆盖 `exit`、`quit`、`q`。
- [ ] Step 6: 编写输出测试，覆盖字符串回答和流式片段回答。
- [ ] Step 7: 运行 `pytest tests/unit/test_cli.py -v`，确认由于入口尚未实现而失败。
- [ ] Step 8: 实现 CLI 参数解析。
- [ ] Step 9: 实现交互式问答循环。
- [ ] Step 10: 实现流式覆盖、debug 覆盖、空输入处理和退出处理。
- [ ] Step 11: 运行 `pytest tests/unit/test_cli.py -v`，确认通过。
- [ ] Step 12: 如当前目录是 git 仓库，提交本任务变更，提交信息为 `feat: add interactive cli`。

### Task 10: 评测扩展骨架

**Files:**
- Create: `src/evaluation/__init__.py`
- Create: `src/evaluation/dataset_loader.py`
- Create: `src/evaluation/retrieval_eval.py`
- Create: `src/evaluation/qa_eval.py`
- Test: `tests/unit/test_evaluation_skeleton.py`

**Interfaces:**
- Produces: `UnsupportedEvaluationFeature`
- Produces: `load_dataset(path)`
- Produces: `evaluate_retrieval(...)`
- Produces: `evaluate_qa(...)`

**Local Constraints:**
- 第一版仅保留扩展骨架。
- 第一版不支持 parquet/jsonl/tsv 数据集加载。
- 第一版不支持 Recall@k、MRR、Hit Rate。
- 第一版不支持端到端 QA 评测。
- 第一版评测模块不接入 CLI 主流程。
- 骨架接口被调用时必须明确说明第一版不支持。

- [ ] Step 1: 编写数据集加载骨架测试，覆盖调用时明确说明第一版不支持。
- [ ] Step 2: 编写检索评测骨架测试，覆盖调用时明确说明第一版不支持。
- [ ] Step 3: 编写 QA 评测骨架测试，覆盖调用时明确说明第一版不支持。
- [ ] Step 4: 运行 `pytest tests/unit/test_evaluation_skeleton.py -v`，确认由于评测骨架尚未实现而失败。
- [ ] Step 5: 实现评测骨架异常和三个显式不可用接口。
- [ ] Step 6: 运行 `pytest tests/unit/test_evaluation_skeleton.py -v`，确认通过。
- [ ] Step 7: 如当前目录是 git 仓库，提交本任务变更，提交信息为 `feat: add evaluation extension skeleton`。

### Task 11: 全链路集成、运行文档与人工验收

**Files:**
- Create or modify: `README.md`
- Test: `tests/integration/test_rag_flow.py`
- Modify as needed: `src/service.py`
- Modify as needed: `src/__main__.py`

**Interfaces:**
- Consumes: Task 1 到 Task 10 的所有公共接口。
- Produces: 可验收的第一版 RAG CLI 应用。

**Local Constraints:**
- 轻量集成测试使用小型 Markdown/TXT fixture 文档库。
- `tests/fixtures/` 只保存无法直接复用项目文件的最小测试样本。
- 测试应优先复用项目模板和小型稳定样本。
- 真实 `DEEPSEEK_API_KEY` 不进入自动化测试。
- 人工验收可使用真实 `.env` 调用 DeepSeek。
- README 必须记录 Python 3.13、`RAG_2026`、`pip install -r requirements.txt`、配置复制、运行命令和退出命令。

- [ ] Step 1: 编写全链路 fixture，包含一个 Markdown 文件、一个 TXT 文件和测试 Prompt。
- [ ] Step 2: 编写索引构建集成测试，验证构建后存在 `index.faiss` 和 `index.pkl`。
- [ ] Step 3: 编写第二次启动集成测试，验证索引未过期则直接加载。
- [ ] Step 4: 编写源文件 mtime 更新集成测试，验证触发重建。
- [ ] Step 5: 编写检索命中集成测试，验证 context 包含 fixture 文档内容。
- [ ] Step 6: 编写检索为空集成测试，验证不调用 LLM 并返回“未检索到相关内容。”。
- [ ] Step 7: 编写 LLM API 失败集成测试，验证返回失败信息并继续下一轮。
- [ ] Step 8: 运行 `pytest tests/integration -v`，修复暴露出的集成问题。
- [ ] Step 9: 运行 `pytest tests -v`，确认全部自动化测试通过。
- [ ] Step 10: 编写 README setup 部分，说明 Python 3.13 与虚拟环境 `RAG_2026`。
- [ ] Step 11: 编写 README 配置部分，说明复制 `config.example.yaml` 到 `config.yaml`，复制 `.env.example` 到 `.env`，并填写 `DEEPSEEK_API_KEY`。
- [ ] Step 12: 编写 README 运行部分，说明 `python -m src`、`python -m src -s y`、`python -m src -s n`、`python -m src --debug`。
- [ ] Step 13: 编写 README 交互部分，说明提示语、空输入行为和退出命令。
- [ ] Step 14: 运行 `python -m src --help`，确认显示 `-s` 与 `--debug`。
- [ ] Step 15: 准备真实 `.env` 后运行 `python -m src -s n`，确认显示 `您的问题是：`。
- [ ] Step 16: 输入一个 `data/cook` 可回答的问题，确认能构建或加载索引、检索、调用 DeepSeek 并输出回答。
- [ ] Step 17: 输入空行，确认不会退出并提示重新输入。
- [ ] Step 18: 输入 `q`，确认程序正常退出。
- [ ] Step 19: 如当前目录是 git 仓库，提交本任务变更，提交信息为 `docs: add rag runtime and acceptance notes`。

## Self-Review

**Spec coverage:** 本计划覆盖 spec 第 1 至 13 节的第一版目标、非目标、推荐方案、架构、目录结构、模块职责、配置设计、运行行为、索引生命周期、查询数据流、Prompt 模板、错误处理、测试验收、依赖环境和依赖维护规范。第 14 节扩展方向只通过评测骨架和模块边界预留，不进入第一版实现。

**Constraint separation:** 全局约束保留项目级、跨任务始终成立的规则，包括依赖维护规则；模块行为、错误处理、测试 fixture、目录迁移等只影响单个任务的要求放入对应任务的 Local Constraints。

**No-code compliance:** 本计划不包含实现代码片段，不包含测试代码片段，不包含 fenced code block；只包含文件职责、接口边界、测试目标、运行命令和验收步骤。

**Scope boundaries:** PDF、网页抓取、图片、多模态、parquet/jsonl/tsv 数据集加载、检索评测 CLI、生成评测 CLI、Web/API 服务、多轮对话记忆、增量索引、日志系统均保持非目标状态。

**Type/interface consistency:** `RagService`、`AppConfig`、`RetrievedChunk`、`DeepSeekGenerator`、`IndexStatus`、评测骨架接口在任务间命名一致，后续任务只消费前序任务定义的接口。

**Known repository state:** 之前在 `E:\007.agent\007.project\RAG` 执行 `git status --short` 显示当前目录不是 git 仓库。实施 commit 步骤前，需要确认是否初始化 git 或切换到正确仓库根目录。
