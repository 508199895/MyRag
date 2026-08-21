# Repository Guidelines

## 项目身份

这是一个通用 RAG 框架项目，第一版目标是用 Python 3.13、LangChain、FAISS、本地 Embedding 与 DeepSeek OpenAI-compatible API，实现本地 Markdown/TXT 文档库的交互式问答。当前仓库已经完成项目骨架、配置模板、配置加载模块和对应单元测试；RAG 主链路仍待实现。与用户交流时始终使用简体中文。

## 目录地图（救命用）

- `docs/spec.md`：项目总规格。仅在涉及功能范围、架构或用户明确指定时读取。
- `docs/superpowers/plans/2026-08-17-rag-framework-v1-no-code-separated-constraints.md`：第一版实现计划。仅在用户明确指定时读取。
- `docs/note/`：用户笔记。除非用户明确要求，不必读取。
- `data/cook/`：第一版可用的 Markdown 文档库来源。
- `data/BEIR-NQ/`、`data/hotpot_qa/`、`data/rag-qa-arena/`、`data/squad2/`：后续评测数据集，第一版不接入主流程。
- `experiments/`：探索性 notebook 目录，历史上的 `test/` notebook 已迁移到这里。
- 目标结构：`src/` 放应用代码，`tests/` 放自动化测试，`tests/fixtures/` 放小型测试样本，`docs/prompts/` 放 Prompt 模板，`experiments/` 放 notebook。

## 代码风格与约定

Python 使用 4 空格缩进。函数、变量、模块名用 `snake_case`；类名用 `PascalCase`，例如 `RagService`、`AppConfig`。公共接口优先加类型标注。CLI 入口只做参数解析和交互循环，核心流程放到 `RagService` 与各领域模块中。新增直接依赖必须写入 `requirements.txt`，不要用 `pip freeze` 生成整棵依赖树。

新增或修改 Markdown 文档时，正文默认使用简体中文；只有代码标识、命令、路径、依赖包名、提交信息等需要保留英文的内容可以使用英文。

## 命令清单

本项目使用的 Python 虚拟环境位于 `E:\007.agent\007.project\RAG\ENV\RAG_2026`。该环境的 Python 解释器为 `E:\007.agent\007.project\RAG\ENV\RAG_2026\python.exe`。在运行 Python、pytest 或安装依赖前，优先使用该环境；PowerShell 下可用 `.\ENV\RAG_2026\Scripts\Activate.ps1` 激活。

计划中的常用命令如下，部分命令需等源码骨架创建后才可运行：

```bash
pip install -r requirements.txt
python -m src
python -m src -s y
python -m src -s n
python -m src --debug
pytest tests -v
python -m pytest tests -v --tb=short --cov=src --cov-report=term-missing --basetemp=.pytest_tmp
act pull_request -W .github\workflows\ci.yml -j test -s DEEPSEEK_API_KEY=dummy
```

`python -m src` 启动连续问答 CLI。`-s y|n` 控制是否流式输出。`--debug` 显示检索来源、chunk id、相似度分数和更具体的错误信息。

## CI 规则
1. 提交前必须本地跑过 `make ci-local`，确保通过；这等价于 CI 上跑的所有检查
在本地 30 秒内能跑完，比等 CI 5 分钟快得多
2. 如果 CI 挂了：
先看哪一步挂的，贴出失败日志
不要"猜测式修复"—先复现再修
修完之后本地跑一次 `make ci-local` 再 push
3. Git 提交前必须明确向用户请求权限，并在用户同意后再执行 `git commit`；不要私自提交。

## 红线

不要提交或泄露 `.env`、API Key、FAISS 索引产物、虚拟环境、缓存目录；`config.yaml` 可以提交，但不得包含密钥或个人本地路径。第一版不实现 PDF、网页抓取、图片、多模态、parquet/jsonl/tsv 数据集加载、Web/API 服务、多轮记忆、增量索引或日志系统。不要把 `test/` 当作正式测试目录；正式测试必须进入 `tests/`。如果需要安装依赖或联网下载模型，先征得用户同意。

## 历史教训

当前目录已经是 Git 仓库，执行提交、分支或工作区相关操作前仍必须先确认仓库状态。PowerShell 默认输出可能把中文读成乱码，读取中文 Markdown 时使用 UTF-8。不要每次都读 `docs/spec.md` 和实现计划；仅在涉及功能范围、架构或用户明确指定时读取。第一版只交付本地 Markdown/TXT RAG 主链路，评测相关模块只保留骨架并明确报“不支持”。
