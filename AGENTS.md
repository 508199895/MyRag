# Repository Guidelines

## 项目身份

这是一个通用 RAG 框架项目，第二版目标是用 Python 3.13、LangChain、FAISS、本地 Embedding 与 DeepSeek OpenAI-compatible API，实现本地 Markdown 文档库的交互式问答；第二版不支持 TXT。当前仓库已经完成项目骨架、配置模板、配置加载模块和对应单元测试；RAG 主链路仍待实现。与用户交流时始终使用简体中文。

## 目录地图（救命用）

- `docs/superpowers/specs/spec-v4.md`：第二版当前实施依据规格。仅在涉及功能范围、架构或用户明确指定时读取。
- `docs/superpowers/plans/plan-v4.md`：第二版当前实施依据计划。仅在用户明确指定时读取。
- `docs/note/`：用户笔记。除非用户明确要求，不必读取。
- `data/cook/`：第二版 Markdown 文档库来源。
- `data/BEIR-NQ/`、`data/hotpot_qa/`、`data/rag-qa-arena/`、`data/squad2/`：后续评测数据集，第二版不接入主流程。
- `experiments/`：探索性 notebook 目录，历史上的 `test/` notebook 已迁移到这里。
- 目标结构：`src/` 放应用代码，`tests/` 放自动化测试，`tests/fixtures/` 放小型测试样本，`docs/prompts/` 放 Prompt 模板，`experiments/` 放 notebook。

## 代码风格与约定

Python 使用 4 空格缩进。函数、变量、模块名用 `snake_case`；类名用 `PascalCase`，例如 `RagService`、`AppConfig`。公共接口优先加类型标注。CLI 入口只做服务启动和交互循环，不解析业务启动参数；核心流程放到 `RagService` 与各领域模块中。新增直接依赖必须写入 `requirements.txt`，不要用 `pip freeze` 生成整棵依赖树。

新增或修改 Markdown 文档时，正文默认使用简体中文；只有代码标识、命令、路径、依赖包名、提交信息等需要保留英文的内容可以使用英文。

## 代码注释规范

总原则：注释解释“为什么”，代码表达“做什么”。

- 只添加有价值的注释。
- 注释应解释代码无法自证的内容：业务意图、设计取舍、边界条件、外部约束、非直觉逻辑。
- 不要写重复代码含义的注释，例如“遍历列表”“返回结果”“设置变量”。
- 复杂函数、类、模块应添加简短 docstring，说明职责、输入、输出、异常或副作用。
- 如果引入设计模式、抽象层、适配器、策略类、工厂类等，需要用一句话说明它解决的变化点或解耦问题。
- 对临时兼容逻辑、TODO、workaround 必须说明原因、触发条件和后续处理方式。
- 优先通过清晰命名、函数拆分和简单结构减少注释需求。

## 命令清单

本项目使用的 Python 虚拟环境位于 `E:\007.agent\007.project\RAG\ENV\RAG_2026`。该环境的 Python 解释器为 `E:\007.agent\007.project\RAG\ENV\RAG_2026\python.exe`。在运行 Python、pytest 或安装依赖前，优先使用该环境；PowerShell 下可用 `.\ENV\RAG_2026\Scripts\Activate.ps1` 激活。

计划中的常用命令如下，部分命令需等源码骨架创建后才可运行：

```bash
pip install -r requirements.txt
make lint
make lint-fix
make format
make format-check
make ci-local
python -m src
pytest tests -v
python -m pytest tests -v --tb=short --cov=src --cov-report=term-missing
```

`python -m src` 启动连续问答 CLI。是否流式输出由 `config.yaml` 的 `generation.stream` 控制；第二版不提供 debug 模式。

`make lint` 使用 `ruff check src tests` 做静态检查；`make lint-fix` 使用 `ruff check --fix src tests` 自动修复 ruff 能安全修复的问题；`make format` 使用 `ruff format src tests` 自动格式化；`make format-check` 使用 `ruff format --check src tests` 只检查格式不修改文件。

## CI 规则
1. 提交前必须本地跑过 `make ci-local`，确保通过；这等价于 CI 上跑的所有检查。`make ci-local` 先跑 `lint` 和 `format-check`，再跑测试；不要把会修改文件的 `format` 或 `lint-fix` 放进 CI 门禁
在本地 30 秒内能跑完，比等 CI 5 分钟快得多
2. 如果 CI 挂了：
先看哪一步挂的，贴出失败日志
不要"猜测式修复"—先复现再修
修完之后本地跑一次 `make ci-local` 再 push
3. Git 提交前必须明确向用户请求权限，并在用户同意后再执行 `git commit`；不要私自提交。

## 协作流程规则

使用 `superpowers:subagent-driven-development` 执行计划时，每个任务完成并通过 agent 内部 review 后，必须停下来把该任务的实现摘要、测试结果和 review 结果交由用户 review。只有用户明确表示通过后，才可以请求并执行该任务的 `git commit`；该任务提交完成后，才可以继续执行下一个任务。用户未确认通过前，不得派发或继续下一个任务。

## 测试原则

每个模块测试不仅验证输出输出格式，还必须验证输出是否保持业务语义。

## Pytest / 临时目录规则

- 测试里统一使用 pytest 提供的 `tmp_path` 创建临时文件和临时目录。
- 不要依赖类似 `temp/`、`tmp/`、`test_tmp/` 这种项目内固定目录。
- 项目内如果确实需要样例数据，就放在 `tests/fixtures/`，并且只读使用。
- 不得为了让测试通过而重命名、移动或新建项目级临时目录，除非用户明确要求。
- 不得修改测试脚本来迁就错误的临时目录路径；除非测试脚本本身确有 bug，并且必须先说明原因。
- 修复 pytest 临时目录相关失败时，应先定位 `cwd`、`PYTHONPATH`、fixture、`tmp_path` / `tmp_path_factory` / `monkeypatch.chdir` 的使用问题。
- 修复后必须清理本次运行产生的项目内临时文件。

## 红线

不要提交或泄露 `.env`、API Key、FAISS 索引产物、虚拟环境、缓存目录；`config.yaml` 可以提交，但不得包含密钥或个人本地路径。第二版不实现 PDF、网页抓取、图片、多模态、parquet/jsonl/tsv 数据集加载、Web/API 服务、多轮记忆、增量索引或日志系统。不要把 `test/` 当作正式测试目录；正式测试必须进入 `tests/`。如果需要安装依赖、联网下载模型，或由 agent 主动触发模型下载，先征得用户同意。

## 历史教训

当前目录已经是 Git 仓库，执行提交、分支或工作区相关操作前仍必须先确认仓库状态。PowerShell 默认输出可能把中文读成乱码，读取中文 Markdown 时使用 UTF-8。不要每次都读 `docs/superpowers/specs/spec-v4.md` 和 `docs/superpowers/plans/plan-v4.md`；仅在涉及功能范围、架构或用户明确指定时读取。第二版只交付本地 Markdown RAG 主链路；不创建可用的 `src/evaluation/` 模块，评测能力仅作为后续扩展方向。
