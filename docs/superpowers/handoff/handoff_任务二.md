# 任务二 Handoff

## 我们在做什么

当前在实现 RAG Framework V1 的任务二：配置读取与模板文件。

主要依据：

- 项目规格：`docs/spec.md`
- 实现计划：`docs/superpowers/plans/2026-08-17-rag-framework-v1-no-code-separated-constraints.md`
- 计划中的位置：`Task 2: 配置读取与模板文件`

任务二当前目标：

- 用 `load_config()` 读取本地 `.env` 和 `config.yaml`。
- 先 `load_dotenv(env_path)`，再 `yaml.safe_load(config.yaml)` 得到 `config_data`。
- 用 `AppConfig(BaseModel).model_validate(config_data)` 做结构校验。
- 删除原先那批 dataclass 和手工 `_load_documents/_load_generation/...`。
- 支持 `$DEEPSEEK_API_KEY` / `${DEEPSEEK_API_KEY}` 这类环境变量占位解析。
- 对 splitter / retrieval 这种“type 决定参数结构”的配置，使用 Pydantic discriminated union。
- 为缺配置、缺 `.env`、缺/空/未解析 API Key、正常加载、不同 type 分支补测试。
- 为任务二测试补 GitHub Actions CI。

## 已经改了哪些文件

- `src/config.py`
  - 新增 `AppConfig` 及其子配置模型。
  - `ConfigModel` 统一使用 `ConfigDict(extra="forbid")`，禁止多余字段。
  - `SplitterConfig` 使用 discriminated union：
    - `MarkdownHeaderSplitterConfig`
    - `RecursiveCharacterSplitterConfig`
    - 通过 `Field(discriminator="type")` 判断分支。
  - `RetrievalConfig` 使用 discriminated union：
    - `SimilarityRetrievalConfig`
    - `MmrRetrievalConfig`
    - 通过 `Field(discriminator="type")` 判断分支。
  - `load_config()` 当前流程为：
    - 检查 `config.yaml` 是否存在。
    - 检查 `.env` 是否存在。
    - `load_dotenv(env_file, override=False)`。
    - `yaml.safe_load()` 读取配置。
    - 递归解析字符串里的环境变量占位符。
    - `AppConfig.model_validate(...)` 做结构校验。
  - `_replace_env_variable()` 已改为拒绝未设置和空字符串环境变量。

- `tests/unit/test_config.py`
  - 当前 pytest 展开后共 16 个测试用例。
  - 覆盖：
    - 缺 `config.yaml`
    - 缺 `.env`
    - 缺必需 section
    - `.env` 缺 `DEEPSEEK_API_KEY`
    - `DEEPSEEK_API_KEY` 为空字符串
    - splitter/retrieval 未知 `type`
    - recursive splitter 分支正常加载
    - mmr retrieval 分支正常加载
    - 默认 similarity + markdown_header 的正常 typed config 加载

- `config.example.yaml`
  - 保留为模板配置。
  - `generation.api_key` 使用 `$DEEPSEEK_API_KEY` 占位。
  - splitter/retrieval 带 `type` 字段，用于 Pydantic discriminated union。

- `.env.example`
  - 模板里使用 `DEEPSEEK_API_KEY=your-api-key`。

- `requirements.txt`
  - 增加/保留：
    - `python-dotenv`
    - `PyYAML`
    - `pydantic`
    - `pytest`
    - `pytest-cov`

- `.github/workflows/ci.yml`
  - 新增 GitHub Actions CI。
  - Python 版本：`3.13`。
  - 使用 `actions/setup-python@v5` 的 pip 缓存。
  - PR 到 `main` 时运行测试。
  - 命令：
    ```bash
    python -m pytest tests -v --tb=short --cov=src --cov-report=term-missing --basetemp=.pytest_tmp
    ```
  - 失败或成功后都会尝试写覆盖率摘要到 `GITHUB_STEP_SUMMARY`。

- `.gitignore`
  - 已忽略本地配置/密钥：
    - `.env`
    - `.env.*`
    - `config.yaml`
    - `config.yml`
  - 保留模板：
    - `!.env.example`
  - 忽略 pytest 临时目录：
    - `.pytest_cache/`
    - `.pytest_tmp/`
  - 忽略虚拟环境和索引产物。

## 测试状态

已通过：

```bash
.\ENV\RAG_2026\python.exe -m pytest tests -v --tb=short --cov=src --cov-report=term-missing
```

最近一次结果：

- `16 passed`
- 覆盖率：`97%`
- 未覆盖行：`src/config.py` 的无效 YAML / 根节点非映射分支附近，报告为 `124-125, 128`

没跑或没有完整验证：

- 没有在真实 GitHub Actions runner 上执行 CI。
- 没有跑全项目未来任务的集成测试，因为目前任务二只涉及配置模块。
- 没有补无效 YAML、空 YAML、根节点非 mapping 的测试，所以覆盖率剩余 3 行未覆盖。

## 下一步计划

1. 先确认是否要补 `src/config.py` 剩余未覆盖分支：
   - 无效 YAML
   - 空 YAML
   - YAML 根节点不是 mapping

2. 在本地用act作CI测试：
   - 在本地做CI测试
   - 在agent.md做CI约束，询问我相关内容，再填充

3. 后续进入任务三前，先把文件的更改状态整理清楚，并合并：
   - 当前 git status 显示一些文件像是“已删除 + 未跟踪重建”的状态。
   - 不要用 `git reset --hard` 或 `git checkout --` 处理。
   - 应该由用户确认后再 staging，避免覆盖用户已有改动。
   - 完成提交与合并

## 还没说出来的重要发现

- PowerShell 直接 `Get-Content` 读取 UTF-8 中文文件时，这个环境里会显示乱码。不能只凭终端输出判断源码中文已经损坏。需要用 UTF-8-aware 的方式检查，或通过测试/编辑器确认。

- `config.example.yaml` 和 `src/config.py`、`tests/unit/test_config.py` 在 git status 里显示为 `D` 加 `??` 的组合，像是索引里已有删除、工作区又重新生成了同名文件。这不是代码逻辑问题，但提交前必须小心整理。

- Pydantic discriminated union 的关键是配置里必须有 `type` 字段；Pydantic 根据 `type` 的 Literal 值选择模型分支。如果 `type` 不认识，或者该分支缺参数，会在 `model_validate()` 阶段失败。
