# Task 3 完成报告：Prompt 模板加载与渲染

## 实现内容

- 新增 `src/generation/__init__.py`，导出 `PromptTemplateError`、`load_prompt_template` 和 `render_prompt`。
- 新增 `src/generation/prompts.py`：
  - 默认模板路径为 `docs/prompts/llm_generator.md`。
  - 以 UTF-8 读取模板，并在文件缺失或不可读取时抛出包含路径的 `PromptTemplateError`。
  - 使用 `string.Formatter` 校验模板包含 `{context}` 和 `{question}`。
  - 使用模板渲染检索内容与用户问题，回答风格完全由模板决定。
- 新增 `tests/unit/test_prompts.py`，覆盖默认模板存在、必需变量缺失、缺失路径错误和 context/question 注入。
- 检查确认已有 `docs/prompts/llm_generator.md` 满足要求：包含两个变量，并表达严格依据检索内容回答、资料不足时说明“无法从资料中确定”。
- 未新增直接依赖，`requirements.txt` 无需修改。

## 测试结果

- `ENV\\RAG_2026\\python.exe -m pytest tests/unit/test_prompts.py -v`：5 passed。
- `ENV\\RAG_2026\\python.exe -m pytest tests -q`：24 passed。
- `git diff --check`：通过，无空白错误。

## TDD Evidence

### RED

命令：`ENV\\RAG_2026\\python.exe -m pytest tests/unit/test_prompts.py -v`

实现前关键输出：

```text
collected 0 items / 1 error
ModuleNotFoundError: No module named 'src.generation.prompts'
```

该失败由待实现模块不存在引起。最初直接执行 brief 指定的 `pytest tests/unit/test_prompts.py -v` 时，系统未找到 `pytest` 命令；随后使用仓库已有 Python 3.13 环境中的 pytest executable 完成了有效 RED 验证，未安装依赖。

### GREEN

命令：`ENV\\RAG_2026\\python.exe -m pytest tests/unit/test_prompts.py -v`

关键输出：

```text
collected 5 items
5 passed in 0.05s
```

## 改动文件

- `src/generation/__init__.py`
- `src/generation/prompts.py`
- `tests/unit/test_prompts.py`
- `docs/prompts/llm_generator.md`：已检查，原有文件满足要求，未改动
- 本报告文件：`task-3-report.md`

## 自审发现

- 缺失模板路径会进入 `PromptTemplateError`，错误消息包含传入路径。
- 模板变量校验同时覆盖加载阶段和渲染阶段，格式语法错误也会转换为领域异常。
- Windows 临时路径断言使用异常文本包含判断，避免把反斜杠误当作正则转义。
- 工作区原先干净；本任务未混入其他源文件或依赖变更。

## 关注点

- 当前 `pytest` 不在系统 PATH 中，验证使用仓库内置环境 `ENV\\RAG_2026` 的 Python 3.13 与 pytest；功能测试和全量测试均已通过。
- 未执行 `make ci-local`，因为本任务验收范围要求的目标测试及相关已有测试已完成，且当前实现没有新增依赖或 CI 配置变化。
