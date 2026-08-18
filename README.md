# 通用 RAG 框架

这是一个本地文档库问答项目。第一版目标是用 Python 3.13、LangChain、FAISS、本地 Embedding 与 DeepSeek OpenAI-compatible API，实现对本地 Markdown/TXT 文档的交互式检索增强生成。

当前项目仍在逐步实现中，已包含配置加载模块、单元测试与 GitHub Actions CI 骨架。

## 环境要求

- Python 3.13
- pip
- act 0.2.89

本项目的 CI 在提交前会先在本地进行测试，因此需要安装 `act`。我当前使用的版本是：

```bash
act --version
# act version 0.2.89
```

## 快速开始

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

准备本地配置文件：

```bash
copy config.example.yaml config.yaml
copy .env.example .env
```

然后在 `.env` 中填写 `DEEPSEEK_API_KEY`。

## 测试

运行提交前本地 CI 检查：

```bash
make ci-local
```

该命令会在本地虚拟环境中执行与 GitHub Actions 对齐的覆盖率测试：

```bash
./ENV/RAG_2026/python.exe -m pytest tests -v --tb=short --cov=src --cov-report=term-missing --basetemp=.pytest_tmp
```

需要完整复现 GitHub Actions workflow 时，可以使用 `act`：

```bash
act pull_request -W .github\workflows\ci.yml -j test -s DEEPSEEK_API_KEY=dummy
```

当前 CI 工作流位于 `.github/workflows/ci.yml`，会在 Python 3.13 环境中安装依赖并运行测试覆盖率检查。

## 运行

计划中的 CLI 启动命令如下：

```bash
python -m src
python -m src -s y
python -m src -s n
python -m src --debug
```

其中 `-s y|n` 用于控制是否流式输出，`--debug` 用于显示检索来源、chunk id、相似度分数和更具体的错误信息。

## 目录说明

- `src/`：应用源码。
- `tests/`：正式自动化测试。
- `docs/`：规格、计划、提示词和项目文档。
- `data/cook/`：第一版可用的本地 Markdown 文档库来源。
- `.github/workflows/ci.yml`：GitHub Actions CI 配置。

## 注意事项

不要提交 `.env`、API Key、FAISS 索引产物、虚拟环境或缓存目录。`config.yaml` 可以提交，但不得包含密钥或个人本地路径。
