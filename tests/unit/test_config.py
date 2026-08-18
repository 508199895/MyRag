from pathlib import Path

import pytest

from src.config import ConfigError, load_config


VALID_CONFIG = """
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
"""


def write_config(tmp_path: Path, content: str = VALID_CONFIG) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def write_env(tmp_path: Path, content: str = "DEEPSEEK_API_KEY=test-key\n") -> Path:
    env_path = tmp_path / ".env"
    env_path.write_text(content, encoding="utf-8")
    return env_path


def test_load_config_reads_yaml_and_env(tmp_path: Path):
    config = load_config(config_path=write_config(tmp_path), env_path=write_env(tmp_path))

    assert config.documents.library_paths == ["data/cook"]
    assert config.documents.include_extensions == [".md", ".txt"]
    assert config.index.persist_dir == "storage/faiss/default"
    assert config.splitter.type == "markdown_header"
    assert config.embedding.model_name == "BAAI/bge-small-zh-v1.5"
    assert config.retrieval.top_k == 4
    assert config.generation.api_key == "test-key"
    assert config.generation.base_url == "https://api.deepseek.com"
    assert config.runtime.stream is True
    assert config.runtime.debug is False


def test_load_config_fails_when_config_yaml_is_missing(tmp_path: Path):
    missing_config = tmp_path / "config.yaml"

    with pytest.raises(ConfigError, match="复制 config.example.yaml 为 config.yaml"):
        load_config(config_path=missing_config, env_path=write_env(tmp_path))


def test_load_config_fails_when_deepseek_api_key_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ConfigError, match="填写 .env"):
        load_config(config_path=write_config(tmp_path), env_path=write_env(tmp_path, "OTHER=value\n"))


@pytest.mark.parametrize(
    "section",
    ["documents", "index", "splitter", "embedding", "retrieval", "generation", "runtime"],
)
def test_load_config_fails_when_required_section_is_missing(tmp_path: Path, section: str):
    lines = VALID_CONFIG.splitlines()
    start = next(index for index, line in enumerate(lines) if line == f"{section}:")
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index] and not lines[index].startswith(" ")),
        len(lines),
    )
    content = "\n".join(lines[:start] + lines[end:])

    with pytest.raises(ConfigError, match=f"缺少必需配置 section: {section}"):
        load_config(config_path=write_config(tmp_path, content), env_path=write_env(tmp_path))
