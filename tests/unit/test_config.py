from pathlib import Path

import pytest

from src.config import (
    ConfigError,
    MarkdownHeaderSplitterConfig,
    MmrRetrievalConfig,
    RecursiveCharacterSplitterConfig,
    SimilarityRetrievalConfig,
    load_config,
)


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
  strip_headers: false

embedding:
  provider: huggingface
  model_name: BAAI/bge-small-zh-v1.5
  device: cpu
  normalize_embeddings: true
  query_instruction: "为这个句子生成表示以用于检索相关文章："

retrieval:
  type: similarity
  top_k: 4

generation:
  provider: openai_compatible
  base_url: https://api.deepseek.com
  model_name: deepseek-v4-flash
  api_key: $DEEPSEEK_API_KEY
  prompt_template_path: docs/prompts/llm_generator.md
  temperature: 0.2
  max_tokens: 1024

runtime:
  stream: true
  debug: false
"""


def write_valid_files(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(VALID_CONFIG, encoding="utf-8")
    env_path.write_text("DEEPSEEK_API_KEY=test-key\n", encoding="utf-8")
    return config_path, env_path


def test_load_config_requires_config_yaml(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("DEEPSEEK_API_KEY=test-key\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="config.yaml"):
        load_config(config_path=tmp_path / "config.yaml", env_path=env_path)


def test_load_config_requires_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(VALID_CONFIG, encoding="utf-8")

    with pytest.raises(ConfigError, match=".env"):
        load_config(config_path=config_path, env_path=tmp_path / ".env")


def test_load_config_rejects_invalid_yaml(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path, env_path = write_valid_files(tmp_path)
    config_path.write_text("documents: [\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="不是有效的 YAML"):
        load_config(config_path=config_path, env_path=env_path)


@pytest.mark.parametrize("config_data", ["", "- documents"])
def test_load_config_requires_yaml_mapping_root(tmp_path, monkeypatch, config_data):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path, env_path = write_valid_files(tmp_path)
    config_path.write_text(config_data, encoding="utf-8")

    with pytest.raises(ConfigError, match="必须是 YAML 映射"):
        load_config(config_path=config_path, env_path=env_path)


@pytest.mark.parametrize(
    "section",
    [
        "documents",
        "index",
        "splitter",
        "embedding",
        "retrieval",
        "generation",
        "runtime",
    ],
)
def test_load_config_requires_top_level_sections(tmp_path, monkeypatch, section):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path, env_path = write_valid_files(tmp_path)
    config_data = VALID_CONFIG.replace(f"\n{section}:\n", f"\nmissing_{section}:\n", 1)
    config_path.write_text(config_data, encoding="utf-8")

    with pytest.raises(ConfigError, match=f"(?s)config.yaml.*{section}"):
        load_config(config_path=config_path, env_path=env_path)


def test_load_config_requires_deepseek_api_key_in_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(VALID_CONFIG, encoding="utf-8")
    env_path.write_text("OTHER_KEY=value\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="DEEPSEEK_API_KEY"):
        load_config(config_path=config_path, env_path=env_path)


def test_load_config_rejects_empty_deepseek_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(VALID_CONFIG, encoding="utf-8")
    env_path.write_text("DEEPSEEK_API_KEY=\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="DEEPSEEK_API_KEY"):
        load_config(config_path=config_path, env_path=env_path)


@pytest.mark.parametrize(
    ("section", "original", "replacement"),
    [
        ("splitter", "  type: markdown_header", "  type: hybrid"),
        ("retrieval", "  type: similarity", "  type: hybrid"),
    ],
)
def test_load_config_rejects_unknown_discriminated_union_type(tmp_path, monkeypatch, section, original, replacement):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path, env_path = write_valid_files(tmp_path)
    config_data = VALID_CONFIG.replace(original, replacement, 1)
    config_path.write_text(config_data, encoding="utf-8")

    with pytest.raises(ConfigError, match=f"(?s)config.yaml.*{section}.*type"):
        load_config(config_path=config_path, env_path=env_path)


def test_load_config_uses_splitter_type_to_validate_matching_params(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path, env_path = write_valid_files(tmp_path)
    config_data = VALID_CONFIG.replace(
        """splitter:
  type: markdown_header
  headers_to_split_on:
    - ["#", "h1"]
    - ["##", "h2"]
    - ["###", "h3"]
  strip_headers: false
""",
        """splitter:
  type: recursive_character
  chunk_size: 800
  chunk_overlap: 120
""",
    )
    config_path.write_text(config_data, encoding="utf-8")

    config = load_config(config_path=config_path, env_path=env_path)

    assert isinstance(config.splitter, RecursiveCharacterSplitterConfig)
    assert config.splitter.type == "recursive_character"
    assert config.splitter.chunk_size == 800
    assert config.splitter.chunk_overlap == 120


def test_load_config_uses_retrieval_type_to_validate_matching_params(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path, env_path = write_valid_files(tmp_path)
    config_data = VALID_CONFIG.replace(
        """retrieval:
  type: similarity
  top_k: 4
""",
        """retrieval:
  type: mmr
  top_k: 4
  fetch_k: 20
  lambda_mult: 0.5
""",
    )
    config_path.write_text(config_data, encoding="utf-8")

    config = load_config(config_path=config_path, env_path=env_path)

    assert isinstance(config.retrieval, MmrRetrievalConfig)
    assert config.retrieval.type == "mmr"
    assert config.retrieval.top_k == 4
    assert config.retrieval.fetch_k == 20
    assert config.retrieval.lambda_mult == 0.5


def test_load_config_returns_typed_app_config(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path, env_path = write_valid_files(tmp_path)

    config = load_config(config_path=config_path, env_path=env_path)

    assert config.documents.library_paths == ["data/cook"]
    assert config.documents.include_extensions == [".md", ".txt"]
    assert config.index.persist_dir == "storage/faiss/default"
    assert config.index.rebuild_on_source_change is True
    assert isinstance(config.splitter, MarkdownHeaderSplitterConfig)
    assert config.splitter.type == "markdown_header"
    assert config.splitter.headers_to_split_on == [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    assert config.splitter.strip_headers is False
    assert config.embedding.provider == "huggingface"
    assert config.embedding.model_name == "BAAI/bge-small-zh-v1.5"
    assert config.embedding.device == "cpu"
    assert config.embedding.normalize_embeddings is True
    assert config.embedding.query_instruction == "为这个句子生成表示以用于检索相关文章："
    assert isinstance(config.retrieval, SimilarityRetrievalConfig)
    assert config.retrieval.type == "similarity"
    assert config.retrieval.top_k == 4
    assert config.generation.provider == "openai_compatible"
    assert config.generation.base_url == "https://api.deepseek.com"
    assert config.generation.model_name == "deepseek-v4-flash"
    assert config.generation.api_key == "test-key"
    assert config.generation.prompt_template_path == "docs/prompts/llm_generator.md"
    assert config.generation.temperature == 0.2
    assert config.generation.max_tokens == 1024
    assert config.runtime.stream is True
    assert config.runtime.debug is False
