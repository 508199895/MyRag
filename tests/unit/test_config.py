from pathlib import Path

import pytest

from src.config import (
    ConfigError,
    MarkdownHeaderSplitterConfig,
    SimilarityRetrievalConfig,
    load_config,
)

VALID_CONFIG = """
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
    config_path.write_text("data_path: [\n", encoding="utf-8")

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
    "field",
    [
        "data_path",
        "index_save_path",
        "splitter",
        "embedding",
        "retrieval",
        "generation",
    ],
)
def test_load_config_requires_top_level_fields(tmp_path, monkeypatch, field):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path, env_path = write_valid_files(tmp_path)
    if field in {"data_path", "index_save_path"}:
        config_data = VALID_CONFIG.replace(f"\n{field}: ", f"\nmissing_{field}: ", 1)
    else:
        config_data = VALID_CONFIG.replace(f"\n{field}:\n", f"\nmissing_{field}:\n", 1)
    config_path.write_text(config_data, encoding="utf-8")

    with pytest.raises(ConfigError, match=f"(?s)config.yaml.*{field}"):
        load_config(config_path=config_path, env_path=env_path)


@pytest.mark.parametrize(
    "legacy_section",
    [
        "documents:\n  library_paths:\n    - data/cook\n",
        "index:\n  persist_dir: storage/faiss/default\n",
    ],
)
def test_load_config_rejects_legacy_documents_and_index_sections(
    tmp_path, monkeypatch, legacy_section
):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path, env_path = write_valid_files(tmp_path)
    config_path.write_text(VALID_CONFIG + f"\n{legacy_section}", encoding="utf-8")

    with pytest.raises(ConfigError, match="Extra inputs are not permitted"):
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
    ("original", "replacement", "assertion"),
    [
        (
            "  type: markdown_header",
            "  type: recursive_character",
            lambda config: config.splitter.type == "recursive_character",
        ),
        (
            "  type: markdown_header",
            "  type: markdown_header\n  custom_option: custom-value",
            lambda config: config.splitter.custom_option == "custom-value",
        ),
        (
            "  type: similarity",
            "  type: mmr",
            lambda config: config.retrieval.type == "mmr",
        ),
        (
            "  type: similarity",
            "  type: similarity\n  fetch_k: 20",
            lambda config: config.retrieval.fetch_k == 20,
        ),
        (
            "  model_name: BAAI/bge-small-zh-v1.5",
            "  model_name: BAAI/bge-small-zh-v1.5\n  device: cpu",
            lambda config: config.embedding.device == "cpu",
        ),
        (
            "data_path: data/cook",
            "data_path: ''",
            lambda config: config.data_path == "",
        ),
        (
            "index_save_path: storage/faiss/default",
            "index_save_path: ''",
            lambda config: config.index_save_path == "",
        ),
        (
            "  model_name: BAAI/bge-small-zh-v1.5",
            "  model_name: ''",
            lambda config: config.embedding.model_name == "",
        ),
        ("  top_k: 4", "  top_k: 0", lambda config: config.retrieval.top_k == 0),
        (
            "  temperature: 0.2",
            "  temperature: 3",
            lambda config: config.generation.temperature == 3,
        ),
        (
            "  max_tokens: 1024",
            "  max_tokens: 0",
            lambda config: config.generation.max_tokens == 0,
        ),
    ],
)
def test_load_config_accepts_type_valid_values_without_value_validation(
    tmp_path, monkeypatch, original, replacement, assertion
):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path, env_path = write_valid_files(tmp_path)
    config_path.write_text(
        VALID_CONFIG.replace(original, replacement, 1), encoding="utf-8"
    )

    config = load_config(config_path=config_path, env_path=env_path)

    assert assertion(config)


@pytest.mark.parametrize(
    ("field_path", "original", "replacement", "reason"),
    [
        (
            "embedding.model_name",
            "  model_name: BAAI/bge-small-zh-v1.5",
            "  model_name: 123",
            "Input should be a valid string",
        ),
        (
            "retrieval.top_k",
            "  top_k: 4",
            "  top_k: abc",
            "Input should be a valid integer",
        ),
        (
            "generation.temperature",
            "  temperature: 0.2",
            "  temperature: abc",
            "Input should be a valid number",
        ),
    ],
)
def test_load_config_reports_field_and_raw_reason_for_type_errors(
    tmp_path, monkeypatch, field_path, original, replacement, reason
):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path, env_path = write_valid_files(tmp_path)
    config_path.write_text(
        VALID_CONFIG.replace(original, replacement, 1), encoding="utf-8"
    )

    with pytest.raises(ConfigError) as exc_info:
        load_config(config_path=config_path, env_path=env_path)

    error_message = str(exc_info.value)
    assert field_path in error_message
    assert reason in error_message
    assert "当前值：" in error_message


def test_load_config_returns_typed_app_config(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path, env_path = write_valid_files(tmp_path)

    config = load_config(config_path=config_path, env_path=env_path)

    assert config.data_path == "data/cook"
    assert config.index_save_path == "storage/faiss/default"
    assert isinstance(config.splitter, MarkdownHeaderSplitterConfig)
    assert config.splitter.type == "markdown_header"
    assert config.splitter.headers_to_split_on == [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    assert config.splitter.strip_headers is False
    assert config.embedding.model_name == "BAAI/bge-small-zh-v1.5"
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
