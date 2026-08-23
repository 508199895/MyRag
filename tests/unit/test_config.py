from pathlib import Path

import pytest

from src.config import AppConfig, ConfigError, MarkdownHeaderSplitterConfig, load_config

PROMPT_PATHS = {
    "query_router_prompt_template_path": "query_router.md",
    "query_rewrite_prompt_template_path": "query_rewrite.md",
    "step_by_step_answer_prompt_template_path": "generate_step_by_step_answer.md",
    "basic_answer_prompt_template_path": "generate_basic_answer.md",
}

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
  vector:
    search_type: similarity
    top_k: 4
  bm25:
    top_k: 5
  hybrid:
    top_k: 3
generation:
  provider: openai_compatible
  base_url: https://api.deepseek.com
  model_name: deepseek-v4-flash
  api_key: $DEEPSEEK_API_KEY
  stream: true
  temperature: 0.2
  max_tokens: 1024
  query_router_prompt_template_path: docs/prompts/query_router.md
  query_rewrite_prompt_template_path: docs/prompts/query_rewrite.md
  step_by_step_answer_prompt_template_path: docs/prompts/generate_step_by_step_answer.md
  basic_answer_prompt_template_path: docs/prompts/generate_basic_answer.md
"""


def write_valid_files(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    prompt_dir = tmp_path / "docs" / "prompts"
    prompt_dir.mkdir(parents=True)
    for prompt_path in PROMPT_PATHS.values():
        (prompt_dir / prompt_path).write_text("minimal prompt\n", encoding="utf-8")
    config_path.write_text(VALID_CONFIG, encoding="utf-8")
    env_path.write_text("DEEPSEEK_API_KEY=test-key\n", encoding="utf-8")
    return config_path, env_path


def test_load_config_requires_config_yaml(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("DEEPSEEK_API_KEY=test-key\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="复制 config.example.yaml"):
        load_config(tmp_path / "config.yaml", env_path)


def test_load_config_requires_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(VALID_CONFIG, encoding="utf-8")
    with pytest.raises(ConfigError, match="复制 .env.example.*DEEPSEEK_API_KEY"):
        load_config(config_path, tmp_path / ".env")


@pytest.mark.parametrize("env_content", ["OTHER_KEY=value\n", "DEEPSEEK_API_KEY=\n"])
def test_load_config_requires_nonempty_deepseek_api_key(
    tmp_path, monkeypatch, env_content
):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(VALID_CONFIG, encoding="utf-8")
    env_path.write_text(env_content, encoding="utf-8")
    with pytest.raises(ConfigError, match="DEEPSEEK_API_KEY"):
        load_config(config_path, env_path)


def test_load_config_reports_unresolved_env_placeholder(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("MISSING_API_KEY", raising=False)
    config_path, env_path = write_valid_files(tmp_path)
    config_path.write_text(
        VALID_CONFIG.replace("$DEEPSEEK_API_KEY", "$MISSING_API_KEY"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="MISSING_API_KEY"):
        load_config(config_path, env_path)


def test_load_config_rejects_invalid_yaml(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path, env_path = write_valid_files(tmp_path)
    config_path.write_text("data_path: [\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="不是有效的 YAML"):
        load_config(config_path, env_path)


@pytest.mark.parametrize("config_data", ["", "- documents"])
def test_load_config_requires_yaml_mapping_root(tmp_path, monkeypatch, config_data):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path, env_path = write_valid_files(tmp_path)
    config_path.write_text(config_data, encoding="utf-8")
    with pytest.raises(ConfigError, match="必须是 YAML 映射"):
        load_config(config_path, env_path)


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
    config_path.write_text(
        VALID_CONFIG.replace(f"{field}:", f"missing_{field}:", 1), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match=f"(?s)config.yaml.*{field}"):
        load_config(config_path, env_path)


def test_load_config_allows_extra_fields_in_all_sections(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path, env_path = write_valid_files(tmp_path)
    extra_config = "extra_top: true\n" + VALID_CONFIG.replace(
        "splitter:\n", "splitter:\n  extra: yes\n", 1
    )
    config_path.write_text(extra_config, encoding="utf-8")
    config = load_config(config_path, env_path)
    assert config.extra_top is True
    assert config.splitter.extra is True


def test_load_config_resolves_system_environment_before_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "system-key")
    config_path, env_path = write_valid_files(tmp_path)
    config = load_config(config_path, env_path)
    assert config.generation.api_key == "system-key"


def test_load_config_reports_type_errors(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path, env_path = write_valid_files(tmp_path)
    config_path.write_text(
        VALID_CONFIG.replace("top_k: 4", "top_k: abc", 1), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="retrieval.vector.top_k.*valid integer"):
        load_config(config_path, env_path)


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ("top_k: 4", "top_k: 0"),
        ("temperature: 0.2", "temperature: 3"),
        ("max_tokens: 1024", "max_tokens: 0"),
    ],
)
def test_load_config_accepts_type_correct_values_without_range_validation(
    tmp_path, monkeypatch, original, replacement
):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path, env_path = write_valid_files(tmp_path)
    config_path.write_text(
        VALID_CONFIG.replace(original, replacement, 1), encoding="utf-8"
    )
    config = load_config(config_path, env_path)
    assert isinstance(config, AppConfig)


def test_load_config_uses_splitter_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path, env_path = write_valid_files(tmp_path)
    splitter_options = '  headers_to_split_on:\n    - ["#", "h1"]\n    - ["##", "h2"]\n    - ["###", "h3"]\n  strip_headers: false\n'
    config_path.write_text(VALID_CONFIG.replace(splitter_options, ""), encoding="utf-8")
    config = load_config(config_path, env_path)
    assert config.splitter.headers_to_split_on == [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    assert config.splitter.strip_headers is False


def test_load_config_reports_all_missing_prompt_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path, env_path = write_valid_files(tmp_path)
    missing_paths = [
        tmp_path / "docs" / "prompts" / "generate_step_by_step_answer.md",
        tmp_path / "docs" / "prompts" / "generate_basic_answer.md",
    ]
    for missing_path in missing_paths:
        missing_path.unlink()
    with pytest.raises(ConfigError) as exc_info:
        load_config(config_path, env_path)
    error_message = str(exc_info.value)
    for missing_path in missing_paths:
        assert str(missing_path) in error_message


def test_load_config_returns_v4_typed_app_config(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path, env_path = write_valid_files(tmp_path)
    config = load_config(config_path, env_path)
    assert isinstance(config, AppConfig)
    assert isinstance(config.splitter, MarkdownHeaderSplitterConfig)
    assert config.retrieval.vector.search_type == "similarity"
    assert config.retrieval.vector.top_k == 4
    assert config.retrieval.bm25.top_k == 5
    assert config.retrieval.hybrid.top_k == 3
    assert config.generation.stream is True
    assert (
        config.generation.step_by_step_answer_prompt_template_path
        == "docs/prompts/generate_step_by_step_answer.md"
    )
    assert (
        config.generation.basic_answer_prompt_template_path
        == "docs/prompts/generate_basic_answer.md"
    )
