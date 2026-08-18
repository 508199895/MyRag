from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only before deps are installed.
    yaml = None

try:
    from dotenv import dotenv_values
except ModuleNotFoundError:  # pragma: no cover - exercised only before deps are installed.
    dotenv_values = None


class ConfigError(Exception):
    """Raised when local runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class DocumentsConfig:
    library_paths: list[str]
    include_extensions: list[str]


@dataclass(frozen=True)
class IndexConfig:
    persist_dir: str
    rebuild_on_source_change: bool


@dataclass(frozen=True)
class SplitterConfig:
    type: str
    headers_to_split_on: list[list[str]]


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model_name: str
    device: str
    normalize_embeddings: bool
    query_instruction: str


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int


@dataclass(frozen=True)
class GenerationConfig:
    provider: str
    base_url: str
    model_name: str
    prompt_template_path: str
    temperature: float
    max_tokens: int
    api_key: str


@dataclass(frozen=True)
class RuntimeConfig:
    stream: bool
    debug: bool


@dataclass(frozen=True)
class AppConfig:
    documents: DocumentsConfig
    index: IndexConfig
    splitter: SplitterConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    runtime: RuntimeConfig


REQUIRED_SECTIONS = (
    "documents",
    "index",
    "splitter",
    "embedding",
    "retrieval",
    "generation",
    "runtime",
)


def load_config(config_path: str | Path = "config.yaml", env_path: str | Path = ".env") -> AppConfig:
    config_file = Path(config_path)
    env_file = Path(env_path)

    if not config_file.exists():
        raise ConfigError("缺少 config.yaml，请复制 config.example.yaml 为 config.yaml 后再启动。")

    data = _load_yaml(config_file)
    _validate_required_sections(data)

    env_values = _load_env(env_file)
    api_key = env_values.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ConfigError("缺少 DEEPSEEK_API_KEY，请填写 .env 后再启动。")

    documents = data["documents"]
    index = data["index"]
    splitter = data["splitter"]
    embedding = data["embedding"]
    retrieval = data["retrieval"]
    generation = data["generation"]
    runtime = data["runtime"]

    return AppConfig(
        documents=DocumentsConfig(
            library_paths=list(documents["library_paths"]),
            include_extensions=list(documents["include_extensions"]),
        ),
        index=IndexConfig(
            persist_dir=str(index["persist_dir"]),
            rebuild_on_source_change=bool(index["rebuild_on_source_change"]),
        ),
        splitter=SplitterConfig(
            type=str(splitter["type"]),
            headers_to_split_on=[list(header) for header in splitter["headers_to_split_on"]],
        ),
        embedding=EmbeddingConfig(
            provider=str(embedding["provider"]),
            model_name=str(embedding["model_name"]),
            device=str(embedding["device"]),
            normalize_embeddings=bool(embedding["normalize_embeddings"]),
            query_instruction=str(embedding["query_instruction"]),
        ),
        retrieval=RetrievalConfig(top_k=int(retrieval["top_k"])),
        generation=GenerationConfig(
            provider=str(generation["provider"]),
            base_url=str(generation["base_url"]),
            model_name=str(generation["model_name"]),
            prompt_template_path=str(generation["prompt_template_path"]),
            temperature=float(generation["temperature"]),
            max_tokens=int(generation["max_tokens"]),
            api_key=api_key,
        ),
        runtime=RuntimeConfig(
            stream=bool(runtime["stream"]),
            debug=bool(runtime["debug"]),
        ),
    )


def _load_yaml(config_file: Path) -> dict[str, Any]:
    text = config_file.read_text(encoding="utf-8")
    if yaml is None:
        data = _load_project_yaml(text)
    else:
        data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ConfigError("config.yaml 内容必须是 YAML mapping。")
    return data


def _load_project_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_section: str | None = None
    current_list_key: str | None = None

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        if indent == 0 and line.endswith(":"):
            current_section = line[:-1]
            data[current_section] = {}
            current_list_key = None
            continue

        if current_section is None:
            raise ConfigError("config.yaml 内容必须使用顶层 section。")

        section_data = data[current_section]

        if indent == 2 and line.endswith(":"):
            current_list_key = line[:-1]
            section_data[current_list_key] = []
            continue

        if indent == 2 and ":" in line:
            key, value = line.split(":", 1)
            section_data[key.strip()] = _parse_scalar(value.strip())
            current_list_key = None
            continue

        if indent == 4 and line.startswith("- ") and current_list_key is not None:
            section_data[current_list_key].append(_parse_scalar(line[2:].strip()))
            continue

        raise ConfigError(f"无法解析 config.yaml 行: {raw_line}")

    return data


def _parse_scalar(value: str) -> Any:
    if value == "true":
        return True
    if value == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        return [_parse_scalar(item.strip()) for item in value[1:-1].split(",")]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _validate_required_sections(data: dict[str, Any]) -> None:
    for section in REQUIRED_SECTIONS:
        if section not in data:
            raise ConfigError(f"缺少必需配置 section: {section}")


def _load_env(env_file: Path) -> dict[str, str]:
    if not env_file.exists():
        return {}

    if dotenv_values is not None:
        return {key: value for key, value in dotenv_values(env_file).items() if value is not None}

    values: dict[str, str] = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values
