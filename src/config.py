from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ConfigError(Exception):
    """Raised when local runtime configuration is missing or invalid."""


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentsConfig(ConfigModel):
    library_paths: list[str]
    include_extensions: list[str]


class IndexConfig(ConfigModel):
    persist_dir: str
    rebuild_on_source_change: bool


class MarkdownHeaderSplitterConfig(ConfigModel):
    type: Literal["markdown_header"]
    headers_to_split_on: list[tuple[str, str]]
    strip_headers: bool


class RecursiveCharacterSplitterConfig(ConfigModel):
    type: Literal["recursive_character"]
    chunk_size: int
    chunk_overlap: int


class EmbeddingConfig(ConfigModel):
    provider: str
    model_name: str
    device: str
    normalize_embeddings: bool
    query_instruction: str


class SimilarityRetrievalConfig(ConfigModel):
    type: Literal["similarity"]
    top_k: int


class MmrRetrievalConfig(ConfigModel):
    type: Literal["mmr"]
    top_k: int
    fetch_k: int
    lambda_mult: float


class GenerationConfig(ConfigModel):
    provider: str
    base_url: str
    model_name: str
    api_key: str
    prompt_template_path: str
    temperature: float
    max_tokens: int


class RuntimeConfig(ConfigModel):
    stream: bool
    debug: bool


SplitterConfig = Annotated[
    MarkdownHeaderSplitterConfig | RecursiveCharacterSplitterConfig,
    Field(discriminator="type"),
]

RetrievalConfig = Annotated[
    SimilarityRetrievalConfig | MmrRetrievalConfig,
    Field(discriminator="type"),
]


class AppConfig(ConfigModel):
    documents: DocumentsConfig
    index: IndexConfig
    splitter: SplitterConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    runtime: RuntimeConfig


ENV_PATTERN = re.compile(r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))")


def load_config(config_path: str | Path = "config.yaml", env_path: str | Path = ".env") -> AppConfig:
    config_file = Path(config_path)
    env_file = Path(env_path)

    if not config_file.exists():
        raise ConfigError(f"缺少 {config_file}，请复制 config.example.yaml 为 config.yaml 并填写配置。")
    if not env_file.exists():
        raise ConfigError(f"缺少 {env_file}，请复制 .env.example 为 .env 并填写 DEEPSEEK_API_KEY。")

    load_dotenv(env_file, override=False)
    config_data = _read_config_data(config_file)
    resolved_config_data = _resolve_env_variables(config_data)

    try:
        return AppConfig.model_validate(resolved_config_data)
    except ValidationError as exc:
        raise ConfigError(f"config.yaml 配置不完整或格式错误：{exc}") from exc


def _read_config_data(config_file: Path) -> dict[str, Any]:
    try:
        config_data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"config.yaml 不是有效的 YAML：{exc}") from exc

    if not isinstance(config_data, dict):
        raise ConfigError("config.yaml 必须是 YAML 映射。")
    return config_data


def _resolve_env_variables(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_env_variables(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env_variables(item) for item in value]
    if isinstance(value, str):
        return ENV_PATTERN.sub(_replace_env_variable, value)
    return value


def _replace_env_variable(match: re.Match[str]) -> str:
    name = match.group("braced") or match.group("plain")
    value = os.environ.get(name)
    if not value:
        raise ConfigError(f"环境变量 {name} 未设置，请在 .env 中填写。")
    return value
