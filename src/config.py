from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ConfigError(Exception):
    """Raised when local runtime configuration is missing or invalid."""


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MarkdownHeaderSplitterConfig(ConfigModel):
    model_config = ConfigDict(extra="allow")

    type: str = Field(description="文本切分器类型。")
    headers_to_split_on: list[tuple[str, str]] | None = Field(
        default=None,
        description="Markdown header 与 metadata key 的映射。",
    )
    strip_headers: bool | None = Field(
        default=None, description="切分后是否移除 Markdown header。"
    )


class EmbeddingConfig(ConfigModel):
    model_config = ConfigDict(extra="allow")

    model_name: str = Field(description="Embedding 模型名。")


class SimilarityRetrievalConfig(ConfigModel):
    model_config = ConfigDict(extra="allow")

    type: str = Field(description="检索类型。")
    top_k: int = Field(description="返回的检索结果数量。")


class GenerationConfig(ConfigModel):
    provider: str = Field(description="LLM provider 名称。")
    base_url: str = Field(description="OpenAI-compatible API base URL。")
    model_name: str = Field(description="LLM 模型名。")
    api_key: str = Field(description="LLM API key。")
    prompt_template_path: str = Field(description="Prompt 模板路径。")
    temperature: float = Field(description="生成温度。")
    max_tokens: int = Field(description="最大输出 token 数。")


SplitterConfig = MarkdownHeaderSplitterConfig
RetrievalConfig = SimilarityRetrievalConfig


class AppConfig(ConfigModel):
    data_path: str = Field(description="本地文档库路径。")
    index_save_path: str = Field(description="FAISS 索引持久化目录。")
    splitter: SplitterConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig


ENV_PATTERN = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)


def load_config(
    config_path: str | Path = "config.yaml", env_path: str | Path = ".env"
) -> AppConfig:
    config_file = Path(config_path)
    env_file = Path(env_path)

    if not config_file.exists():
        raise ConfigError(
            f"缺少 {config_file}，请复制 config.example.yaml 为 config.yaml 并填写配置。"
        )
    if not env_file.exists():
        raise ConfigError(
            f"缺少 {env_file}，请复制 .env.example 为 .env 并填写 DEEPSEEK_API_KEY。"
        )

    load_dotenv(env_file, override=False)
    config_data = _read_config_data(config_file)
    resolved_config_data = _resolve_env_variables(config_data)

    try:
        return AppConfig.model_validate(resolved_config_data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_errors(exc)) from exc


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


def _format_validation_errors(exc: ValidationError) -> str:
    lines = ["config.yaml 配置不完整或格式错误："]
    for error in exc.errors(include_url=False):
        field_path = ".".join(str(part) for part in error["loc"])
        reason = error.get("msg", "配置无效")
        current_value = error.get("input")
        lines.append(f"- {field_path}：{reason}；当前值：{current_value!r}")
    return "\n".join(lines)
