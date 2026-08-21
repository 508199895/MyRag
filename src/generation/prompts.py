from __future__ import annotations

from pathlib import Path
from string import Formatter

DEFAULT_PROMPT_PATH = Path("docs/prompts/llm_generator.md")
_REQUIRED_VARIABLES = {"context", "question"}


class PromptTemplateError(Exception):
    """Raised when a prompt template cannot be loaded or rendered."""


def load_prompt_template(path: str | Path = DEFAULT_PROMPT_PATH) -> str:
    """Load and validate a prompt template from a UTF-8 text file."""
    prompt_path = Path(path)
    if not prompt_path.is_file():
        raise PromptTemplateError(f"缺少 Prompt 模板：{prompt_path}")

    try:
        template = prompt_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PromptTemplateError(f"无法读取 Prompt 模板 {prompt_path}：{exc}") from exc

    _validate_prompt_template(template, prompt_path)
    return template


def render_prompt(template: str, context: str, question: str) -> str:
    """Inject retrieved context and the user question into a prompt template."""
    _validate_prompt_template(template)
    try:
        return template.format(context=context, question=question)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise PromptTemplateError(f"Prompt 模板渲染失败：{exc}") from exc


def _validate_prompt_template(template: str, path: Path | None = None) -> None:
    try:
        variables = {
            field_name
            for _, field_name, _, _ in Formatter().parse(template)
            if field_name is not None and field_name
        }
    except ValueError as exc:
        location = f" {path}" if path is not None else ""
        raise PromptTemplateError(f"Prompt 模板{location}格式无效：{exc}") from exc

    unsupported = variables - _REQUIRED_VARIABLES
    if unsupported:
        unsupported_names = "、".join(sorted(unsupported))
        location = f" {path}" if path is not None else ""
        raise PromptTemplateError(
            f"Prompt 模板{location}包含不支持的变量：{unsupported_names}"
        )

    missing = _REQUIRED_VARIABLES - variables
    if missing:
        missing_names = "、".join(sorted(missing))
        location = f" {path}" if path is not None else ""
        raise PromptTemplateError(f"Prompt 模板{location}缺少变量：{missing_names}")
