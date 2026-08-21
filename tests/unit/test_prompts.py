from pathlib import Path

import pytest

from src.generation.prompts import (
    PromptTemplateError,
    load_prompt_template,
    render_prompt,
)

DEFAULT_PROMPT_PATH = Path("docs/prompts/llm_generator.md")


def test_default_prompt_template_exists_and_contains_required_variables() -> None:
    assert DEFAULT_PROMPT_PATH.is_file()

    template = load_prompt_template(DEFAULT_PROMPT_PATH)

    assert "{context}" in template
    assert "{question}" in template
    assert "严格依据下面的检索内容回答" in template
    assert "无法从资料中确定" in template


@pytest.mark.parametrize(
    ("template", "missing_variable"),
    [
        ("用户问题：{question}", "context"),
        ("检索内容：{context}", "question"),
    ],
)
def test_load_prompt_template_rejects_missing_required_variable(
    tmp_path: Path, template: str, missing_variable: str
) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text(template, encoding="utf-8")

    with pytest.raises(PromptTemplateError) as exc_info:
        load_prompt_template(prompt_path)

    assert missing_variable in str(exc_info.value)


def test_load_prompt_template_reports_missing_file_path(tmp_path: Path) -> None:
    prompt_path = tmp_path / "missing.md"

    with pytest.raises(PromptTemplateError) as exc_info:
        load_prompt_template(prompt_path)

    assert str(prompt_path) in str(exc_info.value)


def test_load_prompt_template_wraps_decode_error(tmp_path: Path) -> None:
    prompt_path = tmp_path / "invalid-utf8.md"
    prompt_path.write_bytes(b"\xff")

    with pytest.raises(PromptTemplateError) as exc_info:
        load_prompt_template(prompt_path)

    assert str(prompt_path) in str(exc_info.value)


@pytest.mark.parametrize(
    "template", ["{context.foo} {question}", "{context} {question[0]}"]
)
def test_render_prompt_rejects_unsupported_field_access(template: str) -> None:
    with pytest.raises(PromptTemplateError):
        render_prompt(template, context="资料", question="问题")


def test_render_prompt_wraps_format_error_after_validation_passes() -> None:
    with pytest.raises(PromptTemplateError, match="渲染失败"):
        render_prompt("{context:bad} {question}", context="资料", question="问题")


def test_load_prompt_template_wraps_formatter_parse_error(tmp_path: Path) -> None:
    prompt_path = tmp_path / "invalid-format.md"
    prompt_path.write_text("检索内容：{context\n用户问题：{question}", encoding="utf-8")

    with pytest.raises(PromptTemplateError) as exc_info:
        load_prompt_template(prompt_path)

    assert str(prompt_path) in str(exc_info.value)
    assert "格式无效" in str(exc_info.value)


def test_render_prompt_injects_context_and_question() -> None:
    rendered = render_prompt(
        "检索内容：{context}\n用户问题：{question}",
        context="资料中的答案",
        question="用户的问题",
    )

    assert "检索内容：资料中的答案" in rendered
    assert "用户问题：用户的问题" in rendered
