from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from src.config import GenerationConfig


class GenerationModuleError(Exception):
    """Raised when generation module setup or execution fails."""


class GenerationModule:
    """Load generation prompts and initialize the configured chat model."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self.prompts = self._load_prompts()
        self.llm: Any | None = None

    def setup_llm(self) -> None:
        """Initialize the OpenAI-compatible LangChain chat model."""
        try:
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                streaming=self.config.stream,
            )
        except Exception as exc:
            raise GenerationModuleError(
                "LLM 初始化失败："
                f"provider={self.config.provider}，"
                f"base_url={self.config.base_url}，"
                f"model_name={self.config.model_name}，"
                f"原因：{exc}"
            ) from exc

    def _load_prompts(self) -> dict[str, ChatPromptTemplate]:
        paths = {
            "query_router": self.config.query_router_prompt_template_path,
            "query_rewrite": self.config.query_rewrite_prompt_template_path,
            "step_by_step_answer": self.config.step_by_step_answer_prompt_template_path,
            "basic_answer": self.config.basic_answer_prompt_template_path,
        }
        prompts: dict[str, ChatPromptTemplate] = {}
        for name, path_value in paths.items():
            path = Path(path_value)
            if not path.is_file():
                raise GenerationModuleError(f"缺少 Prompt 模板：{path}")
            try:
                prompts[name] = ChatPromptTemplate.from_template(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, ValueError) as exc:
                raise GenerationModuleError(
                    f"无法加载 Prompt 模板 {path}：{exc}"
                ) from exc
        return prompts
