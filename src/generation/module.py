from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.config import GenerationConfig


class GenerationModuleError(Exception):
    """Raised when generation module setup or execution fails."""


class RouteQueryOutput(BaseModel):
    """Structured route output validated from the router LLM response."""

    intent: Literal["list", "detail", "general"]


class GenerationModule:
    """Load generation prompts and initialize the configured chat model."""

    MAX_CONTEXT_CHARS = 6000

    def __init__(self, config: GenerationConfig) -> None:
        print("生成模块初始化")
        self.config = config
        self.prompts = self._load_prompts()
        self.llm: Any | None = None
        self.setup_llm()
        print("生成模块初始化成功")

    def setup_llm(self) -> None:
        """Initialize the OpenAI-compatible LangChain chat model."""
        print("LLM 初始化")
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
            print("LLM 初始化成功")
        except Exception as exc:
            print(f"LLM 初始化失败：{exc}")
            raise GenerationModuleError(
                "LLM 初始化失败："
                f"provider={self.config.provider}，"
                f"base_url={self.config.base_url}，"
                f"model_name={self.config.model_name}，"
                f"原因：{exc}"
            ) from exc

    def route_query(self, question: str) -> Literal["list", "detail", "general"]:
        """Classify a question, falling back to the general intent on failure."""
        print("查询路由")
        prompt = self.prompts["query_router"]
        prompt.format_prompt(query=question)
        chain = (
            prompt | self.llm | PydanticOutputParser(pydantic_object=RouteQueryOutput)
        )
        try:
            parsed = chain.invoke({"query": question})
        except OutputParserException:
            print("查询路由失败：模型输出无法解析，使用 general")
            return "general"
        except Exception as exc:  # noqa: BLE001 - LLM failures must degrade safely.
            print(f"查询路由失败：{exc}")
            return "general"
        print(f"查询路由成功：{parsed.intent}")
        return parsed.intent

    def rewrite_query(self, question: str) -> str:
        """Rewrite a question for retrieval, preserving the original on failure."""
        print("查询优化")
        prompt = self.prompts["query_rewrite"]
        prompt.format_prompt(query=question)
        chain = prompt | self.llm | StrOutputParser()
        try:
            rewritten = chain.invoke({"query": question})
        except Exception as exc:  # noqa: BLE001 - LLM failures must degrade safely.
            print(f"查询优化失败：{exc}")
            return question
        print(f"查询优化成功：{rewritten}")
        return rewritten

    def _build_context(self, docs: list[Document]) -> str:
        """Render parent documents in retrieval order and cap the total context."""
        print("构建生成上下文")
        sections = []
        for index, doc in enumerate(docs, start=1):
            metadata = doc.metadata
            print(
                "生成上下文文档："
                f"rank={index}，"
                f"parent_id={metadata.get('parent_id')}，"
                f"source={metadata.get('source')}，"
                f"dish_name={metadata.get('dish_name')}"
            )
            sections.append(
                "\n".join(
                    [
                        f"[文档{index}]",
                        f"菜名：{metadata.get('dish_name', '未知')}",
                        f"类别：{metadata.get('category', '未知')}",
                        f"难度：{metadata.get('difficulty', '未知')}",
                        f"来源：{metadata.get('source', '')}",
                        "内容：",
                        doc.page_content,
                    ]
                )
            )
        context = "\n".join(sections)[: self.MAX_CONTEXT_CHARS]
        print(f"构建生成上下文成功：文档数={len(docs)}，字符数={len(context)}")
        return context

    def generate_answer(
        self,
        question: str,
        intent: Literal["list", "detail", "general"],
        docs: list[Document],
        stream: bool,
    ) -> str:
        """Generate an answer from parent documents or format a list response."""
        print(f"生成路由：intent={intent}，stream={stream}")
        if intent == "list":
            dish_names = []
            seen = set()
            for doc in docs:
                value = doc.metadata.get("dish_name")
                if value is None:
                    continue
                dish_name = str(value).strip()
                if dish_name and dish_name not in seen:
                    seen.add(dish_name)
                    dish_names.append(dish_name)
            if not dish_names:
                print("生成失败：列表结果缺少菜名")
                return "检索文档无菜名，请检查索引是否损坏"
            answer = "为您推荐以下菜品：\n" + "\n".join(
                f"{index}. {dish_name}"
                for index, dish_name in enumerate(dish_names, start=1)
            )
            print("生成成功")
            return answer

        context = self._build_context(docs)
        prompt_name = (
            "generate_step_by_step_answer"
            if intent == "detail"
            else "generate_basic_answer"
        )
        prompt = self.prompts[prompt_name]
        missing_variables = {"question", "context"} - set(prompt.input_variables)
        if missing_variables:
            raise KeyError(
                f"回答 Prompt 缺少变量：{', '.join(sorted(missing_variables))}"
            )
        prompt.format_prompt(question=question, context=context)
        chain = prompt | self.llm | StrOutputParser()
        if not stream:
            try:
                answer = chain.invoke({"question": question, "context": context})
            except Exception:  # noqa: BLE001 - LLM failures use the public fallback.
                print("生成失败：LLM API 调用失败")
                return "LLM API 调用失败，请检查 API Key、base_url、模型名或网络连接"
            print("生成成功")
            return answer

        answer_parts = []
        try:
            for chunk in chain.stream({"question": question, "context": context}):
                answer_parts.append(chunk)
                print(chunk, end="", flush=True)
            print()
            print("生成成功")
            return "".join(answer_parts)
        except Exception:  # noqa: BLE001 - preserve partial stream output on failure.
            error = "回答生成失败，请重试或检查配置/网络。"
            print(error)
            return "".join(answer_parts) + error

    def _load_prompts(self) -> dict[str, ChatPromptTemplate]:
        print("加载 Prompt 模板")
        paths = {
            "query_router": self.config.query_router_prompt_template_path,
            "query_rewrite": self.config.query_rewrite_prompt_template_path,
            "generate_step_by_step_answer": self.config.step_by_step_answer_prompt_template_path,
            "generate_basic_answer": self.config.basic_answer_prompt_template_path,
        }
        prompts: dict[str, ChatPromptTemplate] = {}
        for name, path_value in paths.items():
            path = Path(path_value)
            if not path.is_file():
                print(f"加载 Prompt 模板失败：{path}")
                raise GenerationModuleError(f"缺少 Prompt 模板：{path}")
            try:
                prompts[name] = ChatPromptTemplate.from_template(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, ValueError) as exc:
                print(f"加载 Prompt 模板失败：{path}，{exc}")
                raise GenerationModuleError(
                    f"无法加载 Prompt 模板 {path}：{exc}"
                ) from exc
        print("加载 Prompt 模板成功")
        return prompts
