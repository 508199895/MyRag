import sys
from pathlib import Path
from types import ModuleType

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from src import generation
from src.config import GenerationConfig
from src.generation import module as generation_module
from src.generation.module import GenerationModule, GenerationModuleError


class FakeLlm(Runnable[object, AIMessage]):
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = list(outputs)
        self.seen_messages: list[object] = []

    def invoke(
        self, input: object, config: object | None = None, **kwargs: object
    ) -> AIMessage:
        messages = self._capture_prompt_messages(input)
        self.seen_messages.append(messages)
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return AIMessage(content=str(value))

    def stream(self, input: object, config: object | None = None, **kwargs: object):
        messages = self._capture_prompt_messages(input)
        self.seen_messages.append(messages)
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        for part in str(value):
            yield AIMessageChunk(content=part)

    def _capture_prompt_messages(self, input: object) -> list[object]:
        if isinstance(input, list):
            raise TypeError("LLM 应通过 LCEL Prompt 链接收 PromptValue")
        if hasattr(input, "to_messages"):
            return input.to_messages()
        return [input]


def make_generation_config(tmp_path: Path, **overrides: object) -> GenerationConfig:
    paths = {}
    prompts = {
        "query_router": "用户问题：{query}",
        "query_rewrite": "用户问题：{query}",
        "generate_step_by_step_answer": (
            "DETAIL_PROMPT\n用户问题：{question}\n检索内容：{context}"
        ),
        "generate_basic_answer": (
            "BASIC_PROMPT\n用户问题：{question}\n检索内容：{context}"
        ),
    }
    for name, content in prompts.items():
        prompt_path = tmp_path / f"{name}.md"
        prompt_path.write_text(content, encoding="utf-8")
        paths[name] = str(prompt_path)
    data = {
        "provider": "openai_compatible",
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-v4-flash",
        "api_key": "test-key",
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 1024,
        "query_router_prompt_template_path": paths["query_router"],
        "query_rewrite_prompt_template_path": paths["query_rewrite"],
        "step_by_step_answer_prompt_template_path": paths[
            "generate_step_by_step_answer"
        ],
        "basic_answer_prompt_template_path": paths["generate_basic_answer"],
    }
    data.update(overrides)
    return GenerationConfig.model_validate(data)


def test_generation_module_checks_prompt_paths(tmp_path: Path) -> None:
    missing = tmp_path / "missing.md"
    config = make_generation_config(
        tmp_path, basic_answer_prompt_template_path=str(missing)
    )

    with pytest.raises(GenerationModuleError) as exc_info:
        GenerationModule(config)

    assert str(missing) in str(exc_info.value)


def test_generation_module_loads_all_configured_prompts(tmp_path: Path) -> None:
    module = GenerationModule(make_generation_config(tmp_path))

    assert set(module.prompts) == {
        "query_router",
        "query_rewrite",
        "generate_step_by_step_answer",
        "generate_basic_answer",
    }
    assert all(
        isinstance(prompt, ChatPromptTemplate) for prompt in module.prompts.values()
    )


def test_setup_llm_maps_generation_config_without_api_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeChatOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    fake_package = ModuleType("langchain_openai")
    fake_package.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_package)
    config = make_generation_config(tmp_path, stream=True)
    module = GenerationModule(config)

    module.setup_llm()

    assert isinstance(module.llm, FakeChatOpenAI)
    assert module.llm.kwargs == {
        "api_key": "test-key",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "temperature": 0.2,
        "max_tokens": 1024,
        "streaming": True,
    }


def test_setup_llm_wraps_initialization_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingChatOpenAI:
        def __init__(self, **kwargs: object) -> None:
            raise RuntimeError("构造失败")

    fake_package = ModuleType("langchain_openai")
    fake_package.ChatOpenAI = FailingChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_package)
    module = GenerationModule(make_generation_config(tmp_path))

    with pytest.raises(GenerationModuleError) as exc_info:
        module.setup_llm()

    error_message = str(exc_info.value)
    assert "provider=openai_compatible" in error_message
    assert "base_url=https://api.deepseek.com" in error_message
    assert "model_name=deepseek-v4-flash" in error_message
    assert "构造失败" in error_message


def test_generation_package_exports_module_interfaces() -> None:
    assert generation.GenerationModule is GenerationModule
    assert generation.GenerationModuleError is GenerationModuleError


def test_route_query_reads_json_intent(tmp_path: Path) -> None:
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm(['{"intent": "list", "reason": "推荐"}'])

    assert module.route_query("推荐几个菜") == "list"


def test_route_query_uses_pydantic_parser_for_intent_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parser_models: list[type[object]] = []

    class SpyPydanticOutputParser(PydanticOutputParser):
        def __init__(self, **kwargs: object) -> None:
            parser_models.append(kwargs["pydantic_object"])
            super().__init__(**kwargs)

    monkeypatch.setattr(
        generation_module, "PydanticOutputParser", SpyPydanticOutputParser
    )
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm(['{"intent": "detail"}'])

    assert module.route_query("怎么做红烧肉") == "detail"
    assert parser_models == [generation_module.RouteQueryOutput]


def test_route_query_defaults_general_on_invalid_json(tmp_path: Path) -> None:
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm(["不是 JSON"])

    assert module.route_query("随便问") == "general"


@pytest.mark.parametrize(
    "router_output",
    [
        '{"intent": ""}',
        '{"intent": "unknown"}',
        '{"intent": 123}',
        '{"not_intent": "list"}',
    ],
)
def test_route_query_defaults_general_on_invalid_intent(
    tmp_path: Path, router_output: str
) -> None:
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm([router_output])

    assert module.route_query("随便问") == "general"


def test_route_query_defaults_general_on_llm_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm([RuntimeError("network")])

    assert module.route_query("随便问") == "general"
    assert "查询路由失败" in capsys.readouterr().out


def test_route_query_does_not_print_on_invalid_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm(["不是 JSON"])

    assert module.route_query("随便问") == "general"
    assert capsys.readouterr().out == ""


def test_route_query_propagates_prompt_format_error(tmp_path: Path) -> None:
    module = GenerationModule(make_generation_config(tmp_path))
    module.prompts["query_router"] = ChatPromptTemplate.from_template(
        "缺少变量：{missing}"
    )
    module.llm = FakeLlm(["不会被调用"])

    with pytest.raises(KeyError):
        module.route_query("问题")


def test_rewrite_query_returns_raw_llm_content(tmp_path: Path) -> None:
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm(["改写后的问题"])

    assert module.rewrite_query("原问题") == "改写后的问题"


def test_rewrite_query_falls_back_to_original_question(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm([RuntimeError("network")])

    assert module.rewrite_query("原问题") == "原问题"
    assert "查询优化失败" in capsys.readouterr().out


def test_rewrite_query_propagates_prompt_format_error(tmp_path: Path) -> None:
    module = GenerationModule(make_generation_config(tmp_path))
    module.prompts["query_rewrite"] = ChatPromptTemplate.from_template(
        "缺少变量：{missing}"
    )
    module.llm = FakeLlm(["不会被调用"])

    with pytest.raises(KeyError):
        module.rewrite_query("问题")


def test_build_context_preserves_parent_order_and_metadata_defaults(
    tmp_path: Path,
) -> None:
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm(["回答"])
    docs = [
        Document(page_content="第二份", metadata={"dish_name": "乙"}),
        Document(page_content="第一份", metadata={"dish_name": "甲", "source": "a.md"}),
    ]

    assert module.generate_answer("问题", "detail", docs, stream=False) == "回答"
    messages = module.llm.seen_messages[-1]
    context = messages[0].content.split("检索内容：", maxsplit=1)[1]

    assert context.index("[文档1]") < context.index("[文档2]")
    assert "菜名：乙\n类别：未知\n难度：未知\n来源：\n内容：\n第二份" in context
    assert "菜名：甲\n类别：未知\n难度：未知\n来源：a.md\n内容：\n第一份" in context


def test_build_context_truncates_after_assembling_in_order(tmp_path: Path) -> None:
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm(["回答"])
    docs = [
        Document(page_content="A" * 4000, metadata={"dish_name": "甲"}),
        Document(page_content="B" * 4000, metadata={"dish_name": "乙"}),
    ]

    assert module.generate_answer("问题", "detail", docs, stream=False) == "回答"
    messages = module.llm.seen_messages[-1]
    context = messages[0].content.split("检索内容：", maxsplit=1)[1]

    assert len(context) == 6000
    assert context.startswith("[文档1]")
    assert "[文档2]\n菜名：乙" in context
    assert context.count("B") < 4000


def test_generate_answer_builds_context_from_parent_documents(tmp_path: Path) -> None:
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm(["回答"])
    docs = [
        Document(
            page_content="制作步骤",
            metadata={
                "dish_name": "红烧肉",
                "category": "荤菜",
                "difficulty": "中等",
                "source": "meat_dish/红烧肉.md",
            },
        )
    ]

    answer = module.generate_answer("怎么做红烧肉", "detail", docs, stream=False)

    assert answer == "回答"
    messages = module.llm.seen_messages[-1]
    rendered = "\n".join(message.content for message in messages)
    assert "DETAIL_PROMPT" in rendered
    assert "[文档1]" in rendered
    assert "菜名：红烧肉" in rendered
    assert "类别：荤菜" in rendered
    assert "难度：中等" in rendered
    assert "来源：meat_dish/红烧肉.md" in rendered
    assert "内容：\n制作步骤" in rendered


def test_generate_answer_uses_basic_prompt_for_general(tmp_path: Path) -> None:
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm(["回答"])

    assert module.generate_answer("问题", "general", [], stream=False) == "回答"
    message = module.llm.seen_messages[-1][0]
    assert "BASIC_PROMPT" in message.content
    assert "用户问题：问题" in message.content


def test_generate_answer_propagates_missing_question_prompt_variable(
    tmp_path: Path,
) -> None:
    module = GenerationModule(make_generation_config(tmp_path))
    module.prompts["generate_basic_answer"] = ChatPromptTemplate.from_template(
        "检索内容：{context}"
    )
    module.llm = FakeLlm(["不会被调用"])

    with pytest.raises(KeyError):
        module.generate_answer("问题", "general", [], stream=False)


def test_generate_answer_propagates_missing_context_prompt_variable(
    tmp_path: Path,
) -> None:
    module = GenerationModule(make_generation_config(tmp_path))
    module.prompts["generate_step_by_step_answer"] = ChatPromptTemplate.from_template(
        "用户问题：{question}"
    )
    module.llm = FakeLlm(["不会被调用"])

    with pytest.raises(KeyError):
        module.generate_answer("问题", "detail", [], stream=False)


def test_generate_answer_formats_deduplicated_list_without_calling_llm(
    tmp_path: Path,
) -> None:
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm([RuntimeError("list should not call llm")])
    docs = [
        Document(page_content="a", metadata={"dish_name": "西红柿炒鸡蛋"}),
        Document(page_content="b", metadata={"dish_name": "凉拌黄瓜"}),
        Document(page_content="c", metadata={"dish_name": "西红柿炒鸡蛋"}),
        Document(page_content="d", metadata={"dish_name": "紫菜蛋花汤"}),
    ]

    assert module.generate_answer("推荐几个菜", "list", docs, stream=False) == (
        "为您推荐以下菜品：\n1. 西红柿炒鸡蛋\n2. 凉拌黄瓜\n3. 紫菜蛋花汤"
    )
    assert module.llm.seen_messages == []


def test_generate_answer_returns_error_when_list_docs_have_no_dish_name(
    tmp_path: Path,
) -> None:
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm([RuntimeError("list should not call llm")])
    docs = [
        Document(page_content="a", metadata={}),
        Document(page_content="b", metadata={"dish_name": "  "}),
    ]

    assert module.generate_answer("推荐几个菜", "list", docs, stream=False) == (
        "检索文档无菜名，请检查索引是否损坏"
    )


def test_generate_answer_returns_non_stream_error_message_on_llm_failure(
    tmp_path: Path,
) -> None:
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm([RuntimeError("network")])

    assert module.generate_answer("问题", "detail", [], stream=False) == (
        "LLM API 调用失败，请检查 API Key、base_url、模型名或网络连接"
    )


def test_generate_answer_streams_and_returns_concatenated_content(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm(["流式回答"])

    assert module.generate_answer("问题", "detail", [], stream=True) == "流式回答"
    assert capsys.readouterr().out == "流式回答\n"


def test_generate_answer_stream_failure_keeps_fragments_and_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class FailingStreamLlm(FakeLlm):
        def stream(self, input: object, config: object | None = None, **kwargs: object):
            messages = self._capture_prompt_messages(input)
            self.seen_messages.append(messages)
            yield AIMessageChunk(content="部分")
            raise RuntimeError("network")

    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FailingStreamLlm([])

    result = module.generate_answer("问题", "detail", [], stream=True)

    error = "回答生成失败，请重试或检查配置/网络。"
    assert result == "部分" + error
    assert capsys.readouterr().out == "部分" + error + "\n"
