import sys
from pathlib import Path
from types import ModuleType

import pytest
from langchain_core.prompts import ChatPromptTemplate

from src import generation
from src.config import GenerationConfig
from src.generation.module import GenerationModule, GenerationModuleError


def make_generation_config(tmp_path: Path, **overrides: object) -> GenerationConfig:
    paths = {}
    prompts = {
        "query_router": "用户问题：{query}",
        "query_rewrite": "用户问题：{query}",
        "step_by_step_answer": "用户问题：{question}\n检索内容：{context}",
        "basic_answer": "用户问题：{question}\n检索内容：{context}",
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
        "step_by_step_answer_prompt_template_path": paths["step_by_step_answer"],
        "basic_answer_prompt_template_path": paths["basic_answer"],
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
        "step_by_step_answer",
        "basic_answer",
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
