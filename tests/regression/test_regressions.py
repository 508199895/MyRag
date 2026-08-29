from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from src.config import GenerationConfig, MarkdownHeaderSplitterConfig
from src.document_preparation.module import DocumentPreparationModule
from src.generation.module import GenerationModule
from src.service import RagService


@pytest.fixture(autouse=True)
def disable_llm_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(GenerationModule, "setup_llm", lambda self: None)


class FakeLlm(Runnable[object, AIMessage]):
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)

    def invoke(
        self, input: object, config: object | None = None, **kwargs: object
    ) -> AIMessage:
        return AIMessage(content=self.outputs.pop(0))


class FakeGeneration:
    def __init__(self) -> None:
        self.generated = False

    def route_query(self, question: str) -> str:
        return "general"

    def rewrite_query(self, question: str) -> str:
        return question

    def generate_answer(self, *args: object) -> str:
        self.generated = True
        return "不应生成"


class FakeDocs:
    def load_documents(self) -> list[Document]:
        return []

    def split_documents(self) -> list[Document]:
        return []

    def get_category_values(self) -> list[str]:
        return []

    def get_difficulty_values(self) -> list[str]:
        return []


class FakeIndex:
    def __init__(self, load_result: bool) -> None:
        self.load_result = load_result
        self.built = False
        self.saved = False

    def load_index(self) -> bool:
        return self.load_result

    def build_index(self, chunks: list[Document]) -> None:
        self.built = True

    def save_index(self) -> None:
        self.saved = True


def make_generation_config(tmp_path: Path) -> GenerationConfig:
    prompt_paths = {}
    for name, content in {
        "query_router": "问题：{query}",
        "query_rewrite": "问题：{query}",
        "step_by_step_answer": "问题：{question}\n内容：{context}",
        "basic_answer": "问题：{question}\n内容：{context}",
    }.items():
        path = tmp_path / f"{name}.md"
        path.write_text(content, encoding="utf-8")
        prompt_paths[name] = str(path)
    return GenerationConfig.model_validate(
        {
            "provider": "openai_compatible",
            "base_url": "https://example.invalid",
            "model_name": "fake-model",
            "api_key": "test-key",
            "stream": False,
            "temperature": 0.0,
            "max_tokens": 128,
            "query_router_prompt_template_path": prompt_paths["query_router"],
            "query_rewrite_prompt_template_path": prompt_paths["query_rewrite"],
            "step_by_step_answer_prompt_template_path": prompt_paths[
                "step_by_step_answer"
            ],
            "basic_answer_prompt_template_path": prompt_paths["basic_answer"],
        }
    )


def make_splitter_config() -> MarkdownHeaderSplitterConfig:
    return MarkdownHeaderSplitterConfig.model_validate(
        {
            "type": "markdown_header",
            "headers_to_split_on": [["#", "h1"], ["##", "h2"]],
            "strip_headers": False,
        }
    )


def test_invalid_route_json_defaults_to_general(tmp_path: Path) -> None:
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm(["不是 JSON"])

    assert module.route_query("随便问") == "general"


def test_illegal_route_intent_defaults_to_general(tmp_path: Path) -> None:
    module = GenerationModule(make_generation_config(tmp_path))
    module.llm = FakeLlm(['{"intent": "unknown"}'])

    assert module.route_query("推荐几个菜") == "general"


def test_empty_retrieval_returns_not_found_without_generation() -> None:
    service = RagService.__new__(RagService)
    service.generation_module = FakeGeneration()
    service.retrieval_module = type(
        "Retrieval", (), {"hybrid_search": lambda self, query: []}
    )()
    service.document_module = FakeDocs()
    service.config = type(
        "Config", (), {"generation": type("Gen", (), {"stream": False})()}
    )()

    assert service.ask("不存在的菜") == "未检索到相关内容。"
    assert service.generation_module.generated is False


def test_missing_parent_chunk_is_ignored_and_warns(tmp_path: Path, capsys) -> None:
    module = DocumentPreparationModule(tmp_path, make_splitter_config())
    module.documents = []

    ranked = module.get_ranked_parent_docs(
        [
            Document(
                page_content="orphan",
                metadata={"parent_id": "missing", "chunk_id": "c1"},
            )
        ]
    )

    assert ranked == []
    assert "无法回溯父文档" in capsys.readouterr().out


def test_existing_faiss_index_load_skips_rebuild_even_if_source_changes() -> None:
    service = RagService.__new__(RagService)
    service.document_module = FakeDocs()
    service.index_module = FakeIndex(load_result=True)

    service.build_knowledge_base()

    assert service.index_module.built is False
    assert service.index_module.saved is False
