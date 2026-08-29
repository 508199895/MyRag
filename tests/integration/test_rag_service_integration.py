from pathlib import Path

import yaml
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from src.config import load_config
from src.indexing.module import IndexConstructionModule
from src.service import RagService


class DeterministicEmbeddings:
    """提供稳定向量，避免测试下载或调用本地 embedding 模型。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)


class FakeLlm(Runnable[object, AIMessage]):
    """记录调用并依次返回预设结果，确保测试不触达真实 API。"""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls = 0

    def invoke(
        self, input: object, config: object | None = None, **kwargs: object
    ) -> AIMessage:
        self.calls += 1
        return AIMessage(content=self.outputs.pop(0))


def write_test_runtime(tmp_path: Path) -> tuple[Path, Path]:
    prompt_paths = {}
    prompts = {
        "query_router": "用户问题：{query}",
        "query_rewrite": "用户问题：{query}",
        "generate_step_by_step_answer": "问题：{question}\n上下文：{context}",
        "generate_basic_answer": "问题：{question}\n上下文：{context}",
    }
    for name, content in prompts.items():
        path = tmp_path / f"{name}.md"
        path.write_text(content, encoding="utf-8")
        prompt_paths[name] = str(path)

    env_path = tmp_path / ".env"
    env_path.write_text("DEEPSEEK_API_KEY=test-key\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data_path": str(Path(__file__).parents[1] / "fixtures" / "cook_docs"),
                "index_save_path": str(tmp_path / "faiss"),
                "splitter": {
                    "type": "markdown_header",
                    "headers_to_split_on": [["#", "h1"], ["##", "h2"], ["###", "h3"]],
                    "strip_headers": False,
                },
                "embedding": {"model_name": "fake-model"},
                "retrieval": {
                    "vector": {"search_type": "similarity", "top_k": 3},
                    "bm25": {"top_k": 3},
                    "hybrid": {"top_k": 3},
                    "metadata_filter": {"top_k": 3},
                },
                "generation": {
                    "provider": "openai_compatible",
                    "base_url": "https://example.invalid",
                    "model_name": "fake-model",
                    "api_key": "$DEEPSEEK_API_KEY",
                    "stream": False,
                    "temperature": 0.0,
                    "max_tokens": 128,
                    "query_router_prompt_template_path": prompt_paths["query_router"],
                    "query_rewrite_prompt_template_path": prompt_paths["query_rewrite"],
                    "step_by_step_answer_prompt_template_path": prompt_paths[
                        "generate_step_by_step_answer"
                    ],
                    "basic_answer_prompt_template_path": prompt_paths[
                        "generate_basic_answer"
                    ],
                },
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path, env_path


def test_rag_service_builds_and_reloads_knowledge_base_with_fake_llm(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, env_path = write_test_runtime(tmp_path)
    fake_llms: list[FakeLlm] = []

    def setup_fake_llm(module) -> None:
        fake_llm = FakeLlm(['{"intent": "detail"}', "红烧肉做法", "假的回答"])
        fake_llms.append(fake_llm)
        module.llm = fake_llm

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        "src.service.load_config", lambda: load_config(config_path, env_path)
    )
    monkeypatch.setattr(
        IndexConstructionModule,
        "_load_embedding_model",
        lambda self: DeterministicEmbeddings(),
    )
    monkeypatch.setattr("src.service.GenerationModule.setup_llm", setup_fake_llm)

    first = RagService()
    first.startup()

    assert (tmp_path / "faiss" / "index.faiss").is_file()
    assert (tmp_path / "faiss" / "index.pkl").is_file()
    assert first.ask("红烧肉怎么做") == "假的回答"
    assert fake_llms[0].calls == 3

    def fail_if_rebuilt(self, chunks) -> None:
        raise AssertionError("已有索引不应重新构建")

    monkeypatch.setattr(IndexConstructionModule, "build_index", fail_if_rebuilt)
    second = RagService()
    second.startup()

    assert second.index_module.vectorstore is not None
    assert second.retrieval_module.bm25_retriever.k == 3
