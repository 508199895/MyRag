from langchain_core.documents import Document

from src.config import EmbeddingConfig
from src.indexing.module import IndexConstructionModule
from tests.unit.test_indexing import DeterministicEmbeddings


def test_build_save_and_load_faiss_index(tmp_path, monkeypatch):
    monkeypatch.setattr(
        IndexConstructionModule,
        "_load_embedding_model",
        lambda self: DeterministicEmbeddings(),
    )
    config = EmbeddingConfig.model_validate({"model_name": "fake-model"})
    chunks = [
        Document(page_content="红烧肉", metadata={"chunk_id": "c1", "parent_id": "p1"})
    ]

    first = IndexConstructionModule(tmp_path / "faiss", config)
    first.build_index(chunks)
    first.save_index()

    assert (tmp_path / "faiss" / "index.faiss").is_file()
    assert (tmp_path / "faiss" / "index.pkl").is_file()

    second = IndexConstructionModule(tmp_path / "faiss", config)
    assert second.load_index() is True
    assert second.vectorstore is not None
