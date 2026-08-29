import pytest
from langchain_core.documents import Document

from src.config import EmbeddingConfig
from src.indexing.module import IndexConstructionError, IndexConstructionModule


class DeterministicEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0, 0.0]


def make_embedding_config():
    return EmbeddingConfig.model_validate({"model_name": "fake-model"})


def test_load_index_returns_false_when_index_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        IndexConstructionModule,
        "_load_embedding_model",
        lambda self: DeterministicEmbeddings(),
    )
    module = IndexConstructionModule(tmp_path / "missing", make_embedding_config())

    assert module.load_index() is False


def test_load_index_returns_false_when_index_files_are_incomplete(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        IndexConstructionModule,
        "_load_embedding_model",
        lambda self: DeterministicEmbeddings(),
    )
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "index.faiss").write_bytes(b"not-real")

    module = IndexConstructionModule(index_dir, make_embedding_config())

    assert module.load_index() is False


def test_load_index_returns_false_and_prints_message_when_faiss_load_fails(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        IndexConstructionModule,
        "_load_embedding_model",
        lambda self: DeterministicEmbeddings(),
    )
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "index.faiss").write_bytes(b"not-real")
    (index_dir / "index.pkl").write_bytes(b"not-real")
    module = IndexConstructionModule(index_dir, make_embedding_config())

    assert module.load_index() is False
    assert "索引加载失败，将重新构建索引。" in capsys.readouterr().out


def test_build_index_uses_passed_chunks_without_loading_documents(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        IndexConstructionModule,
        "_load_embedding_model",
        lambda self: DeterministicEmbeddings(),
    )
    captured = {}

    class FakeFAISS:
        @staticmethod
        def from_documents(chunks, embedding_model):
            captured["chunks"] = chunks
            captured["embedding_model"] = embedding_model
            return "vectorstore"

    monkeypatch.setattr("src.indexing.module.FAISS", FakeFAISS)
    module = IndexConstructionModule(tmp_path / "index", make_embedding_config())
    chunks = [
        Document(
            page_content="番茄炒蛋", metadata={"chunk_id": "c1", "parent_id": "p1"}
        )
    ]

    module.build_index(chunks)

    assert captured["chunks"] is chunks
    assert captured["embedding_model"] is module.embedding_model
    assert module.vectorstore == "vectorstore"


def test_embedding_load_failure_mentions_model_name_and_reason(tmp_path, monkeypatch):
    def fail_load(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("src.indexing.module.HuggingFaceEmbeddings", fail_load)

    with pytest.raises(IndexConstructionError, match="fake-model.*boom"):
        IndexConstructionModule(tmp_path / "index", make_embedding_config())


def test_save_index_failure_mentions_index_save_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        IndexConstructionModule,
        "_load_embedding_model",
        lambda self: DeterministicEmbeddings(),
    )

    class BrokenVectorstore:
        def save_local(self, path):
            raise OSError("disk full")

    module = IndexConstructionModule(tmp_path / "index", make_embedding_config())
    module.vectorstore = BrokenVectorstore()

    with pytest.raises(IndexConstructionError) as exc_info:
        module.save_index()
    message = str(exc_info.value)
    assert str(tmp_path / "index") in message
    assert "disk full" in message
