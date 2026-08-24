from langchain_core.documents import Document

from src.config import RetrievalConfig
from src.retrieval.module import RetrievalModule


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, query):
        return self.docs


class FakeVectorstore:
    def __init__(self, docs):
        self.docs = docs
        self.kwargs = None

    def as_retriever(self, search_type, search_kwargs):
        self.kwargs = {"search_type": search_type, "search_kwargs": search_kwargs}
        return FakeRetriever(self.docs)


def make_retrieval_config(vector_top_k=5, bm25_top_k=5, hybrid_top_k=3):
    return RetrievalConfig.model_validate(
        {
            "vector": {"search_type": "similarity", "top_k": vector_top_k},
            "bm25": {"top_k": bm25_top_k},
            "hybrid": {"top_k": hybrid_top_k},
        }
    )


def test_vector_retriever_receives_configured_search_kwargs():
    docs = [Document(page_content="红烧肉", metadata={"chunk_id": "a"})]
    vectorstore = FakeVectorstore(docs)
    module = RetrievalModule(docs, vectorstore, make_retrieval_config(vector_top_k=5))

    assert module.vector_search("红烧肉") == docs
    assert vectorstore.kwargs["search_type"] == "similarity"
    assert vectorstore.kwargs["search_kwargs"] == {"k": 5}


def test_bm25_retriever_receives_configured_top_k():
    docs = [
        Document(page_content="红烧肉 炒糖色", metadata={"chunk_id": "a"}),
        Document(page_content="清炒菜心 蒜蓉", metadata={"chunk_id": "b"}),
    ]
    module = RetrievalModule(
        docs, FakeVectorstore([]), make_retrieval_config(bm25_top_k=1)
    )

    assert module.bm25_retriever.k == 1


def test_rrf_deduplicates_and_orders_by_score():
    a = Document(page_content="a", metadata={"chunk_id": "a"})
    b = Document(page_content="b", metadata={"chunk_id": "b"})
    c = Document(page_content="c", metadata={"chunk_id": "c"})
    module = RetrievalModule(
        [a, b, c], FakeVectorstore([a, b]), make_retrieval_config()
    )

    ranked = module._rrf_rerank([[a, b], [b, c]])

    assert [doc.metadata["chunk_id"] for doc in ranked] == ["b", "a", "c"]


def test_rrf_warns_and_uses_content_hash_when_chunk_id_is_missing(capsys):
    doc = Document(page_content="没有 chunk_id 的较长内容", metadata={})
    module = RetrievalModule([doc], FakeVectorstore([doc]), make_retrieval_config())

    assert module._rrf_rerank([[doc], [doc]]) == [doc]
    assert "检索结果缺少 chunk_id" in capsys.readouterr().out


def test_hybrid_search_uses_available_route_when_other_is_empty(monkeypatch):
    a = Document(page_content="红烧肉", metadata={"chunk_id": "a"})
    module = RetrievalModule(
        [a], FakeVectorstore([]), make_retrieval_config(hybrid_top_k=1)
    )
    monkeypatch.setattr(module, "bm25_search", lambda query: [a])

    assert module.hybrid_search("红烧肉") == [a]


def test_hybrid_search_returns_empty_when_both_routes_empty(monkeypatch):
    module = RetrievalModule([], FakeVectorstore([]), make_retrieval_config())
    monkeypatch.setattr(module, "bm25_search", lambda query: [])

    assert module.hybrid_search("什么都没有") == []


def test_hybrid_search_truncates_to_hybrid_top_k(monkeypatch):
    a = Document(page_content="a", metadata={"chunk_id": "a"})
    b = Document(page_content="b", metadata={"chunk_id": "b"})
    c = Document(page_content="c", metadata={"chunk_id": "c"})
    module = RetrievalModule(
        [a, b, c], FakeVectorstore([]), make_retrieval_config(hybrid_top_k=2)
    )
    monkeypatch.setattr(module, "vector_search", lambda query: [a, b])
    monkeypatch.setattr(module, "bm25_search", lambda query: [c])

    assert module.hybrid_search("红烧肉") == [a, c]
