from langchain_core.documents import Document

from src.config import RetrievalConfig
from src.retrieval.module import RetrievalModule


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def invoke(self, query, **kwargs):
        self.calls.append({"query": query, "kwargs": kwargs})
        if "k" in kwargs:
            return self.docs[: kwargs["k"]]
        return self.docs


class FakeVectorstore:
    def __init__(self, docs):
        self.docs = docs
        self.kwargs = None
        self.retriever = None

    def as_retriever(self, search_type, search_kwargs):
        self.kwargs = {"search_type": search_type, "search_kwargs": search_kwargs}
        self.retriever = FakeRetriever(self.docs)
        return self.retriever


def make_retrieval_config(vector_top_k=5, bm25_top_k=5, hybrid_top_k=5):
    return RetrievalConfig.model_validate(
        {
            "vector": {"search_type": "similarity", "top_k": vector_top_k},
            "bm25": {"top_k": bm25_top_k},
            "hybrid": {"top_k": hybrid_top_k},
            "metadata_filter": {"top_k": 5},
        }
    )


def test_vector_retriever_receives_configured_search_kwargs():
    docs = [Document(page_content="红烧肉", metadata={"chunk_id": "a"})]
    vectorstore = FakeVectorstore(docs)
    module = RetrievalModule(docs, vectorstore, make_retrieval_config(vector_top_k=5))

    assert module.vector_search("红烧肉") == docs
    assert vectorstore.kwargs["search_type"] == "similarity"
    assert vectorstore.kwargs["search_kwargs"] == {"k": 5}
    assert vectorstore.retriever.calls == [{"query": "红烧肉", "kwargs": {}}]


def test_vector_search_passes_temporary_top_k_without_rebuilding_retriever():
    docs = [
        Document(page_content="a", metadata={"chunk_id": "a"}),
        Document(page_content="b", metadata={"chunk_id": "b"}),
        Document(page_content="c", metadata={"chunk_id": "c"}),
    ]
    vectorstore = FakeVectorstore(docs)
    module = RetrievalModule(docs, vectorstore, make_retrieval_config(vector_top_k=5))

    assert module.vector_search("红烧肉", top_k=2) == docs[:2]
    assert vectorstore.kwargs["search_kwargs"] == {"k": 5}
    assert vectorstore.retriever.calls == [{"query": "红烧肉", "kwargs": {"k": 2}}]


def test_bm25_retriever_receives_configured_top_k():
    docs = [
        Document(page_content="红烧肉 炒糖色", metadata={"chunk_id": "a"}),
        Document(page_content="清炒菜心 蒜蓉", metadata={"chunk_id": "b"}),
    ]
    module = RetrievalModule(
        docs, FakeVectorstore([]), make_retrieval_config(bm25_top_k=1)
    )

    assert module.bm25_retriever.k == 1


def test_bm25_search_temporarily_overrides_top_k_and_restores_previous_value():
    docs = [
        Document(page_content="红烧肉 炒糖色", metadata={"chunk_id": "a"}),
        Document(page_content="红烧肉 收汁", metadata={"chunk_id": "b"}),
        Document(page_content="红烧肉 焯水", metadata={"chunk_id": "c"}),
    ]
    module = RetrievalModule(
        docs, FakeVectorstore([]), make_retrieval_config(bm25_top_k=3)
    )
    module.bm25_retriever.k = 2

    results = module.bm25_search("红烧肉", top_k=1)

    assert len(results) == 1
    assert module.bm25_retriever.k == 2


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
    monkeypatch.setattr(module, "bm25_search", lambda query, top_k=None: [a])

    assert module.hybrid_search("红烧肉") == [a]


def test_hybrid_search_returns_empty_when_both_routes_empty(monkeypatch):
    module = RetrievalModule([], FakeVectorstore([]), make_retrieval_config())
    monkeypatch.setattr(module, "bm25_search", lambda query, top_k=None: [])

    assert module.hybrid_search("什么都没有") == []


def test_hybrid_search_truncates_to_hybrid_top_k(monkeypatch):
    a = Document(page_content="a", metadata={"chunk_id": "a"})
    b = Document(page_content="b", metadata={"chunk_id": "b"})
    c = Document(page_content="c", metadata={"chunk_id": "c"})
    module = RetrievalModule(
        [a, b, c], FakeVectorstore([]), make_retrieval_config(hybrid_top_k=2)
    )
    monkeypatch.setattr(module, "vector_search", lambda query, top_k=None: [a, b])
    monkeypatch.setattr(module, "bm25_search", lambda query, top_k=None: [c])

    assert module.hybrid_search("红烧肉") == [a, c]


def test_hybrid_search_passes_temporary_top_k_to_both_routes_and_final_limit(
    monkeypatch,
):
    calls = []
    docs = [
        Document(page_content="a", metadata={"chunk_id": "a"}),
        Document(page_content="b", metadata={"chunk_id": "b"}),
        Document(page_content="c", metadata={"chunk_id": "c"}),
        Document(page_content="d", metadata={"chunk_id": "d"}),
    ]
    module = RetrievalModule(
        docs, FakeVectorstore([]), make_retrieval_config(hybrid_top_k=1)
    )

    def fake_vector_search(query, top_k=None):
        calls.append(("vector", query, top_k))
        return [docs[0], docs[1], docs[2]]

    def fake_bm25_search(query, top_k=None):
        calls.append(("bm25", query, top_k))
        return [docs[3]]

    monkeypatch.setattr(module, "vector_search", fake_vector_search)
    monkeypatch.setattr(module, "bm25_search", fake_bm25_search)

    assert module.hybrid_search("红烧肉", top_k=2) == [docs[0], docs[3]]
    assert calls == [("vector", "红烧肉", 2), ("bm25", "红烧肉", 2)]


def test_metadata_filter_search_uses_config_top_k_and_filters_all_conditions(
    monkeypatch,
):
    docs = [
        Document(
            page_content="a",
            metadata={"chunk_id": "a", "category": "素菜", "difficulty": "简单"},
        ),
        Document(
            page_content="b",
            metadata={"chunk_id": "b", "category": "荤菜", "difficulty": "简单"},
        ),
        Document(
            page_content="c",
            metadata={"chunk_id": "c", "category": "汤品", "difficulty": "简单"},
        ),
        Document(
            page_content="d",
            metadata={"chunk_id": "d", "category": "素菜", "difficulty": "困难"},
        ),
    ]
    module = RetrievalModule(docs, FakeVectorstore([]), make_retrieval_config())
    calls = []

    def fake_hybrid_search(query, top_k=None):
        calls.append((query, top_k))
        return docs

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)

    results = module.metadata_filter_search(
        "家常菜",
        filters={"category": ["素菜", "荤菜"], "difficulty": "简单"},
    )

    assert results == [docs[0], docs[1]]
    assert calls == [("家常菜", 25)]


def test_metadata_filter_search_uses_temporary_top_k_and_truncates(monkeypatch):
    docs = [
        Document(page_content="a", metadata={"chunk_id": "a", "category": "素菜"}),
        Document(page_content="b", metadata={"chunk_id": "b", "category": "素菜"}),
        Document(page_content="c", metadata={"chunk_id": "c", "category": "素菜"}),
    ]
    module = RetrievalModule(docs, FakeVectorstore([]), make_retrieval_config())
    calls = []

    def fake_hybrid_search(query, top_k=None):
        calls.append((query, top_k))
        return docs

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)

    results = module.metadata_filter_search(
        "素菜",
        top_k=2,
        filters={"category": "素菜"},
    )

    assert results == docs[:2]
    assert calls == [("素菜", 10)]


def test_metadata_filter_search_empty_filters_only_truncates(monkeypatch):
    docs = [
        Document(page_content="a", metadata={"chunk_id": "a", "category": "素菜"}),
        Document(page_content="b", metadata={"chunk_id": "b", "category": "荤菜"}),
        Document(page_content="c", metadata={"chunk_id": "c", "category": "汤品"}),
    ]
    module = RetrievalModule(docs, FakeVectorstore([]), make_retrieval_config())
    monkeypatch.setattr(module, "hybrid_search", lambda query, top_k=None: docs)

    assert module.metadata_filter_search("菜", top_k=2, filters={}) == docs[:2]
