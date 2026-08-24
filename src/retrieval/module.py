from __future__ import annotations

import hashlib

import jieba
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.config import RetrievalConfig


class RetrievalModule:
    """Coordinates vector, BM25, and RRF-based hybrid retrieval over chunks."""

    def __init__(self, chunks: list[Document], vectorstore, config: RetrievalConfig):
        self.chunks = chunks
        self.vectorstore = vectorstore
        self.config = config
        self.vector_retriever = None
        self.bm25_retriever = None
        self._init_retrievers()

    def _init_retrievers(self) -> None:
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type=self.config.vector.search_type,
            search_kwargs={"k": self.config.vector.top_k},
        )
        if self.chunks:
            self.bm25_retriever = BM25Retriever.from_documents(
                self.chunks,
                preprocess_func=jieba.lcut,
                k=self.config.bm25.top_k,
            )

    def vector_search(self, query: str) -> list[Document]:
        return self.vector_retriever.invoke(query)

    def bm25_search(self, query: str) -> list[Document]:
        if self.bm25_retriever is None:
            return []
        return self.bm25_retriever.invoke(query)

    def hybrid_search(self, query: str) -> list[Document]:
        reranked_docs = self._rrf_rerank(
            [self.vector_search(query), self.bm25_search(query)]
        )
        return reranked_docs[: self.config.hybrid.top_k]

    def _rrf_rerank(self, ranked_results: list[list[Document]]) -> list[Document]:
        rrf_k = 60
        rrf_scores: dict[str, float] = {}
        original_docs: dict[str, Document] = {}

        for results in ranked_results:
            for rank, doc in enumerate(results, start=1):
                chunk_id = doc.metadata.get("chunk_id")
                if chunk_id:
                    doc_key = str(chunk_id)
                else:
                    print(
                        "检索结果缺少 chunk_id，已使用 page_content 的 MD5 作为临时去重键。"
                    )
                    doc_key = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()

                original_docs.setdefault(doc_key, doc)
                rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + 1 / (rrf_k + rank)

        reranked_docs = [
            original_docs[doc_key]
            for doc_key, _ in sorted(
                rrf_scores.items(), key=lambda item: item[1], reverse=True
            )
        ]
        return reranked_docs
