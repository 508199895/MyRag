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

    def vector_search(self, query: str, top_k: int | None = None) -> list[Document]:
        if top_k is None:
            docs = self.vector_retriever.invoke(query)
        else:
            docs = self.vector_retriever.invoke(query, k=top_k)
        self._print_docs("向量检索结果", docs)
        return docs

    def bm25_search(self, query: str, top_k: int | None = None) -> list[Document]:
        if self.bm25_retriever is None:
            return []
        if top_k is None:
            docs = self.bm25_retriever.invoke(query)
        else:
            previous_top_k = self.bm25_retriever.k
            try:
                self.bm25_retriever.k = top_k
                docs = self.bm25_retriever.invoke(query)
            finally:
                self.bm25_retriever.k = previous_top_k
        self._print_docs("BM25 检索结果", docs)
        return docs

    def hybrid_search(self, query: str, top_k: int | None = None) -> list[Document]:
        reranked_docs = self._rrf_rerank(
            [
                self.vector_search(query, top_k=top_k),
                self.bm25_search(query, top_k=top_k),
            ]
        )
        if top_k is None:
            final_top_k = self.config.hybrid.top_k
        else:
            final_top_k = top_k
        docs = reranked_docs[:final_top_k]
        self._print_docs("混合检索结果", docs)
        return docs

    def metadata_filter_search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, object] | None = None,
    ) -> list[Document]:
        if top_k is None:
            effective_top_k = self.config.metadata_filter.top_k
        else:
            effective_top_k = top_k

        candidate_docs = self.hybrid_search(query, top_k=effective_top_k * 5)
        active_filters = filters or {}
        filtered_docs = [
            doc for doc in candidate_docs if self._matches_filters(doc, active_filters)
        ]
        docs = filtered_docs[:effective_top_k]
        self._print_docs("元数据过滤检索结果", docs)
        return docs

    def _matches_filters(self, doc: Document, filters: dict[str, object]) -> bool:
        for key, expected in filters.items():
            actual = doc.metadata.get(key)
            if isinstance(expected, list | tuple | set | frozenset):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    def _print_docs(self, stage: str, docs: list[Document]) -> None:
        print(f"{stage}：数量={len(docs)}")
        for rank, doc in enumerate(docs, start=1):
            metadata = doc.metadata
            print(
                f"{stage}："
                f"rank={rank}，"
                f"chunk_id={metadata.get('chunk_id')}，"
                f"parent_id={metadata.get('parent_id')}，"
                f"source={metadata.get('source')}"
            )

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
