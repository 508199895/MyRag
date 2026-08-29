from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import ClassVar

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from src.config import MarkdownHeaderSplitterConfig


class DocumentPreparationError(Exception):
    """Raised when the local Markdown document library cannot be prepared."""


class DocumentPreparationModule:
    """Load, enrich, split, and rank local Markdown documents."""

    CATEGORY_MAPPING: ClassVar[dict[str, str]] = {
        "meat_dish": "荤菜",
        "vegetable_dish": "素菜",
        "soup": "汤品",
        "dessert": "甜品",
        "breakfast": "早餐",
        "staple": "主食",
        "aquatic": "水产",
        "condiment": "调料",
        "drink": "饮品",
    }
    DIFFICULTY_MAPPING: ClassVar[dict[int, str]] = {
        5: "非常困难",
        4: "困难",
        3: "中等",
        2: "简单",
        1: "非常简单",
    }

    def __init__(
        self, data_path: str | Path, splitter_config: MarkdownHeaderSplitterConfig
    ) -> None:
        self.data_path = Path(data_path)
        self.splitter_config = splitter_config
        self.documents: list[Document] = []
        self.chunks: list[Document] = []
        self.child_parent_map: dict[str, str] = {}

    def load_documents(self) -> list[Document]:
        if not self.data_path.exists():
            raise DocumentPreparationError(f"路径不存在：{self.data_path}")

        documents: list[Document] = []
        markdown_paths = sorted(self.data_path.rglob("*.md"))
        for path in markdown_paths:
            content = path.read_text(encoding="utf-8")
            if not content.strip():
                print(f"文档内容为空：{path}")
                continue
            source = path.relative_to(self.data_path).as_posix()
            documents.append(
                Document(page_content=content, metadata={"source": source})
            )

        self.documents = documents
        if not self.documents:
            print("文档库为空")
            raise DocumentPreparationError("文档库为空")
        self._enhance_metadata()
        return self.documents

    def _enhance_metadata(self) -> None:
        for document in self.documents:
            source = document.metadata["source"]
            path_parts = Path(source).parts
            category = next(
                (
                    self.CATEGORY_MAPPING[part]
                    for part in path_parts
                    if part in self.CATEGORY_MAPPING
                ),
                "未知",
            )
            stars = re.search(r"★+", document.page_content)
            difficulty = self.DIFFICULTY_MAPPING.get(
                len(stars.group(0)) if stars else 0, "未知"
            )
            document.metadata.update(
                {
                    "parent_id": hashlib.md5(source.encode("utf-8")).hexdigest(),
                    "doc_type": "parent",
                    "category": category,
                    "dish_name": Path(source).stem,
                    "difficulty": difficulty,
                }
            )

    def split_documents(self) -> list[Document]:
        if not self.documents:
            raise DocumentPreparationError("文档库为空")

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.splitter_config.headers_to_split_on,
            strip_headers=self.splitter_config.strip_headers,
        )
        chunks: list[Document] = []
        self.child_parent_map = {}
        for parent in self.documents:
            parent_id = parent.metadata["parent_id"]
            split_chunks = splitter.split_text(parent.page_content)
            chunk_index = 0
            for split_chunk in split_chunks:
                page_content = split_chunk.page_content.replace("  \n", "\n\n")
                if not page_content.strip():
                    continue
                chunk_id = hashlib.md5(f"{parent_id}{chunk_index}".encode()).hexdigest()
                metadata = {
                    **parent.metadata,
                    "doc_type": "child",
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                }
                chunk = Document(page_content=page_content, metadata=metadata)
                chunks.append(chunk)
                self.child_parent_map[chunk_id] = parent_id
                chunk_index += 1

        if not chunks:
            raise DocumentPreparationError("文档库为空")
        self.chunks = chunks
        return self.chunks

    def get_parent_source(self, parent_id: str) -> str | None:
        """Map a parent document id back to its Markdown source path."""
        for document in self.documents:
            if document.metadata.get("parent_id") == parent_id:
                source = document.metadata.get("source")
                return str(source) if source is not None else None
        return None

    def get_chunk_source(self, chunk_id: str) -> str | None:
        """Map a child chunk id back to its parent Markdown source path."""
        for chunk in self.chunks:
            if chunk.metadata.get("chunk_id") == chunk_id:
                source = chunk.metadata.get("source")
                return str(source) if source is not None else None
        parent_id = self.child_parent_map.get(chunk_id)
        return self.get_parent_source(parent_id) if parent_id is not None else None

    def get_category_values(self) -> list[str]:
        return list(self.CATEGORY_MAPPING.values())

    def get_difficulty_values(self) -> list[str]:
        return list(self.DIFFICULTY_MAPPING.values())

    def get_ranked_parent_docs(self, chunks: list[Document]) -> list[Document]:
        parents_by_id = {
            document.metadata.get("parent_id"): document for document in self.documents
        }
        hit_counts: dict[str, int] = {}
        first_positions: dict[str, int] = {}
        print("父文档回溯")
        for position, chunk in enumerate(chunks):
            parent_id = chunk.metadata.get("parent_id")
            chunk_id = chunk.metadata.get("chunk_id")
            source = chunk.metadata.get("source")
            print(
                "父文档回溯输入："
                f"rank={position + 1}，"
                f"chunk_id={chunk_id}，"
                f"parent_id={parent_id}，"
                f"source={source}"
            )
            if parent_id not in parents_by_id:
                print(f"无法回溯父文档：parent_id={parent_id}")
                continue
            hit_counts[parent_id] = hit_counts.get(parent_id, 0) + 1
            first_positions.setdefault(parent_id, position)

        ranked_ids = sorted(
            hit_counts,
            key=lambda parent_id: (-hit_counts[parent_id], first_positions[parent_id]),
        )
        for rank, parent_id in enumerate(ranked_ids, start=1):
            print(
                "父文档回溯成功："
                f"rank={rank}，"
                f"parent_id={parent_id}，"
                f"source={self.get_parent_source(str(parent_id))}，"
                f"hit_count={hit_counts[parent_id]}"
            )
        return [parents_by_id[parent_id] for parent_id in ranked_ids]
