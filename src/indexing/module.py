from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import EmbeddingConfig


class IndexConstructionError(Exception):
    """Raised when embedding or FAISS index construction cannot continue."""


class IndexConstructionModule:
    def __init__(self, index_save_path: str | Path, embedding_config: EmbeddingConfig):
        self.index_save_path = Path(index_save_path)
        self.embedding_config = embedding_config
        self.embedding_model = self._load_embedding_model()
        self.vectorstore = None

    def _load_embedding_model(self):
        try:
            return HuggingFaceEmbeddings(
                model_name=self.embedding_config.model_name,
                cache_folder="model/bge-small-zh-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"batch_size": 32, "normalize_embeddings": True},
                show_progress=True,
            )
        except Exception as exc:
            raise IndexConstructionError(
                f"Embedding 模型加载失败：{self.embedding_config.model_name}；原因：{exc}"
            ) from exc

    def load_index(self) -> bool:
        faiss_file = self.index_save_path / "index.faiss"
        pkl_file = self.index_save_path / "index.pkl"
        if not faiss_file.is_file() or not pkl_file.is_file():
            return False
        try:
            self.vectorstore = FAISS.load_local(
                str(self.index_save_path),
                self.embedding_model,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            print("索引加载失败，将重新构建索引。")
            return False
        return True

    def build_index(self, chunks: list[Document]) -> None:
        self.vectorstore = FAISS.from_documents(chunks, self.embedding_model)

    def save_index(self) -> None:
        try:
            self.vectorstore.save_local(str(self.index_save_path))
        except Exception as exc:
            raise IndexConstructionError(
                f"FAISS 索引保存失败：{self.index_save_path}；原因：{exc}"
            ) from exc
