from src.config import load_config
from src.document_preparation import DocumentPreparationModule
from src.generation import GenerationModule
from src.indexing import IndexConstructionModule
from src.retrieval.module import RetrievalModule


class RagService:
    """编排本地 Markdown 知识库的初始化、检索和问答流程。"""

    def startup(self) -> None:
        self.config = load_config()
        self.document_module = DocumentPreparationModule(
            self.config.data_path, self.config.splitter
        )
        self.index_module = IndexConstructionModule(
            self.config.index_save_path, self.config.embedding
        )
        self.generation_module = GenerationModule(self.config.generation)
        self.build_knowledge_base()
        self.retrieval_module = RetrievalModule(
            self.document_module.chunks,
            self.index_module.vectorstore,
            self.config.retrieval,
        )

    def build_knowledge_base(self) -> None:
        self.document_module.load_documents()
        chunks = self.document_module.split_documents()
        if not self.index_module.load_index():
            self.index_module.build_index(chunks)
            self.index_module.save_index()
        print("构建知识库完成")

    def ask(self, question: str) -> str:
        self._last_answer_printed_by_generation = False
        if not question.strip():
            return "未检索到相关内容。"
        intent = self.generation_module.route_query(question)
        query = (
            question
            if intent == "list"
            else self.generation_module.rewrite_query(question)
        )
        filters = self.extract_filters(question)
        chunks = (
            self.retrieval_module.metadata_filter_search(query, filters=filters)
            if filters
            else self.retrieval_module.hybrid_search(query)
        )
        if not chunks:
            return "未检索到相关内容。"

        parent_docs = self.document_module.get_ranked_parent_docs(chunks)
        if not parent_docs:
            return "未检索到相关内容。"

        self._last_answer_printed_by_generation = bool(
            self.config.generation.stream and intent != "list"
        )
        return self.generation_module.generate_answer(
            question, intent, parent_docs, self.config.generation.stream
        )

    def extract_filters(self, question: str) -> dict[str, list[str]]:
        filters: dict[str, list[str]] = {}
        categories = self._extract_metadata_values(
            question, self.document_module.get_category_values()
        )
        difficulties = self._extract_metadata_values(
            question, self.document_module.get_difficulty_values()
        )
        if categories:
            filters["category"] = categories
        if difficulties:
            filters["difficulty"] = difficulties
        print(f"元数据过滤条件：{filters}")
        return filters

    def _extract_metadata_values(self, question: str, values: list[str]) -> list[str]:
        return [value for value in values if value in question]

    def run_interactive(self) -> None:
        while True:
            print("您的问题是：")
            question = input().strip()
            if not question:
                print("请输入问题")
                continue
            if question in {"exit", "quit", "q"}:
                return
            answer = self.ask(question)
            if not getattr(self, "_last_answer_printed_by_generation", False):
                print(answer)

    def shutdown(self) -> None:
        return None
