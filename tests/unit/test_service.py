from langchain_core.documents import Document

from src.service import RagService


class FakeDocs:
    def __init__(self):
        self.documents = []
        self.chunks = [
            Document(page_content="chunk", metadata={"parent_id": "p", "chunk_id": "c"})
        ]
        self.parent = Document(page_content="parent", metadata={"parent_id": "p"})
        self.ranked_calls = []

    def load_documents(self):
        self.documents = [self.parent]
        return self.documents

    def split_documents(self):
        return self.chunks

    def get_ranked_parent_docs(self, chunks):
        self.ranked_calls.append(chunks)
        return [self.parent] if chunks else []

    def get_category_values(self):
        return ["荤菜", "素菜", "汤品"]

    def get_difficulty_values(self):
        return ["非常困难", "困难", "中等", "简单", "非常简单"]


class FakeIndex:
    def __init__(self, load_result=False):
        self.vectorstore = object()
        self.load_result = load_result
        self.built = False
        self.saved = False

    def load_index(self):
        return self.load_result

    def build_index(self, chunks):
        self.built = True

    def save_index(self):
        self.saved = True


class FakeRetrieval:
    def __init__(self, results=None):
        self.queries = []
        self.metadata_filter_calls = []
        self.results = (
            results
            if results is not None
            else [
                Document(
                    page_content="chunk", metadata={"parent_id": "p", "chunk_id": "c"}
                )
            ]
        )

    def hybrid_search(self, query):
        self.queries.append(query)
        return self.results

    def metadata_filter_search(self, query, top_k=None, filters=None):
        self.metadata_filter_calls.append(
            {"query": query, "top_k": top_k, "filters": filters}
        )
        return self.results


class FakeGeneration:
    def __init__(self, intent="detail"):
        self.intent = intent
        self.rewritten = []
        self.generated = []

    def route_query(self, question):
        return self.intent

    def rewrite_query(self, question):
        self.rewritten.append(question)
        return "优化问题"

    def generate_answer(self, question, intent, docs, stream):
        self.generated.append((question, intent, docs, stream))
        return f"{intent}:{stream}:{len(docs)}"


def test_build_knowledge_base_rebuilds_when_index_missing(capsys):
    service = RagService.__new__(RagService)
    service.document_module = FakeDocs()
    service.index_module = FakeIndex(load_result=False)

    service.build_knowledge_base()

    assert service.index_module.built is True
    assert service.index_module.saved is True
    assert "构建知识库完成" in capsys.readouterr().out


def test_startup_initializes_modules_in_lifecycle_order(monkeypatch):
    events = []
    data_path = object()
    splitter = object()
    index_save_path = object()
    embedding = object()
    generation = object()
    retrieval = object()
    chunks = [object()]
    vectorstore = object()
    config = type(
        "Config",
        (),
        {
            "data_path": data_path,
            "splitter": splitter,
            "index_save_path": index_save_path,
            "embedding": embedding,
            "generation": generation,
            "retrieval": retrieval,
        },
    )()

    class StartupDocs:
        def __init__(self, received_data_path, received_splitter):
            events.append(("document_module", received_data_path, received_splitter))
            self.chunks = chunks

    class StartupIndex:
        def __init__(self, received_index_path, received_embedding):
            events.append(("index_module", received_index_path, received_embedding))
            self.vectorstore = vectorstore

    class StartupGeneration:
        def __init__(self, received_generation):
            events.append(("generation_module", received_generation))

    class StartupRetrieval:
        def __init__(self, received_chunks, received_vectorstore, received_retrieval):
            events.append(
                (
                    "retrieval_module",
                    received_chunks,
                    received_vectorstore,
                    received_retrieval,
                )
            )

    monkeypatch.setattr(
        "src.service.load_config", lambda: events.append(("load_config",)) or config
    )
    monkeypatch.setattr("src.service.DocumentPreparationModule", StartupDocs)
    monkeypatch.setattr("src.service.IndexConstructionModule", StartupIndex)
    monkeypatch.setattr("src.service.GenerationModule", StartupGeneration)
    monkeypatch.setattr("src.service.RetrievalModule", StartupRetrieval)
    monkeypatch.setattr(
        RagService,
        "build_knowledge_base",
        lambda self: events.append(("build_knowledge_base",)),
    )

    service = RagService()
    service.startup()

    assert events == [
        ("load_config",),
        ("document_module", data_path, splitter),
        ("index_module", index_save_path, embedding),
        ("generation_module", generation),
        ("build_knowledge_base",),
        ("retrieval_module", chunks, vectorstore, retrieval),
    ]


def test_ask_runs_single_turn_without_terminal_input():
    service = RagService.__new__(RagService)
    service.generation_module = FakeGeneration()
    service.retrieval_module = FakeRetrieval()
    service.document_module = FakeDocs()
    service.config = type(
        "Config", (), {"generation": type("Gen", (), {"stream": False})()}
    )()

    assert service.ask("怎么做红烧肉") == "detail:False:1"
    assert service.retrieval_module.queries == ["优化问题"]
    assert service.generation_module.generated == [
        ("怎么做红烧肉", "detail", [service.document_module.parent], False)
    ]


def test_ask_short_circuits_when_question_is_blank():
    service = RagService.__new__(RagService)
    service.generation_module = FakeGeneration()
    service.retrieval_module = FakeRetrieval()
    service.document_module = FakeDocs()
    service.config = type(
        "Config", (), {"generation": type("Gen", (), {"stream": False})()}
    )()

    assert service.ask("   ") == "未检索到相关内容。"
    assert service.retrieval_module.queries == []
    assert service.retrieval_module.metadata_filter_calls == []
    assert service.generation_module.rewritten == []
    assert service.generation_module.generated == []


def test_extract_filters_returns_categories_and_difficulties_from_question():
    service = RagService.__new__(RagService)
    service.document_module = FakeDocs()

    assert service.extract_filters("推荐简单的素菜和荤菜") == {
        "category": ["荤菜", "素菜"],
        "difficulty": ["简单"],
    }


def test_extract_filters_keeps_all_matched_difficulty_values():
    service = RagService.__new__(RagService)
    service.document_module = FakeDocs()

    assert service.extract_filters("推荐非常简单的早餐") == {
        "difficulty": ["简单", "非常简单"]
    }


def test_ask_uses_metadata_filter_search_when_filters_exist():
    service = RagService.__new__(RagService)
    service.generation_module = FakeGeneration()
    service.retrieval_module = FakeRetrieval()
    service.document_module = FakeDocs()
    service.config = type(
        "Config", (), {"generation": type("Gen", (), {"stream": False})()}
    )()

    assert service.ask("怎么做简单的素菜") == "detail:False:1"
    assert service.retrieval_module.queries == []
    assert service.retrieval_module.metadata_filter_calls == [
        {
            "query": "优化问题",
            "top_k": None,
            "filters": {"category": ["素菜"], "difficulty": ["简单"]},
        }
    ]


def test_ask_uses_hybrid_search_when_filters_empty():
    service = RagService.__new__(RagService)
    service.generation_module = FakeGeneration()
    service.retrieval_module = FakeRetrieval()
    service.document_module = FakeDocs()
    service.config = type(
        "Config", (), {"generation": type("Gen", (), {"stream": False})()}
    )()

    assert service.ask("怎么做红烧肉") == "detail:False:1"
    assert service.retrieval_module.queries == ["优化问题"]
    assert service.retrieval_module.metadata_filter_calls == []


def test_ask_uses_original_query_for_list_intent():
    service = RagService.__new__(RagService)
    service.generation_module = FakeGeneration(intent="list")
    service.retrieval_module = FakeRetrieval()
    service.document_module = FakeDocs()
    service.config = type(
        "Config", (), {"generation": type("Gen", (), {"stream": False})()}
    )()

    assert service.ask("推荐几个菜") == "list:False:1"
    assert service.generation_module.rewritten == []
    assert service.retrieval_module.queries == ["推荐几个菜"]


def test_ask_short_circuits_when_retrieval_empty():
    service = RagService.__new__(RagService)
    service.generation_module = FakeGeneration()
    service.retrieval_module = FakeRetrieval(results=[])
    service.document_module = FakeDocs()
    service.config = type(
        "Config", (), {"generation": type("Gen", (), {"stream": False})()}
    )()

    assert service.ask("不存在的菜") == "未检索到相关内容。"
    assert service.document_module.ranked_calls == []
    assert service.generation_module.generated == []


def test_ask_short_circuits_when_parent_docs_empty():
    service = RagService.__new__(RagService)
    service.generation_module = FakeGeneration()
    service.retrieval_module = FakeRetrieval()
    service.document_module = FakeDocs()
    service.document_module.get_ranked_parent_docs = lambda chunks: []
    service.config = type(
        "Config", (), {"generation": type("Gen", (), {"stream": False})()}
    )()

    assert service.ask("孤立 chunk") == "未检索到相关内容。"
    assert service.generation_module.generated == []
