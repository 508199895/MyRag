import pytest
from langchain_core.documents import Document

from src.config import MarkdownHeaderSplitterConfig
from src.document_preparation.module import (
    DocumentPreparationError,
    DocumentPreparationModule,
)

REQUIRED_PARENT_METADATA = {
    "parent_id",
    "doc_type",
    "source",
    "category",
    "dish_name",
    "difficulty",
}
REQUIRED_CHUNK_METADATA = REQUIRED_PARENT_METADATA | {"chunk_id", "chunk_index"}


def make_splitter_config() -> MarkdownHeaderSplitterConfig:
    return MarkdownHeaderSplitterConfig.model_validate(
        {
            "type": "markdown_header",
            "headers_to_split_on": [("#", "h1"), ("##", "h2"), ("###", "h3")],
            "strip_headers": False,
        }
    )


def test_load_documents_recurses_md_only_and_enhances_parent_metadata(tmp_path):
    root_content = "# 根目录菜\n无星级"
    root_doc = tmp_path / "根目录菜.md"
    root_doc.write_text(root_content, encoding="utf-8")
    dessert_content = "# 双皮奶\n难度：★\n一级目录"
    dessert_doc = tmp_path / "dessert" / "双皮奶.md"
    dessert_doc.parent.mkdir(parents=True)
    dessert_doc.write_text(dessert_content, encoding="utf-8")
    aquatic_content = "# 红烧鲤鱼\n难度：★★★\n哨兵：糖色"
    doc_path = tmp_path / "dishes" / "aquatic" / "红烧鲤鱼.md"
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text(aquatic_content, encoding="utf-8")
    (tmp_path / "dishes" / "aquatic" / "ignore.txt").write_text("txt", encoding="utf-8")

    module = DocumentPreparationModule(tmp_path, make_splitter_config())
    docs = module.load_documents()

    assert len(docs) == 3
    by_source = {doc.metadata["source"]: doc for doc in docs}
    assert set(by_source) == {
        "根目录菜.md",
        "dessert/双皮奶.md",
        "dishes/aquatic/红烧鲤鱼.md",
    }
    aquatic = by_source["dishes/aquatic/红烧鲤鱼.md"].metadata
    assert by_source["dishes/aquatic/红烧鲤鱼.md"].page_content == aquatic_content
    assert set(aquatic) >= REQUIRED_PARENT_METADATA
    assert aquatic["doc_type"] == "parent"
    assert aquatic["category"] == "水产"
    assert aquatic["dish_name"] == "红烧鲤鱼"
    assert aquatic["difficulty"] == "中等"
    assert len(aquatic["parent_id"]) == 32
    assert by_source["根目录菜.md"].metadata["category"] == "未知"
    assert by_source["根目录菜.md"].page_content == root_content
    assert by_source["dessert/双皮奶.md"].metadata["category"] == "甜品"


def test_load_documents_rejects_missing_data_path(tmp_path):
    module = DocumentPreparationModule(tmp_path / "missing", make_splitter_config())
    with pytest.raises(DocumentPreparationError, match="路径不存在"):
        module.load_documents()


def test_load_documents_rejects_library_without_markdown_files(tmp_path, capsys):
    (tmp_path / "ignore.txt").write_text("not markdown", encoding="utf-8")
    module = DocumentPreparationModule(tmp_path, make_splitter_config())
    with pytest.raises(DocumentPreparationError, match="文档库为空"):
        module.load_documents()
    assert "文档库为空" in capsys.readouterr().out


def test_load_documents_skips_empty_and_whitespace_markdown_files(tmp_path, capsys):
    empty_doc = tmp_path / "dishes" / "soup" / "空.md"
    blank_doc = tmp_path / "dishes" / "soup" / "空白.md"
    valid_doc = tmp_path / "dishes" / "soup" / "番茄蛋汤.md"
    valid_doc.parent.mkdir(parents=True)
    empty_doc.write_text("", encoding="utf-8")
    blank_doc.write_text(" \n\t\n", encoding="utf-8")
    valid_doc.write_text("# 番茄蛋汤\n难度：★★", encoding="utf-8")
    docs = DocumentPreparationModule(tmp_path, make_splitter_config()).load_documents()
    assert [doc.metadata["source"] for doc in docs] == ["dishes/soup/番茄蛋汤.md"]
    output = capsys.readouterr().out
    assert "文档内容为空" in output
    assert str(empty_doc) in output
    assert str(blank_doc) in output


def test_load_documents_rejects_when_all_markdown_files_are_empty(tmp_path, capsys):
    empty_doc = tmp_path / "dishes" / "soup" / "空.md"
    blank_doc = tmp_path / "dishes" / "soup" / "空白.md"
    blank_doc.parent.mkdir(parents=True)
    empty_doc.write_text("", encoding="utf-8")
    blank_doc.write_text(" \n", encoding="utf-8")
    with pytest.raises(DocumentPreparationError, match="文档库为空"):
        DocumentPreparationModule(tmp_path, make_splitter_config()).load_documents()
    output = capsys.readouterr().out
    assert "文档内容为空" in output
    assert "文档库为空" in output


@pytest.mark.parametrize(
    ("stars", "expected"),
    [
        ("★", "非常简单"),
        ("★★", "简单"),
        ("★★★", "中等"),
        ("★★★★", "困难"),
        ("★★★★★", "非常困难"),
        ("★★★★★★", "未知"),
        ("", "未知"),
    ],
)
def test_load_documents_maps_difficulty_from_first_star_run(tmp_path, stars, expected):
    doc_path = tmp_path / "dishes" / "breakfast" / "测试菜.md"
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text(f"# 测试菜\n难度：{stars}\n说明：无其他星级", encoding="utf-8")
    docs = DocumentPreparationModule(tmp_path, make_splitter_config()).load_documents()
    assert docs[0].metadata["difficulty"] == expected


def test_load_documents_generates_stable_parent_id_for_same_source(tmp_path):
    doc_path = tmp_path / "dishes" / "aquatic" / "白灼虾.md"
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text("# 白灼虾\n难度：★★", encoding="utf-8")
    first = DocumentPreparationModule(tmp_path, make_splitter_config()).load_documents()
    second = DocumentPreparationModule(
        tmp_path, make_splitter_config()
    ).load_documents()
    assert first[0].metadata["parent_id"] == second[0].metadata["parent_id"]


def test_load_documents_maps_unknown_category_when_no_path_segment_matches(tmp_path):
    doc_path = tmp_path / "dishes" / "unknown_label" / "神秘菜.md"
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text("# 神秘菜\n难度：★", encoding="utf-8")
    docs = DocumentPreparationModule(tmp_path, make_splitter_config()).load_documents()
    assert docs[0].metadata["category"] == "未知"


def test_split_documents_generates_child_metadata_and_map(tmp_path):
    doc_path = tmp_path / "dishes" / "vegetable_dish" / "清炒菜心.md"
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text("# 清炒菜心\n## 食材\n菜心\n## 做法\n快炒", encoding="utf-8")
    module = DocumentPreparationModule(tmp_path, make_splitter_config())
    parents = module.load_documents()
    chunks = module.split_documents()
    assert parents and chunks
    assert len(module.child_parent_map) == len(chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(
        range(len(chunks))
    )
    seen_chunk_ids = set()
    for chunk in chunks:
        metadata = chunk.metadata
        assert set(metadata) >= REQUIRED_CHUNK_METADATA
        assert metadata["doc_type"] == "child"
        assert metadata["parent_id"] == parents[0].metadata["parent_id"]
        assert len(metadata["chunk_id"]) == 32
        assert metadata["chunk_id"] not in seen_chunk_ids
        assert module.child_parent_map[metadata["chunk_id"]] == metadata["parent_id"]
        seen_chunk_ids.add(metadata["chunk_id"])
    assert all(chunk.page_content.strip() for chunk in chunks)


def test_split_documents_preserves_chunk_order_and_content_membership(tmp_path):
    expected_chunks = [
        "# 白灼虾\n\n简介：鲜虾快速汆烫，保留原味。",
        "## 食材\n\n鲜虾、姜片、葱段。",
        "## 做法\n\n水沸后下虾，变红后捞出。",
        "## 蘸料\n\n生抽、香醋、姜末混合。",
    ]
    doc_path = tmp_path / "dishes" / "aquatic" / "白灼虾.md"
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text("\n\n".join(expected_chunks), encoding="utf-8")
    module = DocumentPreparationModule(tmp_path, make_splitter_config())
    module.load_documents()
    chunks = module.split_documents()
    assert [chunk.page_content.strip() for chunk in chunks] == expected_chunks
    assert all(
        chunk.page_content.strip() in module.documents[0].page_content
        for chunk in chunks
    )
    merged_chunks = "\n".join(chunk.page_content for chunk in chunks)
    assert merged_chunks.index("简介：鲜虾快速汆烫") < merged_chunks.index("鲜虾、姜片")
    assert merged_chunks.index("鲜虾、姜片") < merged_chunks.index("水沸后下虾")
    assert merged_chunks.index("水沸后下虾") < merged_chunks.index("生抽、香醋")


def test_get_ranked_parent_docs_orders_by_hit_count_then_first_position(
    tmp_path, capsys
):
    module = DocumentPreparationModule(tmp_path, make_splitter_config())
    parent_a = Document(page_content="A", metadata={"parent_id": "a", "dish_name": "A"})
    parent_b = Document(page_content="B", metadata={"parent_id": "b", "dish_name": "B"})
    parent_c = Document(page_content="C", metadata={"parent_id": "c", "dish_name": "C"})
    module.documents = [parent_a, parent_b, parent_c]
    chunks = [
        Document(page_content="b1", metadata={"parent_id": "b", "chunk_id": "b1"}),
        Document(page_content="a1", metadata={"parent_id": "a", "chunk_id": "a1"}),
        Document(page_content="a2", metadata={"parent_id": "a", "chunk_id": "a2"}),
        Document(page_content="b2", metadata={"parent_id": "b", "chunk_id": "b2"}),
        Document(page_content="c1", metadata={"parent_id": "c", "chunk_id": "c1"}),
        Document(page_content="c2", metadata={"parent_id": "c", "chunk_id": "c2"}),
        Document(page_content="a3", metadata={"parent_id": "a", "chunk_id": "a3"}),
        Document(page_content="missing", metadata={"parent_id": "x", "chunk_id": "x1"}),
    ]
    assert module.get_ranked_parent_docs(chunks) == [parent_a, parent_b, parent_c]
    assert "无法回溯父文档" in capsys.readouterr().out
