from pathlib import Path

from research_agent.retrieval.ingestion import load_documents


def test_load_documents_is_sorted_and_ignores_unsupported_files(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("second document", encoding="utf-8")
    (tmp_path / "a.md").write_text("# first document\nfirst document", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("indexed, False", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [doc.source_path.name for doc in documents] == ["a.md", "b.txt"]
    assert [doc.doc_type for doc in documents] == ["md", "txt"]


def test_load_documents_is_deterministic_across_runs(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("stable content", encoding="utf-8")

    first_run = load_documents(tmp_path)
    second_run = load_documents(tmp_path)

    assert [doc.doc_id for doc in first_run] == [doc.doc_id for doc in second_run]


def test_load_documents_returns_empty_list_for_empty_corpus(tmp_path: Path) -> None:
    assert load_documents(tmp_path) == []


def test_document_id_is_stable_when_corpus_root_moves(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()

    (first_root / "paper.md").write_text("Same content", encoding="utf-8")
    (second_root / "paper.md").write_text("Same content", encoding="utf-8")

    first = load_documents(first_root)
    second = load_documents(second_root)

    assert first[0].doc_id == second[0].doc_id
