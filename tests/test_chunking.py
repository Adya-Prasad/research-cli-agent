from pathlib import Path

import pytest

from research_agent.retrieval.chunking import WordWindowChunker
from research_agent.retrieval.models import ResearchDocument


def _make_document(text: str, name: str = "paper.md") -> ResearchDocument:
    return ResearchDocument.from_text(source_path=Path(name), doc_type="md", text=text)

def test_chunker_creates_stable_overlapping_windows() -> None:
    document = _make_document("zero one two three four five six seven eight nine")
    chunker = WordWindowChunker(window_size=4, overlap=1)

    chunks = chunker.split([document])

    assert [chunk.text for chunk in chunks] == [
        "zero one two three",
        "three four five six",
        "six seven eight nine",
    ]
    assert [chunk.doc_id for chunk in chunks] == [document.doc_id] * 3
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]

def test_chunk_ids_are_deterministic_and_unique_within_a_document() -> None:
    document = _make_document("zero one two three four five six seven eight nine")
    chunker = WordWindowChunker(window_size=4, overlap=1)

    first_run = chunker.split([document])
    second_run = chunker.split([document])

    assert [c.chunk_id for c in first_run] == [c.chunk_id for c in second_run]
    assert len({c.chunk_id for c in first_run}) == len(first_run)


def test_chunker_rejects_overlap_equal_to_window_size() -> None:
    with pytest.raises(ValueError, match="overlap"):
        WordWindowChunker(window_size=4, overlap=4)

def test_default_chuncker_is_constructable() -> None:
    chunker = WordWindowChunker()
    assert chunker.window_size == 120
    assert chunker.overlap == 20