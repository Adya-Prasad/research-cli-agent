from pathlib import Path
import pytest
from pydantic import ValidationError

from research_agent.retrieval.bm25 import BM25Retriever, tokenize
from research_agent.retrieval.models import Chunk, ResearchDocument, SearchHit


@pytest.fixture
def sample_document() -> ResearchDocument:
    """Fixture to provide a valid, deterministic ResearchDocument value object."""
    return ResearchDocument.from_text(
        source_path=Path("tests/fixtures/ml_notes.md"),
        doc_type="md",
        text="Machine learning is powerful. Python is clean. Deep learning uses neural networks.",
    )


@pytest.fixture
def sample_chunks(sample_document: ResearchDocument) -> list[Chunk]:
    """Fixture to slice a document into valid Pydantic text chunk blocks."""
    words = sample_document.text.split()
    # Create distinct chunks with overlapping structural fields
    chunk_1 = Chunk.from_window(
        doc=sample_document,
        chunk_index=0,
        words=words[:3],  # "Machine learning is"
        start_word=0,
        end_word=3,
    )
    chunk_2 = Chunk.from_window(
        doc=sample_document,
        chunk_index=1,
        words=words[3:6],  # "powerful. Python is"
        start_word=3,
        end_word=6,
    )
    chunk_3 = Chunk.from_window(
        doc=sample_document,
        chunk_index=2,
        words=words[6:],  # "clean. Deep learning uses neural networks."
        start_word=6,
        end_word=len(words),
    )
    return [chunk_1, chunk_2, chunk_3]


def test_tokenize_lowercases_and_strips_punctuation():
    """Verify text is correctly stripped into unified lowercase tokens."""
    text = "Machine, Learning: Fun!..."
    assert tokenize(text) == ["machine", "learning", "fun"]


def test_bm25_initialization_validation():
    """Ensure invalid parameter inputs raise strict ValueError validations."""
    # Ensure empty chunks list is caught
    with pytest.raises(ValueError, match="requires at least one chunk"):
        BM25Retriever(chunks=[])

    # Instantiate valid sample to check parameter ranges
    dummy_chunk = Chunk(
        chunk_id="chk_1",
        doc_id="doc_1",
        source_path=Path("test.txt"),
        chunk_index=0,
        text="hello",
        start_word=0,
        end_word=1,
    )

    with pytest.raises(ValueError, match="k1 must be positive"):
        BM25Retriever(chunks=[dummy_chunk], k1=0.0)

    with pytest.raises(ValueError, match="b must be between 0 and 1"):
        BM25Retriever(chunks=[dummy_chunk], b=-0.1)


def test_bm25_search_returns_valid_sorted_hits(sample_chunks: list[Chunk]):
    """Verify retrieval returns valid models sorted by BM25 scoring limits."""
    retriever = BM25Retriever(chunks=sample_chunks, k1=1.5, b=0.75)

    # Search a keyword explicitly matching chunk 1
    hits = retriever.search(query="machine", top_k=2)

    assert isinstance(hits, list)
    assert len(hits) <= 2
    assert all(isinstance(hit, SearchHit) for hit in hits)

    # Validate output contract properties
    for hit in hits:
        assert hit.retriever == "bm25"
        assert hit.rank >= 1
        assert hit.score > 0.0

    # Chunk 1 contains "machine", it must rank first
    assert hits[0].chunk_id == sample_chunks[0].chunk_id


def test_bm25_search_invalid_arguments(sample_chunks: list[Chunk]):
    """Ensure runtime edge cases trigger explicit validation errors."""
    retriever = BM25Retriever(chunks=sample_chunks)

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.search(query="   ")

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        retriever.search(query="test", top_k=0)
