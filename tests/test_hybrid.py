import pytest

from research_agent.retrieval.hybrid import HybridRetriever
from research_agent.retrieval.models import SearchHit


class MockRetriever:
    """Mock implementation fulfilling the Retriever protocol contract for testing isolated runs."""

    def __init__(self, mock_hits: list[SearchHit]) -> None:
        self.mock_hits = mock_hits

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        # Emulate internal system limit capping behavior
        return self.mock_hits[:top_k]


@pytest.fixture
def dense_mock_hits() -> list[SearchHit]:
    return [
        SearchHit(chunk_id="chunk_A", rank=1, score=0.95, retriever="dense"),
        SearchHit(chunk_id="chunk_B", rank=2, score=0.88, retriever="dense"),
        SearchHit(chunk_id="chunk_C", rank=3, score=0.72, retriever="dense"),
    ]


@pytest.fixture
def bm25_mock_hits() -> list[SearchHit]:
    return [
        SearchHit(chunk_id="chunk_C", rank=1, score=12.4, retriever="bm25"),
        SearchHit(chunk_id="chunk_A", rank=2, score=8.1, retriever="bm25"),
        SearchHit(chunk_id="chunk_D", rank=3, score=4.2, retriever="bm25"),
    ]


def test_hybrid_retriever_initialization_checks(dense_mock_hits: list[SearchHit]):
    """Ensure parameters enforce strict bounds checking validations."""
    dummy_retriever = MockRetriever(dense_mock_hits)

    with pytest.raises(ValueError, match="rrf_k must be positive"):
        HybridRetriever(dense=dummy_retriever, bm25=dummy_retriever, rrf_k=0)

    with pytest.raises(ValueError, match="candidate_multiplier must be positive"):
        HybridRetriever(
            dense=dummy_retriever,
            bm25=dummy_retriever,
            candidate_multiplier=0,
        )


def test_rrf_scoring_and_ranking_fusion(
    dense_mock_hits: list[SearchHit], bm25_mock_hits: list[SearchHit]
):
    """Test that the structural reciprocity fusion sums rank weights accurately."""
    dense_retriever = MockRetriever(dense_mock_hits)
    bm25_retriever = MockRetriever(bm25_mock_hits)

    # Use rrf_k = 60 (code default baseline constraints)
    hybrid = HybridRetriever(
        dense=dense_retriever, bm25=bm25_retriever, rrf_k=60, candidate_multiplier=2
    )

    fused_hits = hybrid.search(query="test hybrid parameters", top_k=2)

    assert len(fused_hits) == 2
    assert fused_hits[0].retriever == "fused"
    assert fused_hits[0].rank == 1

    # Let's perform validation tracing calculations manually to check implementation accuracy:
    # chunk_A ranks: Dense=1, BM25=2 -> RRF score = 1/(60+1) + 1/(60+2) = 0.01639 + 0.01612 = 0.03251
    # chunk_C ranks: Dense=3, BM25=1 -> RRF score = 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226
    # Thus, chunk_A must beat chunk_C and rank first.
    assert fused_hits[0].chunk_id == "chunk_A"
    assert fused_hits[1].chunk_id == "chunk_C"
    assert fused_hits[0].score > fused_hits[1].score


def test_hybrid_search_caps_at_requested_top_k(
    dense_mock_hits: list[SearchHit], bm25_mock_hits: list[SearchHit]
):
    """Ensure final output does not overflow the user defined top_k request limitations."""
    dense_retriever = MockRetriever(dense_mock_hits)
    bm25_retriever = MockRetriever(bm25_mock_hits)

    hybrid = HybridRetriever(dense=dense_retriever, bm25=bm25_retriever, rrf_k=60)

    fused_hits = hybrid.search(query="scaling bounds check", top_k=1)
    assert len(fused_hits) == 1
