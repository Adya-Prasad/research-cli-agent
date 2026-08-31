"""
Hybrid Retriever using Reciprocal Rank Fusion (RRF) algorithm.In a production Search or Retrieval-Augmented Generation (RAG) system, The RRF algorithm converts a hit's rank into a fractional score using this formula: RPF_score = 1 / (K + Rank)

The Problem: You cannot simply add a Dense score (e.g., 0.89 Cosine Similarity) to a BM25 score (e.g., 14.52 word-count score). They are on completely different mathematical scales.

The Solution: This hybrid file drops the scores entirely and looks only at the Rank Order (1st place, 2nd place, 3rd place).
"""

from dataclasses import dataclass

from research_agent.retrieval.models import SearchHit
from research_agent.retrieval.ports import Retriever


@dataclass(frozen=True, slots=True)
class HybridRetriever:
    dense: Retriever
    bm25: Retriever
    rrf_k: int = 60
    candidate_multiplier: int = 4

    def __post_init__(self) -> None:
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        if self.candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be positive")

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        candidate_k = top_k * self.candidate_multiplier
        dense_hits = self.dense.search(query, top_k=candidate_k)
        bm25_hits = self.bm25.search(query, top_k=candidate_k)

        fused_scores: dict[str, float] = {}
        for hit in [*dense_hits, *bm25_hits]:
            fused_scores[hit.chunk_id] = fused_scores.get(hit.chunk_id, 0.0) + 1 / (
                self.rrf_k + hit.rank
            )

        ordered_chunk_ids = sorted(
            fused_scores,
            key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id),
        )

        return [
            SearchHit(
                chunk_id=chunk_id,
                rank=rank,
                score=fused_scores[chunk_id],
                retriever="fused",
            )
            for rank, chunk_id in enumerate(ordered_chunk_ids[:top_k], start=1)
        ]
