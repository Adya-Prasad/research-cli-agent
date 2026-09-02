from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from research_agent.retrieval.lab import RetrievalLab
from research_agent.retrieval.ports import FloatMatrix, FloatVector


class KeywordEmbedder:
    """Deterministic embedding double; never loads a neural model."""

    @staticmethod
    def _vector(text: str) -> list[float]:
        normalized = text.casefold()
        if "meaning" in normalized or "vector" in normalized:
            return [1.0, 0.0]
        return [0.0, 1.0]

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix:
        return np.asarray(
            [self._vector(text) for text in texts],
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> FloatVector:
        return np.asarray(self._vector(text), dtype=np.float32)


def test_lab_composes_ingestion_chunking_and_all_retrievers(
    tmp_path: Path,
) -> None:
    (tmp_path / "embeddings.md").write_text(
        "Dense vectors represent semantic meaning.",
        encoding="utf-8",
    )
    (tmp_path / "runtime.md").write_text(
        "The runtime validates tool execution.",
        encoding="utf-8",
    )

    lab = RetrievalLab.from_corpus(
        corpus=tmp_path,
        embedder=KeywordEmbedder(),
    )

    dense_hits = lab.search("meaning vectors", mode="dense", top_k=1)
    bm25_hits = lab.search("runtime", mode="bm25", top_k=1)
    hybrid_hits = lab.search("runtime", mode="hybrid", top_k=1)

    assert lab.chunk_for(dense_hits[0].chunk_id).source_path == Path(
        "embeddings.md"
    )
    assert lab.chunk_for(bm25_hits[0].chunk_id).source_path == Path(
        "runtime.md"
    )
    assert lab.chunk_for(hybrid_hits[0].chunk_id).source_path == Path(
        "runtime.md"
    )


def test_lab_rejects_corpus_without_searchable_content(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="no searchable content"):
        RetrievalLab.from_corpus(
            corpus=tmp_path,
            embedder=KeywordEmbedder(),
        )