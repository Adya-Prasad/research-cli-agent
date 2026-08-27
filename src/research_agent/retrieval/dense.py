"""
a local Dense Vector Retriever using numpy
"""

from collections.abc import Sequence
from signal import raise_signal
import numpy as np

from research_agent.retrieval.models import Chunk, SearchHit
from research_agent.retrieval.ports import Embedder, FloatVector, FloatMatrix

def _normalize_matrix(matrix: FloatMatrix) -> FloatMatrix:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("document embedding must not be a zero vector")
    return np.asarray(matrix / norms, dtype=np.float32)

def _normalize_vector(vector: FloatVector) -> FloatVector:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("query embedding must not be a zero vector")
    return np.asarray(vector / norm, dtype=np.float32)


class DenseRetriever:
    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder) -> None:
        if not chunks:
            raise ValueError("DenseRetriever requires at least one chunk")

        self._chunks = tuple(chunks)
        self._embedder = embedder

        matrix = np.asarray(
            embedder.embed_documents([chunk.text for chunk in self._chunks]),
            dtype=np.float32,
        )
        if matrix.ndim != 2 or matrix.shape[0] != len(self._chunks):
            raise ValueError("embedding matrix shape does not match chunk count")

        self._matrix = _normalize_matrix(matrix)

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_vector = np.asarray(
            self._embedder.embed_query(query),
            dtype=np.float32,
        )
        if query_vector.ndim != 1:
            raise ValueError("query embedding must be one-dimensional")
        if query_vector.shape[0] != self._matrix.shape[1]:
            raise ValueError("query and document embedding dimensions differ")

        scores = self._matrix @ _normalize_vector(query_vector)

        ordered_indices = sorted(
            range(len(self._chunks)),
            key=lambda index: (-float(scores[index]), self._chunks[index].chunk_id),
        )

        return [
            SearchHit(
                chunk_id=self._chunks[index].chunk_id,
                rank=rank,
                score=float(scores[index]),
                retriever="dense",
            )
            for rank, index in enumerate(ordered_indices[:top_k], start=1)
        ]