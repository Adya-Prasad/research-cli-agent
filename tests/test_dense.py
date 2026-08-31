from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from research_agent.retrieval.dense import DenseRetriever
from research_agent.retrieval.models import Chunk
from research_agent.retrieval.ports import FloatMatrix, FloatVector


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix:
        return np.asarray(
            [self._vectors[text] for text in texts],
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> FloatVector:
        return np.asarray(self._vectors[text], dtype=np.float32)


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("::")[0],
        source_path=Path(f"{chunk_id.split('::')[0]}.md"),
        chunk_index=0,
        text=text,
        start_word=0,
        end_word=len(text.split()),
    )


def test_dense_retriever_ranks_by_cosine_similarity() -> None:
    chunks = [
        make_chunk("embeddings::0", "semantic vectors"),
        make_chunk("runtime::0", "tool execution"),
    ]
    embedder = FakeEmbedder(
        {
            "semantic vectors": [4.0, 0.0],
            "tool execution": [0.0, 8.0],
            "meaning representation": [2.0, 0.1],
        }
    )

    retriever = DenseRetriever(chunks, embedder)
    hits = retriever.search("meaning representation", top_k=2)

    assert [hit.chunk_id for hit in hits] == ["embeddings::0", "runtime::0"]
    assert hits[0].retriever == "dense"
    assert hits[0].score == pytest.approx(0.99875, abs=0.001)


def test_dense_retriever_rejects_zero_query_vector() -> None:
    chunk = make_chunk("embeddings::0", "semantic vectors")
    embedder = FakeEmbedder(
        {
            "semantic vectors": [1.0, 0.0],
            "bad query": [0.0, 0.0],
        }
    )
    retriever = DenseRetriever([chunk], embedder)

    with pytest.raises(ValueError, match="zero"):
        retriever.search("bad query")
