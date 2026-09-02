"""
corpus
  → load_documents
  → WordWindowChunker
  → DenseRetriever
  → BM25Retriever
  → HybridRetriever
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from research_agent.retrieval.bm25 import BM25Retriever
from research_agent.retrieval.chunking import WordWindowChunker
from research_agent.retrieval.dense import DenseRetriever
from research_agent.retrieval.hybrid import HybridRetriever
from research_agent.retrieval.ingestion import load_documents
from research_agent.retrieval.models import Chunk, SearchHit
from research_agent.retrieval.ports import Embedder, Retriever

type RetrievalMode = Literal["dense", "bm25", "hybrid"]

@dataclass(slots=True)
class RetrievalLab:
    """Owns one in-memory corpus and its three retrieval strategies."""

    chunks: tuple[Chunk, ...]
    dense: DenseRetriever
    bm25: BM25Retriever
    hybrid: HybridRetriever
    _chunks_by_id: dict[str, Chunk] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        chunks_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        if len(chunks_by_id) != len(self.chunks):
            raise ValueError("chunk IDs must be unique")
        self._chunks_by_id = chunks_by_id

    @classmethod
    def from_corpus(
        cls,
        corpus: Path,
        embedder: Embedder,
        chunker: WordWindowChunker | None = None,
    ) -> "RetrievalLab":
        documents = load_documents(corpus)
        chunks = tuple(
            (chunker or WordWindowChunker()).split(documents)
        )
        if not chunks:
            raise ValueError("Corpus produced no searchable content")

        dense = DenseRetriever(chunks, embedder)
        bm25 = BM25Retriever(chunks)
        hybrid = HybridRetriever(dense=dense, bm25=bm25)

        return cls(
            chunks=chunks,
            dense=dense,
            bm25=bm25,
            hybrid=hybrid,
        )

    def retriever(self, mode: RetrievalMode) -> Retriever:
        if mode == "dense":
            return self.dense
        if mode == "bm25":
            return self.bm25
        if mode == "hybrid":
            return self.hybrid
        raise ValueError(f"Unknown retrieval mode: {mode}")

    def search(
        self,
        query: str,
        mode: RetrievalMode = "hybrid",
        top_k: int = 5,
    ) -> list[SearchHit]:
        return self.retriever(mode).search(query, top_k=top_k)

    def chunk_for(self, chunk_id: str) -> Chunk:
        try:
            return self._chunks_by_id[chunk_id]
        except KeyError as exc:
            raise KeyError(f"Unknown chunk ID: {chunk_id}") from exc