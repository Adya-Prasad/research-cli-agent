"""Typed contracts shared by ingestion, chunking, retrieval, and evaluation.
 
Every stage in the retrieval subsystem reads and writes these types and
nothing else. It lets ingestion.py, chunking.py, and the future retrieval.py be developed, tested, and swapped independently. So this lives in its own
module rather than inside ingestion.py or chunking.py.

If Chunk lived in chunking.py, ingestion.py would need to import chunking.py to type
its own downstream consumers, and chunking.py would need ingestion.py
for ResearchDocument.

That's a circular import waiting to happen the moment the two files grow. One schema module, imported by everyone, owned by no one stage, sidesteps it entirely.
"""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# Fixed namespace so uuid5(namespace, key) is the same across every run.
# This  mechanism makes doc_id / chunk_id content-derived instead of run-order-derived
# Re-ingesting an unchanged corpus reproduces identical ids. This makes recall@k comparable across runs and lets LabeledQuery
# reference chunk_ids that stay valid as long as the source text does.

_ID_NAMESPACE = uuid.UUID("6f3f5b2e-6e0a-4c1e-9d3c-6a6d4f7c9a10")

def _stable_id(*parts: str) -> str:
    """Deterministic UUID5 derived from the given parts, joind by `::` """
    return str(uuid.uuid5(_ID_NAMESPACE, "::".join(parts)))

class ResearchDocument(BaseModel):
    """
    A single ingested source file, normalized to plain text.

    Frozen: once ingested, a document is a value, not a mutable record.
    That matters here specifically because doc_id is a hash of the text
    at construction time -- letting `.text` change after the fact would
    silently invalidate the id and desynchronize it from `checksum`.
    """

    model_config = {"frozen": True}

    doc_id: str
    source_path: Path
    doc_type: Literal["md", "txt"]
    text: str
    checksum: str = Field(description="sha256 of `text; detects stale ingestion") # Pydantic's way of attaching metadata to that field

    @classmethod
    def from_text(cls, source_path: Path, doc_type: str, text: str) -> ResearchDocument:
        checksum = hashlib.sha256(text.encode('utf-8')).hexdigest()
        doc_id = _stable_id(str(source_path.as_posix()), checksum)
        return cls(doc_id=doc_id, source_path=source_path, doc_type=doc_type, text=text, checksum=checksum)

class Chunk(BaseModel):
    """
    An overlapping window of a ResearchDocument: Retrieval unit
    Carries `source_path`, `start_word`, `end_word` so a SearchHit can be traced back to an exact citable span withour re-reading the document"""

    model_config = {"frozen": True}

    chunk_id: str
    doc_id: str 
    source_path: Path 
    chunk_index: int = Field(ge=0)
    text: str
    start_word: int = Field(ge=0)
    end_word: int = Field(ge=0)

    @classmethod
    def from_window(
        cls,
        doc: ResearchDocument,
        chunk_index: int,
        words: list[str],
        start_word: int,
        end_word: int 
    ) -> "Chunk":

        text = " ".join(words)
        chunk_id = _stable_id(doc.doc_id, str(chunk_index))
        return cls(
            chunk_id=chunk_id,
            doc_id = doc.doc_id,
            source_path=doc.source_path,
            chunk_index=chunk_index,
            text = text,
            start_word=start_word,
            end_word=end_word,
        )

class SearchHit(BaseModel):
    """One retriever's judgment about one chunk, for one query.
 
    Deliberately rank-first: `rank` is required, `score` is kept for
    diagnostics but is never assumed comparable across retrievers.
    
    BM25 scores and cosine similarities live on different scales, which is
    exactly the problem reciprocal-rank fusion is designed to sidestep by
    fusing on rank instead of score."""
    chunk_id: str 
    rank: int = Field(ge=1)
    score: float
    retriever: Literal["dense", "bm25", "fused"]

class LabeledQuery(BaseModel):
    """A query paired with the chunk id(s) considered relevant.
    This is the evaluation harness's unit: recall@k for one LabeledQuery
    is `|retrieved_top_k ∩ relevant_chunk_ids| / |relevant_chunk_ids|`."""
    query_id: str 
    query: str 
    relevant_chunk_ids: list[str] = Field(min_length=1)