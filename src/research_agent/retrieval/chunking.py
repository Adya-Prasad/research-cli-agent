"""Stage 2 of the retrieval subsystem: ResearchDocument -> overlapping Chunks.
 
Chunking is a pure, in-memory transformation: no file I/O, no network, no
model calls. That's deliberate -- it means WordWindowChunker.split can be
unit-tested with plain lists of ResearchDocument built in-memory, and it
means a smarter strategy (sentence-aware, semantic, or contextual chunking
in the style of Anthropic's Contextual Retrieval, which prepends a short
LLM-generated summary to each chunk before embedding) can replace this
class later without ingestion.py or retrieval.py changing at all -- they
only ever see Chunk objects, never how those chunks were produced.
"""
from __future__ import annotations

from research_agent.retrieval.models import Chunk, ResearchDocument

class WordWindowChunker:
    """Splits documents into fixed-size, overlapping windows of words.
 
    Word windows are the simplest chunker that's still retrieval-useful:
    no tokenizer dependency, deterministic given (window_size, overlap)
    """
    def __init__(self, window_size: int = 120, overlap: int = 20) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0 <= overlap < window_size:
            raise ValueError("overlap must be between 0 (inclusive) and window_size (exclusive)")

        self.window_size = window_size
        self.overlap = overlap

    def split(self, documents: list[ResearchDocument]) -> list[Chunk]:
        """Chunk every document, in order, into overlapping word windows.

        Within a document, windows advance by `stride = window_size -
        overlap` words; the final window is truncated rather than padded,
        so short trailing content isn't lost or diluted with duplicates

        Chunk.chunk_id is keyed on (doc_id, chunk_index), so this stays globally unique without a global counter, which would make chunk_ids depend on how many
        documents preceded this one in the corpus.
        """
        chunks: list[Chunk] = []
        stride = self.window_size - self.overlap
 
        for doc in documents:
            words = doc.text.split()
            if not words:
                continue
 
            chunk_index = 0
            start = 0
            while start < len(words):
                end = min(start + self.window_size, len(words))
                window = words[start:end]
                chunks.append(
                    Chunk.from_window(
                        doc=doc,
                        chunk_index=chunk_index,
                        words=window,
                        start_word=start,
                        end_word=end,
                    )
                )
                chunk_index += 1
                if end == len(words):
                    break
                start += stride
 
        return chunks

