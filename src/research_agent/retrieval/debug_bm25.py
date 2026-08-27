from __future__ import annotations
import math
import re
import uuid
import hashlib
from pathlib import Path
from typing import Literal, Sequence
from collections import Counter
from pydantic import BaseModel, Field

# Import Rich elements for high-fidelity scannable terminal prints
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

# =====================================================================
# 1. EXACT SYSTEM CONTRACTS & MODELS (From your models.py)
# =====================================================================
_ID_NAMESPACE = uuid.UUID("6f3f5b2e-6e0a-4c1e-9d3c-6a6d4f7c9a10")

def _stable_id(*parts: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, "::".join(parts)))

class ResearchDocument(BaseModel):
    model_config = {"frozen": True}
    doc_id: str
    source_path: Path
    doc_type: Literal["md", "txt"]
    text: str
    checksum: str = Field(description="sha256 of `text`")

    @classmethod
    def from_text(cls, source_path: Path, doc_type: str, text: str) -> ResearchDocument:
        checksum = hashlib.sha256(text.encode('utf-8')).hexdigest()
        doc_id = _stable_id(str(source_path.as_posix()), checksum)
        return cls(doc_id=doc_id, source_path=source_path, doc_type=doc_type, text=text, checksum=checksum)

class Chunk(BaseModel):
    model_config = {"frozen": True}
    chunk_id: str
    doc_id: str 
    source_path: Path 
    chunk_index: int = Field(ge=0)
    text: str
    start_word: int = Field(ge=0)
    end_word: int = Field(ge=0)

    @classmethod
    def from_window(cls, doc: ResearchDocument, chunk_index: int, words: list[str], start_word: int, end_word: int) -> "Chunk":
        text = " ".join(words)
        chunk_id = _stable_id(doc.doc_id, str(chunk_index))
        return cls(
            chunk_id=chunk_id, doc_id=doc.doc_id, source_path=doc.source_path,
            chunk_index=chunk_index, text=text, start_word=start_word, end_word=end_word,
        )

class SearchHit(BaseModel):
    chunk_id: str 
    rank: int = Field(ge=1)
    score: float
    retriever: Literal["dense", "bm25", "fused"]

# =====================================================================
# 2. INSTRUMENTED RETRIEVER (Your exact logic + Visual Overrides)
# =====================================================================
_TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)

def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.casefold())

class BM25Retriever:
    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("BM25Retriever requires at least one chunk")
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self._chunks = tuple(chunks)
        self._k1 = k1
        self._b = b
        self._tokens = [tokenize(chunk.text) for chunk in self._chunks]
        self._term_frequencies = [Counter(tokens) for tokens in self._tokens]
        self._average_length = sum(map(len, self._tokens)) / len(self._tokens)

        # Custom explicit container to run your requested debug print loop
        console.print(Panel("[bold cyan]🔄 STAGE 1: INITIALIZING BM25 INGESTION LOOP[/bold cyan]"))
        
        document_frequency: Counter[str] = Counter()
        
        for idx, tokens in enumerate(self._tokens):
            document_frequency.update(set(tokens))
            
            # --- REQUESTED PRINT DEBUG HOOK ---
            # Rendered as an isolated table row per iteration cycle
            t_table = Table(title=f"📥 Tokenization Iteration (Chunk Index {idx})", show_header=True, header_style="bold magenta")
            t_table.add_column("Property", style="dim", width=25)
            t_table.add_column("Value Snapshot", overflow="fold")
            
            t_table.add_row("Tokens Generated", f"{tokens[:12]}... [Total Count: {len(tokens)}]")
            t_table.add_row("Document Frequency Map State", str(dict(document_frequency.most_common(6))) + "...")
            console.print(t_table)
            console.print("\n")

        document_count = len(self._chunks)
        
        # Computing the Inverse Document Frequency (IDF) Matrix
        self._idf = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        
        # Visualize calculated Inverse Document Frequencies
        idf_table = Table(title="📊 Computed Term IDF Values (Information Content Weighting)", header_style="bold green")
        idf_table.add_column("Term", style="bold yellow")
        idf_table.add_column("Doc Frequency Count", justify="right")
        idf_table.add_column("IDF Score", justify="right")
        
        # Show top important/uncommon terms vs common terms
        for term, freq in list(document_frequency.most_common(5)) + list(document_frequency.items())[-5:]:
            idf_table.add_row(term, str(freq), f"{self._idf[term]:.4f}")
        console.print(idf_table)
        console.print(f"\n[bold green]✔[/bold green] Ingestion complete. Average Chunk Length: [bold yellow]{self._average_length:.2f}[/bold yellow] tokens.\n")

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        console.print(Panel(f"[bold amber]🔍 STAGE 2: SEARCHING QUERY: \"{query}\"[/bold amber]"))
        
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_terms = tuple(dict.fromkeys(tokenize(query)))
        scored: list[tuple[Chunk, float]] = []

        # Table to profile internal calculation steps for each document
        calc_table = Table(title="⚡ Core Scoring Term-by-Term Accumulation Trace")
        calc_table.add_column("Chunk ID", style="dim")
        calc_table.add_column("Term Found")
        calc_table.add_column("TF (Term Freq)", justify="right")
        calc_table.add_column("Len Ratio", justify="right")
        calc_table.add_column("Denominator Formula", justify="right")
        calc_table.add_column("Delta Score Add", justify="right", style="bold green")

        for chunk, tokens, frequencies in zip(self._chunks, self._tokens, self._term_frequencies, strict=True):
            score = 0.0
            document_length = len(tokens)

            for term in query_terms:
                term_frequency = frequencies[term]
                if term_frequency == 0:
                    continue

                length_ratio = document_length / self._average_length
                
                # BM25 Core scaling function formula
                denominator = term_frequency + self._k1 * (1 - self._b + self._b * length_ratio)
                term_score = self._idf[term] * term_frequency * (self._k1 + 1) / denominator
                score += term_score
                
                calc_table.add_row(
                    chunk.chunk_id[:8] + "...",
                    term,
                    str(term_frequency),
                    f"{length_ratio:.2f}",
                    f"{denominator:.2f}",
                    f"+{term_score:.4f}"
                )

            if score > 0:
                scored.append((chunk, score))

        console.print(calc_table)
        console.print("\n")

        scored.sort(key=lambda item: (-item[1], item[0].chunk_id))

        return [
            SearchHit(chunk_id=chunk.chunk_id, rank=rank, score=score, retriever="bm25")
            for rank, (chunk, score) in enumerate(scored[:top_k], start=1)
        ]

# =====================================================================
# 3. CORPUS ASSEMBLY ENGINE (Lengthy Test Data)
# =====================================================================
if __name__ == "__main__":
    console.print("[bold tracking]🛠️ Setting up long documents for tokenization checks...[/bold tracking]\n")

    # Constructing a lengthy sample document to test realistic text densities
    long_text = (
        "In modern retrieval systems, machine learning is essential. Artificial intelligence drives the embedding layers, "
        "but traditional token match metrics like BM25 remain an incredibly strong baseline for keyword density exploration. "
        "A standard machine learning engine uses vector similarities, but when users search for hyper-specific system tracking codes, "
        "vector models frequently miss exact phrase patterns. That is why hybrid search architectures combine dense retrieval matches "
        "with lexical keyword pipelines like BM25Retriever components to maximize recall parameters across production evaluations."
    )

    doc = ResearchDocument.from_text(
        source_path=Path("docs/architecture_notes.md"),
        doc_type="md",
        text=long_text
    )

    # Simulating long text spans by parsing the text into structural overlapping window sequences
    corpus_words = long_text.split()
    
    # Slice 1: Focuses heavily on ML and AI keywords
    chunk_1 = Chunk.from_window(doc=doc, chunk_index=0, words=corpus_words[:20], start_word=0, end_word=20)
    # Slice 2: Focuses heavily on Vector similarity parameters
    chunk_2 = Chunk.from_window(doc=doc, chunk_index=1, words=corpus_words[20:45], start_word=20, end_word=45)
    # Slice 3: Focuses heavily on Hybrid models and BM25 implementations
    chunk_3 = Chunk.from_window(doc=doc, chunk_index=2, words=corpus_words[45:], start_word=45, end_word=len(corpus_words))

    test_corpus = [chunk_1, chunk_2, chunk_3]

    # Initialize retriever - triggers your requested debug log trace automatically
    retriever = BM25Retriever(chunks=test_corpus, k1=1.2, b=0.75)

    # Execute lexical matching check
    hits = retriever.search(query="machine learning vector models", top_k=2)

    # Print Final Sorted Output Results Mappin
    gres_table = Table(title="🏆 FINAL SEARCH HITS RETRIEVED (Sorted Output Rank Matrix)", header_style="bold royal_blue1")
    res_table.add_column("Rank Position", justify="center")
    res_table.add_column("Chunk ID String Hash", style="dim")
    res_table.add_column("BM25 Retained Score", justify="right", style="bold cyan")
    for h in hits:res_table.add_row(f"Rank #{h.rank}", h.chunk_id, f"{h.score:.4f}")
    console.print(res_table)