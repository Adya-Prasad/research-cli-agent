import math
import re
from collections import Counter
from collections.abc import Sequence

from research_agent.retrieval.models import Chunk, SearchHit

_TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.casefold())


class BM25Retriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
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

        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))

        document_count = len(self._chunks)
        self._idf = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_terms = tuple(dict.fromkeys(tokenize(query)))
        scored: list[tuple[Chunk, float]] = []

        for chunk, tokens, frequencies in zip(
            self._chunks,
            self._tokens,
            self._term_frequencies,
            strict=True,
        ):
            score = 0.0
            document_length = len(tokens)

            for term in query_terms:
                term_frequency = frequencies[term]
                if term_frequency == 0:
                    continue

                length_ratio = document_length / self._average_length
                denominator = term_frequency + self._k1 * (1 - self._b + self._b * length_ratio)
                score += self._idf[term] * term_frequency * (self._k1 + 1) / denominator

            if score > 0:
                scored.append((chunk, score))

        scored.sort(key=lambda item: (-item[1], item[0].chunk_id))

        return [
            SearchHit(
                chunk_id=chunk.chunk_id,
                rank=rank,
                score=score,
                retriever="bm25",
            )
            for rank, (chunk, score) in enumerate(scored[:top_k], start=1)
        ]
