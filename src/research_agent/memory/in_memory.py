"""Deterministic memory-store implementation for tests and local demos."""

import re
from dataclasses import dataclass, field

from research_agent.memory.models import (
    ApprovedMemory,
    MemoryHit,
    MemoryWriteReceipt,
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.casefold()))


@dataclass(slots=True)
class InMemoryMemoryStore:
    _records: dict[tuple[str, str], ApprovedMemory] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def add(self, memory: ApprovedMemory) -> MemoryWriteReceipt:
        key = (memory.user_id, memory.memory_id)
        self._records[key] = memory

        return MemoryWriteReceipt(
            memory_id=memory.memory_id,
            provider_id=memory.memory_id,
            status="ready",
        )

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[MemoryHit]:
        normalized_user = user_id.strip()
        query_terms = _tokens(query)

        if not normalized_user:
            raise ValueError("user_id must not be empty")
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        scored: list[tuple[float, ApprovedMemory]] = []

        for (owner_id, _), memory in self._records.items():
            if owner_id != normalized_user:
                continue

            overlap = query_terms & _tokens(memory.content)
            if not overlap:
                continue

            score = len(overlap) / len(query_terms)
            scored.append((score, memory))

        scored.sort(key=lambda item: (-item[0], item[1].memory_id))

        return [
            MemoryHit(
                memory_id=memory.memory_id,
                user_id=memory.user_id,
                session_id=memory.session_id,
                kind=memory.kind,
                content=memory.content,
                score=score,
                rank=rank,
            )
            for rank, (score, memory) in enumerate(
                scored[:top_k],
                start=1,
            )
        ]

    def delete(self, user_id: str, memory_id: str) -> bool:
        key = (user_id.strip(), memory_id)
        return self._records.pop(key, None) is not None