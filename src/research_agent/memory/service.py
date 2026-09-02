"""Application service coordinating policy and storage."""

from dataclasses import dataclass

from research_agent.memory.models import (
    MemoryCandidate,
    MemoryHit,
    MemoryWriteReceipt,
)
from research_agent.memory.policy import MemoryPolicy
from research_agent.memory.ports import MemoryStore


@dataclass(frozen=True, slots=True)
class MemoryService:
    policy: MemoryPolicy
    store: MemoryStore

    def remember(self, candidate: MemoryCandidate) -> MemoryWriteReceipt:
        approved = self.policy.approve(candidate)
        return self.store.add(approved)

    def recall(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[MemoryHit]:
        return self.store.search(user_id, query, top_k)

    def forget(self, user_id: str, memory_id: str) -> bool:
        return self.store.delete(user_id, memory_id)