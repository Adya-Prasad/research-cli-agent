"""Provider boundary for durable memory."""

from typing import Protocol

from research_agent.memory.models import (
    ApprovedMemory,
    MemoryHit,
    MemoryWriteReceipt,
)


class MemoryStore(Protocol):
    def add(self, memory: ApprovedMemory) -> MemoryWriteReceipt: ...

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[MemoryHit]: ...

    def delete(self, user_id: str, memory_id: str) -> bool: ...