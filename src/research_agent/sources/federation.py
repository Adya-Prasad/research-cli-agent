
"""Bounded fan-out across independent external evidence sources."""

import asyncio
from dataclasses import dataclass, field

from research_agent.orchestration.models import WorkerEvidence
from research_agent.sources.errors import (
    SourceError,
    SourceUnavailable,
)
from research_agent.sources.ports import EvidenceSource
from research_agent.sources.security import EvidenceSecurityGate


@dataclass(frozen=True, slots=True)
class FederatedSearchTool:
    """Merge source results without allowing one failure to erase siblings."""

    sources: tuple[EvidenceSource, ...]
    security_gate: EvidenceSecurityGate = field(
        default_factory=EvidenceSecurityGate
    )

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError(
                "FederatedSearchTool requires at least one source"
            )

    async def search(
        self,
        query: str,
        top_k: int,
    ) -> tuple[WorkerEvidence, ...]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        async def invoke(
            source: EvidenceSource,
        ) -> tuple[WorkerEvidence, ...] | None:
            try:
                return await source.search(
                    query,
                    limit=top_k,
                )
            except SourceError:
                # A failed source is isolated. Day 6 will add the
                # structured retry/timeout trace for this boundary.
                return None

        batches = await asyncio.gather(
            *(invoke(source) for source in self.sources)
        )
        failures = sum(
            batch is None
            for batch in batches
        )
        successful_batches = [
            batch
            for batch in batches
            if batch is not None
        ]

        if failures and not any(successful_batches):
            raise SourceUnavailable(
                "all configured evidence sources failed"
            )

        merged: list[WorkerEvidence] = []
        seen_ids: set[str] = set()
        offset = 0

        while len(merged) < top_k:
            added_at_this_offset = False

            for batch in successful_batches:
                if offset >= len(batch):
                    continue

                item = batch[offset]
                added_at_this_offset = True

                if item.chunk_id in seen_ids:
                    continue

                seen_ids.add(item.chunk_id)
                merged.append(
                    self.security_gate.screen(item)
                )

                if len(merged) == top_k:
                    break

            if not added_at_this_offset:
                break

            offset += 1

        return tuple(
            item.model_copy(update={"rank": rank})
            for rank, item in enumerate(
                merged,
                start=1,
            )
        )