"""Bounded fan-out across independent external evidence sources."""

import asyncio
from dataclasses import dataclass, field

from research_agent.orchestration.models import WorkerEvidence, EvidenceSource
from research_agent.sources.errors import (
    SourceError,
    SourceUnavailable,
)
from research_agent.sources.identity import (
    merge_duplicate_evidence,
    with_canonical_identity,
)
from research_agent.sources.security import EvidenceSecurityGate


@dataclass(frozen=True, slots=True)
class FederatedSearchTool:
    """Merge source results without letting one failure erase siblings."""

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
                return None

        batches = await asyncio.gather(
            *(invoke(source) for source in self.sources)
        )

        failures = sum(batch is None for batch in batches)
        successful_batches = tuple(
            batch
            for batch in batches
            if batch is not None
        )

        if failures and not any(successful_batches):
            raise SourceUnavailable(
                "all configured evidence sources failed"
            )

        merged: list[WorkerEvidence] = []
        position_by_identity: dict[str, int] = {}

        maximum_batch_size = max(
            (
                len(batch)
                for batch in successful_batches
            ),
            default=0,
        )

        for offset in range(maximum_batch_size):
            for batch in successful_batches:
                if offset >= len(batch):
                    continue

                screened = self.security_gate.screen(
                    batch[offset]
                )
                prepared = with_canonical_identity(screened)
                identity = prepared.canonical_identity

                existing_position = position_by_identity.get(
                    identity
                )

                if existing_position is None:
                    position_by_identity[identity] = len(merged)
                    merged.append(prepared)
                    continue

                merged[existing_position] = (
                    merge_duplicate_evidence(
                        merged[existing_position],
                        prepared,
                    )
                )

        return tuple(
            evidence.model_copy(
                update={"rank": rank}
            )
            for rank, evidence in enumerate(
                merged[:top_k],
                start=1,
            )
        )