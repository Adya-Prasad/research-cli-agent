"""Provider-neutral external evidence interface."""

from typing import Protocol

from research_agent.orchestration.models import WorkerEvidence


class EvidenceSource(Protocol):
    async def search(
        self,
        query: str,
        limit: int,
    ) -> tuple[WorkerEvidence, ...]: ...