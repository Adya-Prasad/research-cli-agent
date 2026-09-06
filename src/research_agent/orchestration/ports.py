"""Provider-neutral boundaries used by research workers."""

from collections.abc import Sequence
from typing import Protocol

from research_agent.orchestration.models import (
    SearchObservation,
    WorkerAssignment,
    WorkerBudget,
    WorkerDecision,
    WorkerEvidence,
    WorkerResult,
)


class WorkerDecider(Protocol):
    async def decide(
        self,
        assignment: WorkerAssignment,
        observations: Sequence[SearchObservation],
    ) -> WorkerDecision: ...


class AsyncSearchTool(Protocol):
    async def search(
        self,
        query: str,
        top_k: int,
    ) -> tuple[WorkerEvidence, ...]: ...


class ResearchWorker(Protocol):
    async def run(
        self,
        assignment: WorkerAssignment,
        budget: WorkerBudget,
    ) -> WorkerResult: ...