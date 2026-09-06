"""Bounded specialist-worker runtime and local retrieval adapter."""

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from research_agent.orchestration.models import (
    FinishDecision,
    SearchDecision,
    SearchObservation,
    WorkerAssignment,
    WorkerBudget,
    WorkerDecision,
    WorkerEvidence,
    WorkerResult,
    WorkerStatus,
    WorkerTraceEvent,
    WorkerUsage,
)
from research_agent.orchestration.ports import (
    AsyncSearchTool,
    WorkerDecider,
)
from research_agent.retrieval.lab import RetrievalLab


class DeterministicWorkerDecider:
    """One-search Day 4 policy used to exercise the runtime."""

    async def decide(
        self,
        assignment: WorkerAssignment,
        observations: Sequence[SearchObservation],
    ) -> WorkerDecision:
        if not observations:
            return SearchDecision(
                query=assignment.question,
                top_k=3,
            )

        latest = observations[-1]
        if not latest.evidence:
            return FinishDecision(
                answer=(
                    "No local evidence was found for this assignment. "
                    "External-source investigation is still required."
                )
            )

        primary = latest.evidence[0]
        return FinishDecision(
            answer=(
                f"Top local evidence ({primary.source}): "
                f"{primary.text}"
            )
        )


@dataclass(slots=True)
class LocalRetrievalTool:
    """Async boundary around the existing synchronous retrieval lab."""

    lab: RetrievalLab
    _lock: asyncio.Lock = field(
        default_factory=asyncio.Lock,
        init=False,
        repr=False,
    )

    async def search(
        self,
        query: str,
        top_k: int,
    ) -> tuple[WorkerEvidence, ...]:
        # Sentence-transformer access is serialized until we have evidence
        # that one shared model instance is safe and useful concurrently.
        async with self._lock:
            hits = await asyncio.to_thread(
                self.lab.search,
                query,
                mode="hybrid",
                top_k=top_k,
            )

        evidence: list[WorkerEvidence] = []
        for hit in hits:
            chunk = self.lab.chunk_for(hit.chunk_id)
            evidence.append(
                WorkerEvidence(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source_path.as_posix(),
                    text=chunk.text,
                    rank=hit.rank,
                    score=hit.score,
                )
            )

        return tuple(evidence)


@dataclass(slots=True)
class SpecialistWorker:
    """Execute worker proposals under runtime-enforced budgets."""

    decider: WorkerDecider
    search_tool: AsyncSearchTool

    async def run(
        self,
        assignment: WorkerAssignment,
        budget: WorkerBudget,
    ) -> WorkerResult:
        started_at = time.perf_counter()
        decision_calls = 0
        tool_calls = 0
        observations: list[SearchObservation] = []
        evidence: list[WorkerEvidence] = []
        trace: list[WorkerTraceEvent] = []

        def add_trace(event: str, detail: str = "") -> None:
            trace.append(
                WorkerTraceEvent(
                    event=event,
                    elapsed_ms=(
                        time.perf_counter() - started_at
                    )
                    * 1_000,
                    detail=detail,
                )
            )

        def terminal_result(
            *,
            status: WorkerStatus,
            answer: str | None = None,
            error_code: str | None = None,
        ) -> WorkerResult:
            return WorkerResult(
                task_id=assignment.task_id,
                role=assignment.role,
                status=status,
                answer=answer,
                evidence=tuple(evidence),
                usage=WorkerUsage(
                    decision_calls=decision_calls,
                    tool_calls=tool_calls,
                ),
                error_code=error_code,
                started_at=started_at,
                finished_at=time.perf_counter(),
                trace=tuple(trace),
            )

        add_trace("worker_started", assignment.role)

        try:
            while True:
                # Check before charging or invoking the dependency.
                if decision_calls >= budget.max_decisions:
                    add_trace(
                        "budget_exhausted",
                        "decision budget",
                    )
                    return terminal_result(
                        status="budget_exhausted",
                        error_code="decision_budget_exhausted",
                    )

                decision_calls += 1
                add_trace(
                    "decision_started",
                    f"decision={decision_calls}",
                )
                decision = await self.decider.decide(
                    assignment,
                    tuple(observations),
                )

                if isinstance(decision, FinishDecision):
                    add_trace("worker_completed", "finish")
                    return terminal_result(
                        status="completed",
                        answer=decision.answer,
                    )

                if not isinstance(decision, SearchDecision):
                    raise TypeError(
                        "worker returned an unsupported decision"
                    )

                if tool_calls >= budget.max_tool_calls:
                    add_trace(
                        "budget_exhausted",
                        "tool budget",
                    )
                    return terminal_result(
                        status="budget_exhausted",
                        error_code="tool_budget_exhausted",
                    )

                tool_calls += 1
                add_trace(
                    "tool_started",
                    f"search={tool_calls}",
                )
                found = await self.search_tool.search(
                    decision.query,
                    decision.top_k,
                )
                observation = SearchObservation(
                    query=decision.query,
                    evidence=found,
                )
                observations.append(observation)
                evidence.extend(found)
                add_trace(
                    "tool_completed",
                    f"hits={len(found)}",
                )

        except Exception as exc: #noqa: BLE001
            add_trace(
                "worker_failed",
                exc.__class__.__name__,
            )
            return terminal_result(
                status="failed",
                error_code=exc.__class__.__name__,
            )