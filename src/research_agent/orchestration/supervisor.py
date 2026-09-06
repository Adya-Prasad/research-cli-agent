"""Bounded concurrent execution of a validated research plan."""

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from research_agent.orchestration.models import (
    ResearchBudget,
    ResearchPlan,
    ResearchRun,
    SupervisorTraceEvent,
    WorkerAssignment,
    WorkerResult,
    WorkerRole,
)
from research_agent.orchestration.ports import ResearchWorker


@dataclass(slots=True)
class ResearchSupervisor:
    """Execute independent assignments under deterministic limits."""

    workers: Mapping[WorkerRole, ResearchWorker]
    budget: ResearchBudget

    async def run(self, plan: ResearchPlan) -> ResearchRun:
        # Perform all plan-level validation before any worker starts.
        if len(plan.assignments) > self.budget.max_workers:
            raise ValueError(
                "research plan exceeds the configured worker limit"
            )

        for assignment in plan.assignments:
            if assignment.role not in self.workers:
                raise ValueError(
                    "No worker registered for role "
                    f"{assignment.role!r}"
                )

        started_at = time.perf_counter()
        semaphore = asyncio.Semaphore(
            self.budget.max_concurrency
        )
        state_lock = asyncio.Lock()

        active_workers = 0
        peak_concurrency = 0
        trace: list[SupervisorTraceEvent] = []
        results: list[WorkerResult | None] = [
            None
            for _ in plan.assignments
        ]

        async def record_event(
            event: Literal["worker_started", "worker_finished"],
            task_id: str,
        ) -> None:
            trace.append(
                SupervisorTraceEvent(
                    sequence=len(trace) + 1,
                    event=event, 
                    task_id=task_id,
                    active_workers=active_workers,
                    elapsed_ms=(
                        time.perf_counter() - started_at
                    )
                    * 1_000,
                )
            )

        async def execute(
            index: int,
            assignment: WorkerAssignment,
        ) -> None:
            nonlocal active_workers
            nonlocal peak_concurrency

            async with semaphore:
                async with state_lock:
                    active_workers += 1
                    peak_concurrency = max(
                        peak_concurrency,
                        active_workers,
                    )
                    await record_event(
                        "worker_started",
                        assignment.task_id,
                    )

                worker_started_at = time.perf_counter()

                try:
                    worker = self.workers[assignment.role]
                    result = await worker.run(
                        assignment,
                        self.budget.per_worker,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    result = WorkerResult(
                        task_id=assignment.task_id,
                        role=assignment.role,
                        status="failed",
                        error_code=exc.__class__.__name__,
                        started_at=worker_started_at,
                        finished_at=time.perf_counter(),
                    )
                finally:
                    async with state_lock:
                        active_workers -= 1
                        await record_event(
                            "worker_finished",
                            assignment.task_id,
                        )

                results[index] = result

        async with asyncio.TaskGroup() as task_group:
            for index, assignment in enumerate(
                plan.assignments
            ):
                task_group.create_task(
                    execute(index, assignment),
                    name=f"research-worker:{assignment.task_id}",
                )

        completed_results = tuple(
            result
            for result in results
            if result is not None
        )

        if len(completed_results) != len(plan.assignments):
            raise RuntimeError(
                "supervisor finished without every worker result"
            )

        return ResearchRun(
            plan=plan,
            results=completed_results,
            peak_concurrency=peak_concurrency,
            elapsed_ms=(
                time.perf_counter() - started_at
            )
            * 1_000,
            trace=tuple(trace),
        )