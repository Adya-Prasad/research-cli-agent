import asyncio
import time
from dataclasses import dataclass

import pytest

from research_agent.orchestration.models import (
    ResearchBudget,
    WorkerAssignment,
    WorkerBudget,
    WorkerResult,
    WorkerRole,
    WorkerUsage,
)
from research_agent.orchestration.planner import DeterministicPlanner
from research_agent.orchestration.ports import ResearchWorker
from research_agent.orchestration.supervisor import ResearchSupervisor


@dataclass
class ConcurrencyProbe:
    active: int = 0
    peak: int = 0


class BlockingWorker:
    def __init__(
        self,
        probe: ConcurrencyProbe,
        two_started: asyncio.Event,
        release: asyncio.Event,
    ) -> None:
        self.probe = probe
        self.two_started = two_started
        self.release = release

    async def run(
        self,
        assignment: WorkerAssignment,
        budget: WorkerBudget,
    ) -> WorkerResult:
        started_at = time.perf_counter()
        self.probe.active += 1
        self.probe.peak = max(
            self.probe.peak,
            self.probe.active,
        )

        if self.probe.peak == 2:
            self.two_started.set()

        try:
            await self.release.wait()
        finally:
            self.probe.active -= 1

        return WorkerResult(
            task_id=assignment.task_id,
            role=assignment.role,
            status="completed",
            answer=f"Completed {assignment.role}",
            usage=WorkerUsage(
                decision_calls=1,
                tool_calls=1,
            ),
            started_at=started_at,
            finished_at=time.perf_counter(),
        )


class ImmediateWorker:
    def __init__(self) -> None:
        self.calls = 0

    async def run(
        self,
        assignment: WorkerAssignment,
        budget: WorkerBudget,
    ) -> WorkerResult:
        self.calls += 1
        now = time.perf_counter()
        return WorkerResult(
            task_id=assignment.task_id,
            role=assignment.role,
            status="completed",
            answer=f"Completed {assignment.role}",
            started_at=now,
            finished_at=now,
        )


class RaisingWorker:
    async def run(
        self,
        assignment: WorkerAssignment,
        budget: WorkerBudget,
    ) -> WorkerResult:
        raise RuntimeError("simulated unhandled worker failure")


def test_supervisor_enforces_concurrency_and_preserves_plan_order() -> None:
    async def scenario() -> None:
        plan = DeterministicPlanner().create_plan(
            "How should bounded agents execute?"
        )
        probe = ConcurrencyProbe()
        two_started = asyncio.Event()
        release = asyncio.Event()

        workers: dict[WorkerRole, ResearchWorker] = {
            assignment.role: BlockingWorker(
                probe,
                two_started,
                release,
            )
            for assignment in plan.assignments
        }
        supervisor = ResearchSupervisor(
            workers=workers,
            budget=ResearchBudget(
                max_workers=3,
                max_concurrency=2,
            ),
        )

        pending = asyncio.create_task(supervisor.run(plan))

        await asyncio.wait_for(
            two_started.wait(),
            timeout=1.0,
        )

        assert probe.peak == 2
        assert not pending.done()

        release.set()
        result = await pending

        assert result.peak_concurrency == 2
        assert [item.task_id for item in result.results] == [
            assignment.task_id
            for assignment in plan.assignments
        ]
        assert len(result.trace) == 6
        assert max(
            event.active_workers
            for event in result.trace
        ) == 2

    asyncio.run(scenario())


def test_supervisor_rejects_plan_before_starting_any_worker() -> None:
    plan = DeterministicPlanner().create_plan(
        "How should bounded agents execute?"
    )
    worker = ImmediateWorker()
    workers: dict[WorkerRole, ResearchWorker] = {
        assignment.role: worker
        for assignment in plan.assignments
    }
    supervisor = ResearchSupervisor(
        workers=workers,
        budget=ResearchBudget(
            max_workers=2,
            max_concurrency=2,
        ),
    )

    with pytest.raises(ValueError, match="worker limit"):
        asyncio.run(supervisor.run(plan))

    assert worker.calls == 0


def test_supervisor_keeps_sibling_results_when_one_worker_raises() -> None:
    plan = DeterministicPlanner().create_plan(
        "How should bounded agents execute?"
    )
    workers: dict[WorkerRole, ResearchWorker] = {
        "papers": ImmediateWorker(),
        "documentation_code": RaisingWorker(),
        "skeptic": ImmediateWorker(),
    }
    supervisor = ResearchSupervisor(
        workers=workers,
        budget=ResearchBudget(),
    )

    result = asyncio.run(supervisor.run(plan))

    assert [item.status for item in result.results] == [
        "completed",
        "failed",
        "completed",
    ]
    assert result.results[1].error_code == "RuntimeError"


def test_supervisor_rejects_missing_worker_before_execution() -> None:
    plan = DeterministicPlanner().create_plan(
        "How should bounded agents execute?"
    )
    worker = ImmediateWorker()
    supervisor = ResearchSupervisor(
        workers={"papers": worker},
        budget=ResearchBudget(),
    )

    with pytest.raises(ValueError, match="No worker registered"):
        asyncio.run(supervisor.run(plan))

    assert worker.calls == 0