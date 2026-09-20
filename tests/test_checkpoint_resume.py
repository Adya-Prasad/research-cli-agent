import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from research_agent.durable.errors import (
    DurableRunConflictError,
    DurableRunStateError,
)
from research_agent.durable.models import make_initial_state
from research_agent.durable.policy import WorkerExecutionPolicy
from research_agent.durable.runtime import DurableResearchRuntime
from research_agent.orchestration.models import (
    ResearchBudget,
    WorkerAssignment,
    WorkerBudget,
    WorkerResult,
    WorkerRole,
)
from research_agent.orchestration.planner import (
    DeterministicPlanner,
)
from research_agent.orchestration.ports import ResearchWorker


@dataclass(slots=True)
class ImmediateWorker:
    answer: str
    calls: int = field(default=0, init=False)

    async def run(
        self,
        assignment: WorkerAssignment,
        budget: WorkerBudget,
    ) -> WorkerResult:
        del budget
        self.calls += 1
        now = time.perf_counter()

        return WorkerResult(
            task_id=assignment.task_id,
            role=assignment.role,
            status="completed",
            answer=self.answer,
            started_at=now,
            finished_at=now + 0.001,
        )


@dataclass(slots=True)
class CrashOnceWorker:
    calls: int = field(default=0, init=False)

    async def run(
        self,
        assignment: WorkerAssignment,
        budget: WorkerBudget,
    ) -> WorkerResult:
        del budget
        self.calls += 1

        if self.calls == 1:
            await asyncio.sleep(0.05)
            raise RuntimeError("simulated process interruption")

        now = time.perf_counter()
        return WorkerResult(
            task_id=assignment.task_id,
            role=assignment.role,
            status="completed",
            answer="Recovered skeptic result",
            started_at=now,
            finished_at=now + 0.001,
        )


def _runtime(
    checkpoint_path: Path,
    workers: dict[WorkerRole, ResearchWorker],
) -> DurableResearchRuntime:
    return DurableResearchRuntime(
        checkpoint_path=checkpoint_path,
        workers=workers,
        research_budget=ResearchBudget(
            max_workers=3,
            max_concurrency=3,
        ),
        execution_policy=WorkerExecutionPolicy(
            max_attempts=1,
            timeout_seconds=1.0,
        ),
    )


def test_resume_does_not_repeat_successful_siblings(
    tmp_path: Path,
) -> None:
    papers = ImmediateWorker("Paper result")
    documentation = ImmediateWorker("Documentation result")
    skeptic = CrashOnceWorker()

    workers: dict[WorkerRole, ResearchWorker] = {
        "papers": papers,
        "documentation_code": documentation,
        "skeptic": skeptic,
    }
    runtime = _runtime(
        tmp_path / "checkpoints.sqlite",
        workers,
    )
    plan = DeterministicPlanner().create_plan("How should agents resume after failure?")

    async def scenario() -> None:
        with pytest.raises(
            RuntimeError,
            match="simulated process interruption",
        ):
            await runtime.run("resume-test", plan)

        result = await runtime.run("resume-test", plan)

        assert result.resumed
        assert result.status == "completed"

    asyncio.run(scenario())

    assert papers.calls == 1
    assert documentation.calls == 1
    assert skeptic.calls == 2


def test_run_id_rejects_different_plan(
    tmp_path: Path,
) -> None:
    workers: dict[WorkerRole, ResearchWorker] = {
        "papers": ImmediateWorker("Paper result"),
        "documentation_code": ImmediateWorker("Documentation result"),
        "skeptic": ImmediateWorker("Skeptic result"),
    }
    runtime = _runtime(
        tmp_path / "checkpoints.sqlite",
        workers,
    )
    first_plan = DeterministicPlanner().create_plan("Question one")
    conflicting_plan = DeterministicPlanner().create_plan("Question two")

    async def scenario() -> None:
        await runtime.run("conflict-test", first_plan)

        with pytest.raises(
            DurableRunConflictError,
            match="different research plan",
        ):
            await runtime.run(
                "conflict-test",
                conflicting_plan,
            )

    asyncio.run(scenario())


def test_completed_run_is_not_executed_again(
    tmp_path: Path,
) -> None:
    papers = ImmediateWorker("Paper result")
    documentation = ImmediateWorker("Documentation result")
    skeptic = ImmediateWorker("Skeptic result")

    workers: dict[WorkerRole, ResearchWorker] = {
        "papers": papers,
        "documentation_code": documentation,
        "skeptic": skeptic,
    }
    runtime = _runtime(
        tmp_path / "checkpoints.sqlite",
        workers,
    )
    plan = DeterministicPlanner().create_plan("How should completed runs behave?")

    async def scenario() -> None:
        first = await runtime.run("completed-test", plan)
        second = await runtime.run("completed-test", plan)

        assert not first.resumed
        assert second.resumed
        assert second.final_report == first.final_report

    asyncio.run(scenario())

    assert papers.calls == 1
    assert documentation.calls == 1
    assert skeptic.calls == 1


def test_runtime_refuses_to_materialize_running_state() -> None:
    plan = DeterministicPlanner().create_plan("How should agents recover?")
    state = make_initial_state(
        "still-running",
        plan,
    )

    with pytest.raises(
        DurableRunStateError,
        match="still running",
    ):
        DurableResearchRuntime._materialize(
            state=state,
            plan=plan,
            resumed=False,
        )
