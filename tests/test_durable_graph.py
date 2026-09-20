import asyncio
import time
from dataclasses import dataclass, field
from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from research_agent.durable.graph import (
    build_durable_research_graph,
)
from research_agent.durable.models import (
    AttemptRecord,
    DurableResearchState,
    make_initial_state,
)
from research_agent.durable.policy import WorkerExecutionPolicy
from research_agent.orchestration.models import (
    WorkerAssignment,
    WorkerBudget,
    WorkerResult,
    WorkerRole,
    WorkerStatus,
)
from research_agent.orchestration.planner import (
    DeterministicPlanner,
)
from research_agent.orchestration.ports import ResearchWorker


@dataclass(slots=True)
class ScriptedWorker:
    outcomes: list[tuple[WorkerStatus, str | None, str | None]]
    delay_seconds: float = 0.0
    calls: int = field(default=0, init=False)

    async def run(
        self,
        assignment: WorkerAssignment,
        budget: WorkerBudget,
    ) -> WorkerResult:
        del budget
        self.calls += 1

        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)

        index = min(self.calls - 1, len(self.outcomes) - 1)
        status, error_code, answer = self.outcomes[index]
        now = time.perf_counter()

        return WorkerResult(
            task_id=assignment.task_id,
            role=assignment.role,
            status=status,
            answer=answer,
            error_code=error_code,
            started_at=now,
            finished_at=now + 0.001,
        )


def _completed(answer: str) -> ScriptedWorker:
    return ScriptedWorker(outcomes=[("completed", None, answer)])


def test_graph_retries_transient_failure() -> None:
    papers = ScriptedWorker(
        outcomes=[
            ("failed", "SourceUnavailable", None),
            ("completed", None, "Paper evidence"),
        ]
    )
    documentation = _completed("Documentation evidence")
    skeptic = _completed("Skeptical evidence")

    workers: dict[WorkerRole, ResearchWorker] = {
        "papers": papers,
        "documentation_code": documentation,
        "skeptic": skeptic,
    }

    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    async def scenario() -> DurableResearchState:
        graph = build_durable_research_graph(
            workers=workers,
            worker_budget=WorkerBudget(),
            execution_policy=WorkerExecutionPolicy(
                max_attempts=2,
                timeout_seconds=1.0,
                initial_backoff_seconds=0.01,
                max_backoff_seconds=0.01,
                jitter_ratio=0.0,
            ),
            checkpointer=InMemorySaver(),
            sleeper=record_sleep,
            random_fraction=lambda: 0.5,
        )
        plan = DeterministicPlanner().create_plan("How should agents recover?")
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "retry-test",
            },
            "max_concurrency": 3,
        }

        output = await graph.ainvoke(
            make_initial_state("retry-test", plan),
            config=config,
            durability="sync",
        )
        return cast(DurableResearchState, output)

    state = asyncio.run(scenario())

    assert state["status"] == "completed"
    assert state["attempts"]["papers"] == 2
    assert papers.calls == 2
    assert documentation.calls == 1
    assert skeptic.calls == 1
    assert delays == [0.01]

    first_attempt = AttemptRecord.model_validate(state["attempt_records"]["papers:1"])
    assert first_attempt.retry_scheduled
    assert first_attempt.error_code == "SourceUnavailable"


def test_graph_times_out_worker_and_synthesizes_partial_result() -> None:
    papers = _completed("Paper evidence")
    documentation = _completed("Documentation evidence")
    skeptic = ScriptedWorker(
        outcomes=[("completed", None, "Late skeptic evidence")],
        delay_seconds=0.05,
    )

    workers: dict[WorkerRole, ResearchWorker] = {
        "papers": papers,
        "documentation_code": documentation,
        "skeptic": skeptic,
    }

    async def scenario() -> DurableResearchState:
        graph = build_durable_research_graph(
            workers=workers,
            worker_budget=WorkerBudget(),
            execution_policy=WorkerExecutionPolicy(
                max_attempts=1,
                timeout_seconds=0.005,
            ),
            checkpointer=InMemorySaver(),
        )
        plan = DeterministicPlanner().create_plan("How should agents recover?")
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "timeout-test",
            },
            "max_concurrency": 3,
        }

        output = await graph.ainvoke(
            make_initial_state("timeout-test", plan),
            config=config,
            durability="sync",
        )
        return cast(DurableResearchState, output)

    state = asyncio.run(scenario())

    skeptic_result = WorkerResult.model_validate(state["results"]["skeptic"])

    assert skeptic_result.status == "timed_out"
    assert skeptic_result.error_code == "worker_timeout"
    assert state["status"] == "partial"
    assert "Research status: PARTIAL" in state["final_report"]
    assert "Missing worker evidence" in state["final_report"]


def test_report_preserves_plan_order() -> None:
    workers: dict[WorkerRole, ResearchWorker] = {
        "papers": _completed("Paper evidence"),
        "documentation_code": _completed("Documentation evidence"),
        "skeptic": _completed("Skeptical evidence"),
    }

    async def scenario() -> DurableResearchState:
        graph = build_durable_research_graph(
            workers=workers,
            worker_budget=WorkerBudget(),
            execution_policy=WorkerExecutionPolicy(),
            checkpointer=InMemorySaver(),
        )
        plan = DeterministicPlanner().create_plan("How should agents recover?")
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "order-test",
            }
        }

        output = await graph.ainvoke(
            make_initial_state("order-test", plan),
            config=config,
            durability="sync",
        )
        return cast(DurableResearchState, output)

    report = asyncio.run(scenario())["final_report"]

    assert report.index("[papers]") < report.index("[documentation_code]")
    assert report.index("[documentation_code]") < report.index("[skeptic]")
