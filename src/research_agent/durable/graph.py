"""Checkpointable parallel research StateGraph."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Protocol

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from research_agent.durable.models import (
    REQUIRED_ROLES,
    AttemptRecord,
    DurableResearchState,
    DurableResearchUpdate,
    RetryRoute,
)
from research_agent.durable.policy import WorkerExecutionPolicy
from research_agent.durable.synthesis import (
    DeterministicPartialSynthesizer,
    classify_run_status,
)
from research_agent.orchestration.models import (
    ResearchPlan,
    WorkerAssignment,
    WorkerBudget,
    WorkerResult,
    WorkerRole,
)
from research_agent.orchestration.ports import ResearchWorker

SleepFunction = Callable[[float], Awaitable[None]]
RandomFractionFunction = Callable[[], float]


class AsyncGraphNode(Protocol):
    """Named async state-node contract required by LangGraph."""

    def __call__(
        self,
        state: DurableResearchState,
    ) -> Awaitable[DurableResearchUpdate]: ...

class DurableResearchNodes:
    """Node implementations with injected runtime dependencies."""

    def __init__(
        self,
        *,
        workers: Mapping[WorkerRole, ResearchWorker],
        worker_budget: WorkerBudget,
        execution_policy: WorkerExecutionPolicy,
        sleeper: SleepFunction,
        random_fraction: RandomFractionFunction,
    ) -> None:
        self._workers = workers
        self._worker_budget = worker_budget
        self._execution_policy = execution_policy
        self._sleeper = sleeper
        self._random_fraction = random_fraction
        self._synthesizer = DeterministicPartialSynthesizer()

    @staticmethod
    def _plan(state: DurableResearchState) -> ResearchPlan:
        return ResearchPlan.model_validate(state["plan"])

    @classmethod
    def _assignment(
        cls,
        state: DurableResearchState,
        role: WorkerRole,
    ) -> WorkerAssignment:
        for assignment in cls._plan(state).assignments:
            if assignment.role == role:
                return assignment

        raise ValueError(
            f"plan has no assignment for role {role!r}"
        )

    def attempt_node(
        self,
        role: WorkerRole,
    ) -> AsyncGraphNode:
        async def execute(
            state: DurableResearchState,
        ) -> DurableResearchUpdate:
            assignment = self._assignment(state, role)
            attempt = state["attempts"].get(role, 0) + 1
            started_at = time.perf_counter()

            try:
                async with asyncio.timeout(
                    self._execution_policy.timeout_seconds
                ):
                    result = await self._workers[role].run(
                        assignment,
                        self._worker_budget,
                    )
            except TimeoutError:
                finished_at = time.perf_counter()
                result = WorkerResult(
                    task_id=assignment.task_id,
                    role=assignment.role,
                    status="timed_out",
                    error_code="worker_timeout",
                    started_at=started_at,
                    finished_at=finished_at,
                )

            if (
                result.task_id != assignment.task_id
                or result.role != assignment.role
            ):
                raise ValueError(
                    "worker result identity does not match assignment"
                )

            retry = self._execution_policy.should_retry(
                result,
                attempts_used=attempt,
            )

            backoff = 0.0
            if retry:
                backoff = (
                    self._execution_policy.backoff_seconds(
                        attempt,
                        random_fraction=self._random_fraction(),
                    )
                )

            record = AttemptRecord(
                role=role,
                attempt=attempt,
                status=result.status,
                error_code=result.error_code,
                duration_ms=(
                    time.perf_counter() - started_at
                )
                * 1_000,
                retry_scheduled=retry,
                backoff_seconds=backoff,
            )

            update: DurableResearchUpdate = {
                "attempts": {role: attempt},
                "routes": {
                    role: "retry" if retry else "done"
                },
                "attempt_records": {
                    f"{role}:{attempt}": record.model_dump(
                        mode="json"
                    )
                },
            }

            if not retry:
                update["results"] = {
                    role: result.model_dump(mode="json")
                }

            return update

        return execute

    def route_node(
        self,
        role: WorkerRole,
    ) -> Callable[[DurableResearchState], RetryRoute]:
        def route(
            state: DurableResearchState,
        ) -> RetryRoute:
            try:
                return state["routes"][role]
            except KeyError as exc:
                raise RuntimeError(
                    f"missing retry route for {role}"
                ) from exc

        return route

    def backoff_node(
        self,
        role: WorkerRole,
    ) -> AsyncGraphNode:
        async def backoff(
            state: DurableResearchState,
        ) -> DurableResearchUpdate:
            attempt = state["attempts"].get(role)

            if attempt is None:
                raise RuntimeError(
                    f"missing attempt count for {role}"
                )

            key = f"{role}:{attempt}"
            raw_record = state["attempt_records"].get(key)

            if raw_record is None:
                raise RuntimeError(
                    f"missing attempt record {key}"
                )

            record = AttemptRecord.model_validate(raw_record)
            await self._sleeper(record.backoff_seconds)
            return {}

        return backoff

    @staticmethod
    def done_node(
        state: DurableResearchState,
    ) -> DurableResearchUpdate:
        del state
        return {}

    def synthesize_node(
        self,
        state: DurableResearchState,
    ) -> DurableResearchUpdate:
        plan = self._plan(state)
        ordered_results: list[WorkerResult] = []

        for assignment in plan.assignments:
            raw_result = state["results"].get(
                assignment.role
            )
            if raw_result is None:
                raise RuntimeError(
                    "synthesis started without terminal result "
                    f"for {assignment.role}"
                )

            ordered_results.append(
                WorkerResult.model_validate(raw_result)
            )

        results = tuple(ordered_results)

        return {
            "status": classify_run_status(results),
            "final_report": self._synthesizer.synthesize(
                plan,
                results,
            ),
        }


def build_durable_research_graph(
    *,
    workers: Mapping[WorkerRole, ResearchWorker],
    worker_budget: WorkerBudget,
    execution_policy: WorkerExecutionPolicy,
    checkpointer: BaseCheckpointSaver[str],
    sleeper: SleepFunction = asyncio.sleep,
    random_fraction: RandomFractionFunction = random.random,
):
    """Compile a three-branch checkpointable research graph."""

    missing_roles = set(REQUIRED_ROLES).difference(workers)

    if missing_roles:
        missing = ", ".join(sorted(missing_roles))
        raise ValueError(
            f"missing durable workers: {missing}"
        )

    nodes = DurableResearchNodes(
        workers=workers,
        worker_budget=worker_budget,
        execution_policy=execution_policy,
        sleeper=sleeper,
        random_fraction=random_fraction,
    )
    builder = StateGraph(DurableResearchState)
    done_nodes: list[str] = []

    for role in REQUIRED_ROLES:
        attempt_name = f"{role}_attempt"
        backoff_name = f"{role}_backoff"
        done_name = f"{role}_done"

        builder.add_node(
            attempt_name,
            nodes.attempt_node(role),
        )
        builder.add_node(
            backoff_name,
            nodes.backoff_node(role),
        )
        builder.add_node(
            done_name,
            nodes.done_node,
        )

        builder.add_edge(START, attempt_name)
        builder.add_conditional_edges(
            attempt_name,
            nodes.route_node(role),
            {
                "retry": backoff_name,
                "done": done_name,
            },
        )
        builder.add_edge(backoff_name, attempt_name)
        done_nodes.append(done_name)

    builder.add_node(
        "synthesize",
        nodes.synthesize_node,
    )
    builder.add_edge(done_nodes, "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile(checkpointer=checkpointer)