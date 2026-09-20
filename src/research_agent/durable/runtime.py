"""SQLite-backed execution and resume boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from research_agent.durable.errors import (
    DurableRunConflictError,
    DurableRunStateError,
)
from research_agent.durable.graph import (
    build_durable_research_graph,
)
from research_agent.durable.models import (
    REQUIRED_ROLES,
    AttemptRecord,
    DurableResearchRun,
    DurableResearchState,
    make_initial_state,
    normalize_run_id,
    plan_fingerprint,
)
from research_agent.durable.policy import WorkerExecutionPolicy
from research_agent.orchestration.models import (
    ResearchBudget,
    ResearchPlan,
    WorkerResult,
    WorkerRole,
)
from research_agent.orchestration.ports import ResearchWorker


@dataclass(slots=True)
class DurableResearchRuntime:
    """Create or resume one checkpointed research thread."""

    checkpoint_path: Path
    workers: Mapping[WorkerRole, ResearchWorker]
    research_budget: ResearchBudget
    execution_policy: WorkerExecutionPolicy

    async def run(
        self,
        run_id: str,
        plan: ResearchPlan,
    ) -> DurableResearchRun:
        normalized_run_id = normalize_run_id(run_id)

        if len(plan.assignments) > self.research_budget.max_workers:
            raise ValueError("research plan exceeds max_workers")

        self.checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        config: RunnableConfig = {
            "configurable": {
                "thread_id": normalized_run_id,
            },
            "max_concurrency": (self.research_budget.max_concurrency),
        }

        async with AsyncSqliteSaver.from_conn_string(str(self.checkpoint_path)) as checkpointer:
            graph = build_durable_research_graph(
                workers=self.workers,
                worker_budget=self.research_budget.per_worker,
                execution_policy=self.execution_policy,
                checkpointer=checkpointer,
            )

            existing_snapshot = await graph.aget_state(config)
            resumed = bool(existing_snapshot.values)

            if resumed:
                stored_fingerprint = existing_snapshot.values.get("plan_fingerprint")
                expected_fingerprint = plan_fingerprint(plan)

                if stored_fingerprint != expected_fingerprint:
                    raise DurableRunConflictError(
                        "run_id already belongs to a different research plan"
                    )

                # None is LangGraph's resume-after-error signal.
                resume_input = cast(
                    DurableResearchState,
                    None,
                )
                await graph.ainvoke(
                    resume_input,
                    config=config,
                    durability="sync",
                )
            else:
                await graph.ainvoke(
                    make_initial_state(
                        normalized_run_id,
                        plan,
                    ),
                    config=config,
                    durability="sync",
                )

            final_snapshot = await graph.aget_state(config)

        if final_snapshot.next:
            raise DurableRunStateError("graph returned with unfinished nodes")

        state = cast(
            DurableResearchState,
            dict(final_snapshot.values),
        )

        return self._materialize(
            state=state,
            plan=plan,
            resumed=resumed,
        )

    @staticmethod
    def _materialize(
        *,
        state: DurableResearchState,
        plan: ResearchPlan,
        resumed: bool,
    ) -> DurableResearchRun:

        terminal_status = state["status"]

        if terminal_status == "running":
            raise DurableRunStateError("cannot materialize a run that is still running")

        ordered_results: list[WorkerResult] = []
        attempts: dict[WorkerRole, int] = {}
        records: list[AttemptRecord] = []

        for role in REQUIRED_ROLES:
            raw_result = state["results"].get(role)
            if raw_result is None:
                raise DurableRunStateError(f"missing terminal result for {role}")

            ordered_results.append(WorkerResult.model_validate(raw_result))

            role_attempts = state["attempts"].get(role)
            if role_attempts is None:
                raise DurableRunStateError(f"missing attempt count for {role}")

            attempts[role] = role_attempts

            for attempt in range(1, role_attempts + 1):
                key = f"{role}:{attempt}"
                raw_record = state["attempt_records"].get(key)

                if raw_record is None:
                    raise DurableRunStateError(f"missing attempt record {key}")

                records.append(AttemptRecord.model_validate(raw_record))

        if not state["final_report"]:
            raise DurableRunStateError("completed graph has no final report")

        return DurableResearchRun(
            run_id=state["run_id"],
            plan=plan,
            results=tuple(ordered_results),
            attempts=attempts,
            attempt_records=tuple(records),
            status=terminal_status,
            final_report=state["final_report"],
            resumed=resumed,
        )
