import asyncio
from collections.abc import Sequence

from research_agent.orchestration.models import (
    FinishDecision,
    SearchDecision,
    SearchObservation,
    WorkerAssignment,
    WorkerBudget,
    WorkerEvidence,
)
from research_agent.orchestration.worker import (
    DeterministicWorkerDecider,
    SpecialistWorker,
)


def _assignment() -> WorkerAssignment:
    return WorkerAssignment(
        task_id="papers",
        role="papers",
        question="Find empirical evidence about bounded agents.",
        expected_output="Evidence with source references.",
    )


def _evidence() -> WorkerEvidence:
    return WorkerEvidence(
        chunk_id="chunk-1",
        source="paper.md",
        text="Bounded runtimes prevent unending agent execution.",
        rank=1,
        score=0.9,
    )


class RecordingSearchTool:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def search(
        self,
        query: str,
        top_k: int,
    ) -> tuple[WorkerEvidence, ...]:
        self.calls.append((query, top_k))
        return (_evidence(),)


class AlwaysSearchDecider:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(
        self,
        assignment: WorkerAssignment,
        observations: Sequence[SearchObservation],
    ) -> SearchDecision:
        self.calls += 1
        return SearchDecision(
            query=assignment.question,
            top_k=1,
        )


class FailingSearchTool:
    async def search(
        self,
        query: str,
        top_k: int,
    ) -> tuple[WorkerEvidence, ...]:
        raise RuntimeError("simulated provider failure")


def test_worker_searches_then_finishes() -> None:
    search_tool = RecordingSearchTool()
    worker = SpecialistWorker(
        decider=DeterministicWorkerDecider(),
        search_tool=search_tool,
    )

    result = asyncio.run(
        worker.run(
            _assignment(),
            WorkerBudget(
                max_decisions=3,
                max_tool_calls=2,
            ),
        )
    )

    assert result.status == "completed"
    assert result.answer is not None
    assert "Bounded runtimes" in result.answer
    assert result.evidence == (_evidence(),)
    assert result.usage.decision_calls == 2
    assert result.usage.tool_calls == 1
    assert len(search_tool.calls) == 1


def test_worker_blocks_tool_call_before_exceeding_budget() -> None:
    decider = AlwaysSearchDecider()
    search_tool = RecordingSearchTool()
    worker = SpecialistWorker(
        decider=decider,
        search_tool=search_tool,
    )

    result = asyncio.run(
        worker.run(
            _assignment(),
            WorkerBudget(
                max_decisions=4,
                max_tool_calls=1,
            ),
        )
    )

    assert result.status == "budget_exhausted"
    assert result.error_code == "tool_budget_exhausted"
    assert result.usage.decision_calls == 2
    assert result.usage.tool_calls == 1
    assert decider.calls == 2
    assert len(search_tool.calls) == 1


def test_worker_blocks_decision_before_exceeding_budget() -> None:
    decider = AlwaysSearchDecider()
    search_tool = RecordingSearchTool()
    worker = SpecialistWorker(
        decider=decider,
        search_tool=search_tool,
    )

    result = asyncio.run(
        worker.run(
            _assignment(),
            WorkerBudget(
                max_decisions=1,
                max_tool_calls=2,
            ),
        )
    )

    assert result.status == "budget_exhausted"
    assert result.error_code == "decision_budget_exhausted"
    assert result.usage.decision_calls == 1
    assert result.usage.tool_calls == 1
    assert decider.calls == 1
    assert len(search_tool.calls) == 1


def test_worker_converts_tool_exception_to_failed_result() -> None:
    worker = SpecialistWorker(
        decider=DeterministicWorkerDecider(),
        search_tool=FailingSearchTool(),
    )

    result = asyncio.run(
        worker.run(
            _assignment(),
            WorkerBudget(),
        )
    )

    assert result.status == "failed"
    assert result.error_code == "RuntimeError"
    assert result.usage.decision_calls == 1
    assert result.usage.tool_calls == 1


def test_deterministic_decider_finishes_after_empty_search() -> None:
    decider = DeterministicWorkerDecider()
    assignment = _assignment()

    decision = asyncio.run(
        decider.decide(
            assignment,
            (
                SearchObservation(
                    query=assignment.question,
                    evidence=(),
                ),
            ),
        )
    )

    assert isinstance(decision, FinishDecision)
    assert "No local evidence" in decision.answer