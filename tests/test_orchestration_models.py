import pytest

from research_agent.orchestration.models import (
    ResearchBudget,
    ResearchPlan,
    WorkerAssignment,
    WorkerResult,
    WorkerRole,
)


def _assignment(
    task_id: str,
    role: WorkerRole = "papers",
) -> WorkerAssignment:
    return WorkerAssignment(
        task_id=task_id,
        role=role,
        question="Find evidence about bounded agent systems.",
        expected_output="Relevant evidence with source references.",
    )


def test_plan_requires_unique_task_ids() -> None:
    with pytest.raises(ValueError, match="task IDs must be unique"):
        ResearchPlan(
            question="How should agent systems be bounded?",
            assignments=(
                _assignment("duplicate"),
                _assignment("duplicate", "skeptic"),
            ),
        )


def test_plan_requires_unique_worker_roles() -> None:
    with pytest.raises(ValueError, match="worker roles must be unique"):
        ResearchPlan(
            question="How should agent systems be bounded?",
            assignments=(
                _assignment("papers-1"),
                _assignment("papers-2"),
            ),
        )


def test_budget_rejects_concurrency_above_worker_limit() -> None:
    with pytest.raises(
        ValueError,
        match="max_concurrency cannot exceed max_workers",
    ):
        ResearchBudget(
            max_workers=2,
            max_concurrency=3,
        )


def test_assignment_strips_boundary_whitespace() -> None:
    assignment = WorkerAssignment(
        task_id="  papers  ",
        role="papers",
        question="  Find research evidence.  ",
        expected_output="  Evidence with references.  ",
    )

    assert assignment.task_id == "papers"
    assert assignment.question == "Find research evidence."
    assert assignment.expected_output == "Evidence with references."


def test_worker_result_accepts_timed_out_status() -> None:
    result = WorkerResult(
        task_id="papers",
        role="papers",
        status="timed_out",
        error_code="worker_timeout",
        started_at=1.0,
        finished_at=2.0,
    )

    assert result.status == "timed_out"
    assert result.error_code == "worker_timeout"
