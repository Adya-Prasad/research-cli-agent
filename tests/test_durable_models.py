import pytest

from research_agent.durable.models import (
    make_initial_state,
    merge_mappings,
    normalize_run_id,
    plan_fingerprint,
)
from research_agent.orchestration.models import (
    ResearchPlan,
    WorkerAssignment,
)
from research_agent.orchestration.planner import (
    DeterministicPlanner,
)


def _plan(question: str = "How should agents recover?") -> ResearchPlan:
    return DeterministicPlanner().create_plan(question)


def test_plan_fingerprint_is_deterministic() -> None:
    first = plan_fingerprint(_plan())
    second = plan_fingerprint(_plan())

    assert first == second
    assert len(first) == 64


def test_plan_fingerprint_changes_with_plan() -> None:
    assert plan_fingerprint(_plan("Question A")) != plan_fingerprint(
        _plan("Question B")
    )


def test_initial_state_contains_serializable_values() -> None:
    plan = _plan()

    state = make_initial_state("run-001", plan)

    assert state["run_id"] == "run-001"
    assert state["plan_fingerprint"] == plan_fingerprint(plan)
    assert state["results"] == {}
    assert state["attempts"] == {}
    assert state["routes"] == {}
    assert state["attempt_records"] == {}
    assert state["status"] == "running"
    assert state["final_report"] == ""


def test_initial_state_requires_every_specialist_role() -> None:
    incomplete_plan = ResearchPlan(
        question="How should agents recover?",
        assignments=(
            WorkerAssignment(
                task_id="papers",
                role="papers",
                question="Find papers.",
                expected_output="Paper evidence.",
            ),
        ),
    )

    with pytest.raises(ValueError, match="exactly the required roles"):
        make_initial_state("run-001", incomplete_plan)


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        "contains spaces",
        "../escape",
        "-starts-with-hyphen",
        "x" * 65,
    ],
)
def test_run_id_rejects_unsafe_values(run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        normalize_run_id(run_id)


def test_mapping_reducer_merges_parallel_updates() -> None:
    merged = merge_mappings(
        {"papers": 1},
        {"skeptic": 2},
    )

    assert merged == {
        "papers": 1,
        "skeptic": 2,
    }