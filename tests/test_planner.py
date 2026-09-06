import pytest

from research_agent.orchestration.planner import DeterministicPlanner


def test_planner_creates_three_specialized_assignments() -> None:
    plan = DeterministicPlanner().create_plan(
        "When does hybrid retrieval help an AI research agent?"
    )

    assert plan.question == (
        "When does hybrid retrieval help an AI research agent?"
    )
    assert [assignment.role for assignment in plan.assignments] == [
        "papers",
        "documentation_code",
        "skeptic",
    ]
    assert [assignment.task_id for assignment in plan.assignments] == [
        "papers",
        "documentation-code",
        "skeptic",
    ]

    assert "empirical" in plan.assignments[0].question
    assert "implementation" in plan.assignments[1].question
    assert "counterexamples" in plan.assignments[2].question


def test_planner_is_deterministic() -> None:
    planner = DeterministicPlanner()
    question = "How should an agent runtime enforce budgets?"

    assert planner.create_plan(question) == planner.create_plan(question)


def test_planner_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="question must not be empty"):
        DeterministicPlanner().create_plan("   ")