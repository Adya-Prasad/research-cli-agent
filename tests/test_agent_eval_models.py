import pytest

from research_agent.evaluation.models import (
    EvalSuite,
    EvalTask,
)


def _task(task_id: str) -> EvalTask:
    return EvalTask(
        task_id=task_id,
        question="How should agent runtimes enforce limits?",
        required_source_types=("local",),
    )


def test_suite_requires_unique_task_ids() -> None:
    with pytest.raises(ValueError, match="task IDs"):
        EvalSuite(
            schema_version="1",
            dataset_version="2026-09-15",
            tasks=(
                _task("runtime"),
                _task("runtime"),
            ),
        )


def test_task_rejects_duplicate_expected_terms() -> None:
    with pytest.raises(ValueError, match="required terms"):
        EvalTask(
            task_id="runtime",
            question="How should runtimes enforce limits?",
            required_terms=("runtime", "runtime"),
        )
