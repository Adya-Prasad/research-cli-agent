import time

from rich.console import Console
from typer.testing import CliRunner

from research_agent import cli as cli_module
from research_agent.cli import app
from research_agent.orchestration.models import (
    ResearchRun,
    WorkerResult,
)
from research_agent.orchestration.planner import DeterministicPlanner

runner = CliRunner()


def _completed_run() -> ResearchRun:
    plan = DeterministicPlanner().create_plan(
        "How should agent runtimes enforce budgets?"
    )
    now = time.perf_counter()

    results = tuple(
        WorkerResult(
            task_id=assignment.task_id,
            role=assignment.role,
            status="completed",
            answer=f"Completed {assignment.role}",
            started_at=now,
            finished_at=now + 0.01,
        )
        for assignment in plan.assignments
    )

    return ResearchRun(
        plan=plan,
        results=results,
        peak_concurrency=2,
        elapsed_ms=20.0,
        trace=(),
    )


def test_research_command_renders_worker_results(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "console",
        Console(
            width=180,
            color_system=None,
            force_terminal=False,
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "run_local_research",
        lambda question, corpus, device: _completed_run(),
    )

    result = runner.invoke(
        app,
        [
            "research",
            "--corpus",
            "examples/corpus",
            "--device",
            "cpu",
            "How",
            "should",
            "agents",
            "enforce",
            "budgets?",
        ],
    )

    assert result.exit_code == 0
    assert "papers" in result.output
    assert "documentation_code" in result.output
    assert "skeptic" in result.output
    assert "Peak concurrency: 2" in result.output