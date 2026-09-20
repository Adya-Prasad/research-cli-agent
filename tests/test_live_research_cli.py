import json
import time
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from research_agent import cli as cli_module
from research_agent.cli import app
from research_agent.orchestration.models import (
    ResearchRun,
    WorkerEvidence,
    WorkerResult,
)
from research_agent.orchestration.planner import (
    DeterministicPlanner,
)

runner = CliRunner()


def _completed_live_run() -> ResearchRun:
    plan = DeterministicPlanner().create_plan("How should agents recover?")
    now = time.perf_counter()

    evidence = WorkerEvidence(
        chunk_id="crossref:test-evidence",
        source="https://doi.org/10.1000/reliable-agents",
        source_type="paper_abstract",
        provider="crossref",
        title="Reliable Agent Runtime Design",
        canonical_url=("https://doi.org/10.1000/reliable-agents"),
        text="Bounded retries improve recovery.",
        rank=1,
        score=1.0,
        scope="abstract",
        canonical_identity=("doi:10.1000/reliable-agents"),
        provider_chain=("crossref",),
        content_sha256="abc123",
        eligible_for_synthesis=True,
    )

    results = tuple(
        WorkerResult(
            task_id=assignment.task_id,
            role=assignment.role,
            status="completed",
            answer=f"Completed {assignment.role}",
            evidence=(evidence,),
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


def test_live_web_search_accepts_prompt_option(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "sources.json"
    config_path.write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run_live_web_research(
        **kwargs: object,
    ) -> ResearchRun:
        captured.update(kwargs)
        return _completed_live_run()

    monkeypatch.setattr(
        cli_module,
        "run_live_web_research",
        fake_run_live_web_research,
    )
    monkeypatch.setattr(
        cli_module,
        "console",
        Console(
            width=220,
            color_system=None,
            force_terminal=False,
        ),
    )

    result = runner.invoke(
        app,
        [
            "live-web-search",
            "--prompt",
            "How should agents recover?",
            "--sources-config",
            str(config_path),
            "--request-timeout",
            "12",
            "--no-trace",
        ],
    )

    assert result.exit_code == 0
    assert captured["prompt"] == ("How should agents recover?")
    assert captured["sources_config"] == config_path
    assert captured["request_timeout"] == 12.0
    assert "crossref" in result.output
    assert "10.1000/reliable-agents" in result.output


def test_live_web_search_rejects_blank_prompt(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "sources.json"
    config_path.write_text("{}", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "live-web-search",
            "--prompt",
            "   ",
            "--sources-config",
            str(config_path),
        ],
    )

    assert result.exit_code != 0
    assert "must not be empty" in result.output


def test_live_runtime_requires_crossref_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "sources.json"
    config_path.write_text(
        json.dumps(
            {
                "official_documentation": [
                    {
                        "title": "Documentation",
                        "url": "https://docs.example.org/agents",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv(
        "CROSSREF_MAILTO",
        raising=False,
    )

    with pytest.raises(
        ValueError,
        match="CROSSREF_MAILTO",
    ):
        cli_module.run_live_web_research(
            prompt="reliable agents",
            sources_config=config_path,
            request_timeout=10.0,
        )
