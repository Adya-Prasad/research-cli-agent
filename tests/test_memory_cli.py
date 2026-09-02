import pytest
from rich.console import Console
from typer.testing import CliRunner

from research_agent.cli import app
from research_agent.memory import cli as memory_cli
from research_agent.memory.in_memory import InMemoryMemoryStore
from research_agent.memory.policy import MemoryPolicy
from research_agent.memory.service import MemoryService

runner = CliRunner()
@pytest.fixture(autouse=True)
def fixed_test_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give CLI rendering tests a predictable terminal environment."""

    monkeypatch.setattr(
        memory_cli,
        "console",
        Console(
            width=160,
            color_system=None,
            force_terminal=False,
        ),
    )


def _service() -> MemoryService:
    return MemoryService(
        policy=MemoryPolicy(),
        store=InMemoryMemoryStore(),
    )


def test_memory_cli_supports_remember_recall_and_forget(
    monkeypatch,
) -> None:
    service = _service()
    monkeypatch.setattr(
        memory_cli,
        "build_memory_service",
        lambda: service,
    )

    remembered = runner.invoke(
        app,
        [
            "memory",
            "remember",
            "--user-id",
            "user-1",
            "--session-id",
            "session-a",
            "--kind",
            "research_interest",
            "I",
            "study",
            "agent",
            "memory",
            "systems.",
        ],
    )

    assert remembered.exit_code == 0
    assert "Memory accepted" in remembered.output

    hits = service.recall("user-1", "agent memory systems")
    memory_id = hits[0].memory_id

    recalled = runner.invoke(
        app,
        [
            "memory",
            "recall",
            "--user-id",
            "user-1",
            "agent",
            "memory",
            "systems",
        ],
    )

    assert recalled.exit_code == 0
    assert "I study agent memory systems." in recalled.output

    forgotten = runner.invoke(
        app,
        [
            "memory",
            "forget",
            "--user-id",
            "user-1",
            memory_id,
        ],
    )

    assert forgotten.exit_code == 0
    assert "Memory deleted" in forgotten.output
    assert service.recall("user-1", "agent memory systems") == []


def test_memory_cli_reports_policy_rejection(
    monkeypatch,
) -> None:
    service = _service()
    monkeypatch.setattr(
        memory_cli,
        "build_memory_service",
        lambda: service,
    )

    result = runner.invoke(
        app,
        [
            "memory",
            "remember",
            "--user-id",
            "user-1",
            "--session-id",
            "session-a",
            "--kind",
            "durable_fact",
            "API_KEY=sk_abcdefghijklmnopqrstuvwxyz",
        ],
    )

    assert result.exit_code == 1
    assert "sensitive_content" in result.output