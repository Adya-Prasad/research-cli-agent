import pytest
from pydantic import ValidationError

from research_agent.errors import UnknownToolError
from research_agent.tools import ToolRegistry, WordCountTool


def test_word_count_validates_and_counts() -> None:
    registry = ToolRegistry([WordCountTool()])

    result = registry.invoke(
        "word_count",
        {"text": "agents need explicit state"},
    )

    assert result == "4"


def test_word_count_rejects_missing_text() -> None:
    registry = ToolRegistry([WordCountTool()])

    with pytest.raises(ValidationError):
        registry.invoke("word_count", {})


def test_registry_rejects_unknown_tool() -> None:
    registry = ToolRegistry([WordCountTool()])

    with pytest.raises(
        UnknownToolError,
        match="search_web",
    ):
        registry.invoke(
            "search_web",
            {"query": "agent memory"},
        )


def test_registry_rejects_duplicate_names() -> None:
    with pytest.raises(
        ValueError,
        match="Duplicate tool name",
    ):
        ToolRegistry(
            [
                WordCountTool(),
                WordCountTool(),
            ]
        )