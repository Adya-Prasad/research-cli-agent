from collections.abc import Sequence

import pytest

from research_agent.domain import (
    Decision,
    FinalDecision,
    Message,
    ToolCallDecision,
    ToolSpec,
)
from research_agent.errors import StepLimitExceeded
from research_agent.loop import AgentLoop
from research_agent.tools import ToolRegistry, WordCountTool


class ScriptedModel:
    """Return predetermined decisions for deterministic tests."""

    def __init__(self, decisions: list[Decision]) -> None:
        self._decisions = iter(decisions)

    def decide(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> Decision:
        del messages, tools
        return next(self._decisions)


def test_loop_executes_tool_then_returns_final_answer() -> None:
    model = ScriptedModel(
        decisions=[
            ToolCallDecision(
                tool_name="word_count",
                arguments={
                    "text": "reliable agents preserve state"
                },
            ),
            FinalDecision(
                answer="The text has 4 words."
            ),
        ]
    )

    agent = AgentLoop(
        model=model,
        registry=ToolRegistry([WordCountTool()]),
    )

    result = agent.run("Count the words")

    assert result.answer == "The text has 4 words."
    assert result.steps == 2

    assert result.messages[-2] == Message(
        role="tool",
        name="word_count",
        content="4",
    )

    assert [
        event.event
        for event in result.trace
    ] == [
        "model_decision",
        "tool_result",
        "model_decision",
    ]


def test_loop_stops_at_step_limit() -> None:
    repeated_call = ToolCallDecision(
        tool_name="word_count",
        arguments={"text": "one two"},
    )

    model = ScriptedModel(
        decisions=[
            repeated_call,
            repeated_call,
        ]
    )

    agent = AgentLoop(
        model=model,
        registry=ToolRegistry([WordCountTool()]),
        max_steps=2,
    )

    with pytest.raises(
        StepLimitExceeded,
        match="max_steps=2",
    ):
        agent.run("Never finish")


def test_loop_rejects_empty_query() -> None:
    model = ScriptedModel(
        decisions=[
            FinalDecision(answer="unused"),
        ]
    )

    agent = AgentLoop(
        model=model,
        registry=ToolRegistry([WordCountTool()]),
    )

    with pytest.raises(
        ValueError,
        match="query must not be empty",
    ):
        agent.run("   ")


def test_loop_rejects_invalid_step_limit() -> None:
    model = ScriptedModel(
        decisions=[
            FinalDecision(answer="unused"),
        ]
    )

    with pytest.raises(
        ValueError,
        match="max_steps must be at least 1",
    ):
        AgentLoop(
            model=model,
            registry=ToolRegistry([WordCountTool()]),
            max_steps=0,
        )