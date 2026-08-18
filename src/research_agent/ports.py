from collections.abc import Sequence
from typing import Any, Protocol

from research_agent.domain import Decision, Message, ToolSpec

class ModelClient(Protocol):
    """Anything capable of choosing the agent's next action"""
    def decide(
        self, 
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> Decision:
        ...

class Tool(Protocol):
    """Interface implemented by every agent tool."""

    @property
    def spec(self) -> ToolSpec:
        ...

    def invoke(self, arguments: dict[str, Any]) -> str:
        ...
