from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field

from research_agent.domain import ToolSpec
from research_agent.errors import UnknownToolError
from research_agent.ports import Tool


class WordCountInput(BaseModel):
    """Validated input accepted by WordCountTool"""
    text: str = Field(min_length=1)

class WordCountTool:
    """Deterministically count whitespace-separated words"""
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="word_count",
            description="Count whitespace-separated words in text",
            input_schema=WordCountInput.model_json_schema(),
        )

    def invoke(self, arguments: dict[str, Any]) -> str:
        parsed = WordCountInput.model_validate(arguments)
        count = len(parsed.text.split())
        return str(count)

class ToolRegistry:
    """Allowlisted collection of tools available to the runtimes"""
    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools: dict[str, Tool] = {}

        for tool in tools:
            name = tool.spec.name

            if name in self._tools:
                raise ValueError(f"Duplicate tool name: {name}")
            
            self._tools[name] = tool

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def invoke(
        self, 
        name: str, 
        arguments: dict[str, Any]
    ) -> str:
        tool = self._tools.get(name)
        if tool is None:
            raise UnknownToolError(name)
        return tool.invoke(arguments)
        