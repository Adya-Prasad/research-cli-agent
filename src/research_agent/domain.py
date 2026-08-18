from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, Field

class Message(BaseModel):
    """One item in the agent's current working state"""
    role: Literal["user", "assistant", "tool"]
    content: str
    name: str | None = None # identifies a particular participant or tool when that identity matters.
 
class ToolSpec(BaseModel):
    """The description and input schema shown to a model"""
    name: str
    description: str
    input_schema: dict[str, Any]

class ToolCallDecision(BaseModel):
    """A model proposal to invoke one registered tool."""
    kind: Literal["tool_call"] = "tool_call"
    tool_name: str
    arguments: dict[str, Any]

class FinalDecision(BaseModel):
    """A Model proposal to finish the workkflow"""
    kind: Literal["final"] = "final"
    answer: str

Decision: TypeAlias = Annotated[
    ToolCallDecision | FinalDecision,
    Field(discriminator="kind"),
]

class TraceEvent(BaseModel):
    """One obseravle event produced by the runtime"""
    step: int
    event: Literal["model_decision", "tool_result"]
    detail: str

class AgentResult(BaseModel):
    """Complete result returned by one agent execution"""
    answer: str
    messages: list[Message]
    trace: list[TraceEvent]
    steps: int
    