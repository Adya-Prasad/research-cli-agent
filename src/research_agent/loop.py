from dataclasses import dataclass

from research_agent.domain import AgentResult, FinalDecision, Message, ToolCallDecision, TraceEvent
from research_agent.errors import StepLimitExceeded
from research_agent.ports import ModelClient
from research_agent.tools import ToolRegistry


@dataclass(frozen=True, slots=True)
class AgentLoop:
    """Bounded runtime coordinating model decisions and tool execution"""
    model: ModelClient
    registry: ToolRegistry
    max_steps: int = 4

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")

    def run(self, query: str) -> AgentResult:
        if not query.strip():
            raise ValueError("query must not be empty")

        messages = [
            Message(
                role="user",
                content = query
            )
        ]
        trace: list[TraceEvent] = []

        for step in range(1, self.max_steps + 1):
            decision = self.model.decide(
                messages=messages,
                tools=self.registry.specs(),
            )
            trace.append(
                TraceEvent(
                    step=step,
                    event="model_decision",
                    detail=decision.kind,
                )
            )
            if isinstance(decision, FinalDecision):
                messages.append(
                    Message(
                        role="assistant",
                        content=decision.answer,
                    )
                )

                return AgentResult(
                    answer=decision.answer,
                    messages=messages,
                    trace=trace,
                    steps=step
                )
            
            if isinstance(decision, ToolCallDecision):
                messages.append(
                    Message(
                        role="assistant",
                        name=decision.tool_name,
                        content=f"tool_call: {decision.arguments}"
                    )
                )
                tool_result = self.registry.invoke(
                    name=decision.tool_name,
                    arguments=decision.arguments
                )
                messages.append(
                    Message(
                        role="tool",
                        name=decision.tool_name,
                        content=tool_result
                    )
                )
                trace.append(
                    TraceEvent(
                        step=step,
                        event="tool_result",
                        detail=tool_result,
                    )
                )
        raise StepLimitExceeded(self.max_steps)
