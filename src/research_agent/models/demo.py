from collections.abc import Sequence

from research_agent.domain import (
    Decision,
    FinalDecision,
    Message,
    ToolCallDecision,
    ToolSpec,
)

class DemoModel:
    """
    Deterministic standin for an LLM.
    First Decision: Call word_count
    second decision: use the observation to produce a final answer
    """
    def decide(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> Decision:

        del tools

        tool_messages = [
            message for message in messages if message.role == "tool"
        ]
        if not tool_messages:
            user_messages  = next(
                message for message in messages if message.role == "user"
            )

            return ToolCallDecision(
                tool_name = "word_count",
                arguments = {'text': user_messages.content},
            )
        
        latest_tool_result = tool_messages[-1].content

        return FinalDecision(
            answer=f"Your input contains {latest_tool_result} words"
        )
        
