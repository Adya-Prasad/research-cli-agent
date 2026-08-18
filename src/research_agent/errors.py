class AgentError(Exception):
    """Base class for expected agent run-time failures"""

class UnknownToolError(AgentError):
    """Raise when a model requests a tool outside the allowlist"""
    def __init__(self, tool_name: str) -> None:
        super().__init__(f"Unknown tool: {tool_name}")    

    
class StepLimitExceeded(AgentError):
    """Raised when the model does not finish within its budget."""
    def __init__(self, max_steps: int) -> None:
        super().__init__(f"Agent exceeded max_steps={max_steps}")
        

