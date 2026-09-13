"""Durable execution errors exposed to callers."""

class DurableRunError(Exception):
    """Base class for durable research failures."""

class DurableRunConflictError(DurableRunError):
    """A run ID was reused with a different plan."""

class DurableRunStateError(DurableRunError):
    """Persisted graph state violates runtime invariants."""