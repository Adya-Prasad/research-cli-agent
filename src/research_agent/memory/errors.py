"""Errors exposed by the memory subsystem."""

from typing import Literal

type MemoryRejectionReason = Literal[
    "invalid_identity",
    "empty_content",
    "content_too_long",
    "unsupported_kind",
    "sensitive_content",
]


class MemoryPolicyRejected(ValueError):
    """A candidate violated the deterministic memory-write policy."""

    def __init__(self, reason: MemoryRejectionReason) -> None:
        self.reason = reason
        super().__init__(f"Memory rejected: {reason}")


class MemoryProviderError(RuntimeError):
    """An external memory provider failed behind our application port."""