"""Provider-neutral domain models for durable agent memory."""

from typing import Literal

from pydantic import BaseModel, Field

type MemoryKind = Literal[
    "user_preference",
    "durable_fact",
    "research_interest",
    "session_summary",
]

class MemoryCandidate(BaseModel):
    """Untrusted request to create durable memory."""

    model_config = {"frozen": True}

    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    content: str


class ApprovedMemory(BaseModel):
    """Memory that has passed the deterministic write policy."""

    model_config = {"frozen": True}

    memory_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    kind: MemoryKind
    content: str = Field(min_length=1)


class MemoryWriteReceipt(BaseModel):
    """Acknowledgement returned by a memory provider."""

    model_config = {"frozen": True}

    memory_id: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    status: str = Field(min_length=1)


class MemoryHit(BaseModel):
    """One memory returned for a user-scoped semantic query."""

    model_config = {"frozen": True}

    memory_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    kind: MemoryKind
    content: str = Field(min_length=1)
    score: float = Field(ge=0.0)
    rank: int = Field(ge=1)