"""Typed contracts for bounded multi-worker research orchestration."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

WorkerRole = Literal[
    "papers",
    "documentation_code",
    "skeptic",
]

WorkerStatus = Literal[
    "completed",
    "failed",
    "budget_exhausted",
]
EvidenceSourceType = Literal[
    "local",
    "paper_abstract",
    "official_documentation",
    "source_code",
]

class WorkerBudget(BaseModel):
    """Maximum operations that one worker may consume."""

    model_config = {"frozen": True}

    max_decisions: int = Field(default=3, ge=1, le=20)
    max_tool_calls: int = Field(default=2, ge=0, le=20)


class ResearchBudget(BaseModel):
    """Runtime authority for one complete research run."""

    model_config = {"frozen": True}

    max_workers: int = Field(default=3, ge=1, le=16)
    max_concurrency: int = Field(default=2, ge=1, le=16)
    per_worker: WorkerBudget = Field(default_factory=WorkerBudget)

    @model_validator(mode="after")
    def validate_concurrency(self) -> Self:
        if self.max_concurrency > self.max_workers:
            raise ValueError(
                "max_concurrency cannot exceed max_workers"
            )
        return self


class WorkerAssignment(BaseModel):
    """One independently executable research assignment."""

    model_config = {"frozen": True}

    task_id: str = Field(min_length=1)
    role: WorkerRole
    question: str = Field(min_length=1)
    expected_output: str = Field(min_length=1)

    @field_validator(
        "task_id",
        "question",
        "expected_output",
        mode="before",
    )
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = " ".join(value.split())
            if not normalized:
                raise ValueError("text field must not be empty")
            return normalized
        return value


class ResearchPlan(BaseModel):
    """Validated collection of independent worker assignments."""

    model_config = {"frozen": True}

    question: str = Field(min_length=1)
    assignments: tuple[WorkerAssignment, ...] = Field(min_length=1)

    @field_validator("question", mode="before")
    @classmethod
    def normalize_question(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = " ".join(value.split())
            if not normalized:
                raise ValueError("question must not be empty")
            return normalized
        return value

    @model_validator(mode="after")
    def validate_assignments(self) -> Self:
        task_ids = [
            assignment.task_id
            for assignment in self.assignments
        ]
        roles = [
            assignment.role
            for assignment in self.assignments
        ]

        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task IDs must be unique")
        if len(roles) != len(set(roles)):
            raise ValueError("worker roles must be unique")

        return self


class WorkerEvidence(BaseModel):
    """One source-preserving evidence snapshot collected by a worker."""

    model_config = {"frozen": True}

    # Retained as chunk_id for compatibility with Day 2/4 retrieval.
    # For external sources it identifies an immutable evidence snapshot.
    chunk_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_type: EvidenceSourceType = "local"
    provider: str = Field(default="local", min_length=1)
    title: str = ""
    canonical_url: str | None = None
    text: str = Field(min_length=1)
    rank: int = Field(ge=1)
    score: float
    published_at: str | None = None
    content_sha256: str = ""
    risk_flags: tuple[str, ...] = ()
    eligible_for_synthesis: bool = True


class SearchObservation(BaseModel):
    """Result returned to a worker after one search operation."""

    model_config = {"frozen": True}

    query: str = Field(min_length=1)
    evidence: tuple[WorkerEvidence, ...]


class SearchDecision(BaseModel):
    """Worker proposal to execute the search tool."""

    model_config = {"frozen": True}

    kind: Literal["search"] = "search"
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class FinishDecision(BaseModel):
    """Worker proposal to terminate with an answer."""

    model_config = {"frozen": True}

    kind: Literal["finish"] = "finish"
    answer: str = Field(min_length=1)


WorkerDecision = SearchDecision | FinishDecision


class WorkerUsage(BaseModel):
    """Operations consumed by one worker."""

    model_config = {"frozen": True}

    decision_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)


class WorkerTraceEvent(BaseModel):
    """One timestamped event inside a worker."""

    model_config = {"frozen": True}

    event: str = Field(min_length=1)
    elapsed_ms: float = Field(ge=0)
    detail: str = ""


class WorkerResult(BaseModel):
    """Terminal outcome of exactly one assignment."""

    model_config = {"frozen": True}

    task_id: str = Field(min_length=1)
    role: WorkerRole
    status: WorkerStatus
    answer: str | None = None
    evidence: tuple[WorkerEvidence, ...] = ()
    usage: WorkerUsage = Field(default_factory=WorkerUsage)
    error_code: str | None = None
    started_at: float = Field(ge=0)
    finished_at: float = Field(ge=0)
    trace: tuple[WorkerTraceEvent, ...] = ()

    @property
    def duration_ms(self) -> float:
        return max(
            0.0,
            (self.finished_at - self.started_at) * 1_000,
        )


class SupervisorTraceEvent(BaseModel):
    """Observable scheduling event emitted by the supervisor."""

    model_config = {"frozen": True}

    sequence: int = Field(ge=1)
    event: Literal["worker_started", "worker_finished"]
    task_id: str = Field(min_length=1)
    active_workers: int = Field(ge=0)
    elapsed_ms: float = Field(ge=0)


class ResearchRun(BaseModel):
    """Terminal result of one complete supervisor execution."""

    model_config = {"frozen": True}

    plan: ResearchPlan
    results: tuple[WorkerResult, ...]
    peak_concurrency: int = Field(ge=0)
    elapsed_ms: float = Field(ge=0)
    trace: tuple[SupervisorTraceEvent, ...]