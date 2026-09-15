"""Typed contracts for task, trial, grader, and report evaluation."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from research_agent.orchestration.models import (
    EvidenceSourceType,
    WorkerEvidence,
)
from research_agent.sources.ledger import (
    Citation,
    ClaimDraft,
    ClaimLedger,
)

EvalProfile = Literal["deterministic", "gemini"]
EvalRunStatus = Literal["completed", "partial", "failed"]
GradeKind = Literal["hard", "soft"]

class EvalTask(BaseModel):
    """One versioned problem presented to the research system."""

    model_config = {"frozen": True}

    task_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    tags: tuple[str, ...] = ()
    expected_statuses: tuple[EvalRunStatus, ...] = Field(
        default=("completed",),
        min_length=1,
    )
    required_source_types: tuple[
        EvidenceSourceType,
        ...,
    ] = ()
    required_events: tuple[str, ...] = ()
    required_terms: tuple[str, ...] = ()
    minimum_term_coverage: float = Field(
        default=0.5,
        ge=0,
        le=1,
    )
    minimum_citation_integrity: float = Field(
        default=1.0,
        ge=0,
        le=1,
    )
    max_tool_calls: int = Field(default=6, ge=0)
    max_model_calls: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def validate_collections(self) -> Self:
        if len(self.required_terms) != len(
            set(self.required_terms)
        ):
            raise ValueError(
                "required terms must be unique"
            )

        if len(self.required_events) != len(
            set(self.required_events)
        ):
            raise ValueError(
                "required events must be unique"
            )

        return self


class RegressionThresholds(BaseModel):
    """Absolute quality floors for a suite."""

    model_config = {"frozen": True}

    minimum_trial_pass_rate: float = Field(
        default=1.0,
        ge=0,
        le=1,
    )
    minimum_consistent_task_rate: float = Field(
        default=1.0,
        ge=0,
        le=1,
    )
    minimum_citation_integrity: float = Field(
        default=1.0,
        ge=0,
        le=1,
    )
    maximum_p95_latency_ms: float | None = Field(
        default=None,
        gt=0,
    )
    maximum_nominal_cost_microusd: int | None = Field(
        default=None,
        ge=0,
    )
    maximum_pass_rate_drop: float = Field(
        default=0.0,
        ge=0,
        le=1,
    )


class EvalSuite(BaseModel):
    """Versioned collection of evaluation tasks."""

    model_config = {"frozen": True}

    schema_version: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    tasks: tuple[EvalTask, ...] = Field(min_length=1)
    thresholds: RegressionThresholds = Field(
        default_factory=RegressionThresholds
    )

    @model_validator(mode="after")
    def validate_task_ids(self) -> Self:
        task_ids = [task.task_id for task in self.tasks]

        if len(task_ids) != len(set(task_ids)):
            raise ValueError(
                "evaluation task IDs must be unique"
            )

        return self


class UsageMetrics(BaseModel):
    """Observable resource usage for one trial."""

    model_config = {"frozen": True}

    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=0, ge=0)
    output_tokens: int | None = Field(default=0, ge=0)
    cached_input_tokens: int | None = Field(
        default=0,
        ge=0,
    )
    nominal_cost_microusd: int | None = Field(
        default=0,
        ge=0,
    )


class TrajectoryEvent(BaseModel):
    """One observable event, excluding private model reasoning."""

    model_config = {"frozen": True}

    sequence: int = Field(ge=1)
    event: str = Field(min_length=1)
    component: str = Field(min_length=1)
    elapsed_ms: float = Field(ge=0)
    detail: str = ""


class GeneratedClaim(BaseModel):
    """Provider-neutral structured claim proposed by a model."""

    model_config = {"frozen": True}

    text: str = Field(min_length=1)
    citations: tuple[Citation, ...] = Field(
        min_length=1
    )


class GeneratedSynthesis(BaseModel):
    """Structured output expected from a synthesis model."""

    model_config = {"frozen": True}

    summary: str = Field(min_length=1)
    claims: tuple[GeneratedClaim, ...] = Field(
        min_length=1
    )


class GenerationResult(BaseModel):
    """Structured model response plus measured usage."""

    model_config = {"frozen": True}

    synthesis: GeneratedSynthesis
    usage: UsageMetrics
    model_name: str = Field(min_length=1)
    duration_ms: float = Field(ge=0)


class CitedResearchReport(BaseModel):
    """Accepted and rejected claims with ledger assessments."""

    model_config = {"frozen": True}

    markdown: str = Field(min_length=1)
    claims: tuple[ClaimDraft, ...]
    rejected_claims: tuple[ClaimDraft, ...] = ()
    ledger: ClaimLedger


class SystemOutput(BaseModel):
    """Complete observable output of one research-system trial."""

    model_config = {"frozen": True}

    system_name: str = Field(min_length=1)
    profile: EvalProfile
    model_name: str | None = None
    status: EvalRunStatus
    report: CitedResearchReport
    evidence: tuple[WorkerEvidence, ...]
    trajectory: tuple[TrajectoryEvent, ...]
    usage: UsageMetrics
    duration_ms: float = Field(ge=0)


class GradeResult(BaseModel):
    """One grader judgment."""

    model_config = {"frozen": True}

    name: str = Field(min_length=1)
    kind: GradeKind
    score: float = Field(ge=0, le=1)
    passed: bool
    explanation: str = Field(min_length=1)


class TrialArtifact(BaseModel):
    """Result of one isolated system attempt."""

    model_config = {"frozen": True}

    task_id: str = Field(min_length=1)
    trial_id: str = Field(min_length=1)
    trial_index: int = Field(ge=1)
    profile: EvalProfile
    output: SystemOutput | None = None
    grades: tuple[GradeResult, ...]
    passed: bool
    functional_digest: str = Field(min_length=1)
    error: str | None = None


class EvalSummary(BaseModel):
    """Aggregate metrics across all trials."""

    model_config = {"frozen": True}

    task_count: int = Field(ge=1)
    trial_count: int = Field(ge=1)
    passed_trials: int = Field(ge=0)
    trial_pass_rate: float = Field(ge=0, le=1)
    consistent_task_rate: float = Field(ge=0, le=1)
    pass_at_k: float = Field(ge=0, le=1)
    mean_citation_integrity: float = Field(ge=0, le=1)
    p50_latency_ms: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)
    total_model_calls: int = Field(ge=0)
    total_tool_calls: int = Field(ge=0)
    total_input_tokens: int | None = Field(default=None, ge=0)
    total_output_tokens: int | None = Field(default=None, ge=0)
    total_nominal_cost_microusd: int | None = Field(
        default=None,
        ge=0,
    )


class EvalBaseline(BaseModel):
    """Committed comparison point from an earlier evaluation."""

    model_config = {"frozen": True}

    dataset_version: str = Field(min_length=1)
    profile: EvalProfile
    summary: EvalSummary


class RegressionDecision(BaseModel):
    """Final deployment/regression gate decision."""

    model_config = {"frozen": True}

    passed: bool
    reasons: tuple[str, ...]


class EvalReport(BaseModel):
    """Machine-readable final evaluation report."""

    model_config = {"frozen": True}

    schema_version: str
    dataset_version: str
    profile: EvalProfile
    system_name: str
    trials_per_task: int = Field(ge=1)
    artifacts: tuple[TrialArtifact, ...]
    summary: EvalSummary
    regression: RegressionDecision