"""Serializable contracts for durable research execution."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from research_agent.orchestration.models import (
    ResearchPlan,
    WorkerResult,
    WorkerRole,
    WorkerStatus,
)

TerminalDurableRunStatus = Literal[
    "completed",
    "partial",
    "failed",
]

DurableRunStatus = Literal[
    "running",
    "completed",
    "partial",
    "failed",
]

RetryRoute = Literal["retry", "done"]

REQUIRED_ROLES: tuple[WorkerRole, ...] = (
    "papers",
    "documentation_code",
    "skeptic",
)

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def merge_mappings[ValueT](
    left: dict[str, ValueT],
    right: dict[str, ValueT],
) -> dict[str, ValueT]:
    """Merge independent parallel state updates by key."""

    merged = dict(left)
    merged.update(right)
    return merged


class DurableResearchState(TypedDict):
    """Primitive-only state persisted by LangGraph."""

    run_id: str
    plan: dict[str, object]
    plan_fingerprint: str

    results: Annotated[
        dict[str, dict[str, object]],
        merge_mappings,
    ]
    attempts: Annotated[
        dict[str, int],
        merge_mappings,
    ]
    routes: Annotated[
        dict[str, RetryRoute],
        merge_mappings,
    ]
    attempt_records: Annotated[
        dict[str, dict[str, object]],
        merge_mappings,
    ]

    status: DurableRunStatus
    final_report: str


class DurableResearchUpdate(TypedDict, total=False):
    """Partial update returned by one graph node."""

    results: dict[str, dict[str, object]]
    attempts: dict[str, int]
    routes: dict[str, RetryRoute]
    attempt_records: dict[str, dict[str, object]]
    status: DurableRunStatus
    final_report: str


class AttemptRecord(BaseModel):
    """Durable audit record for one worker attempt."""

    model_config = {"frozen": True}

    role: WorkerRole
    attempt: int = Field(ge=1)
    status: WorkerStatus
    error_code: str | None = None
    duration_ms: float = Field(ge=0)
    retry_scheduled: bool
    backoff_seconds: float = Field(ge=0)


class DurableResearchRun(BaseModel):
    """Materialized result returned by the durable runtime."""

    model_config = {"frozen": True}

    run_id: str
    plan: ResearchPlan
    results: tuple[WorkerResult, ...]
    attempts: dict[WorkerRole, int]
    attempt_records: tuple[AttemptRecord, ...]
    status: TerminalDurableRunStatus
    final_report: str = Field(min_length=1)
    resumed: bool


def normalize_run_id(run_id: str) -> str:
    normalized = run_id.strip()

    if not _RUN_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            "run_id must be 1-64 characters and contain only "
            "letters, digits, periods, underscores, or hyphens; "
            "the first character must be alphanumeric"
        )

    return normalized


def plan_fingerprint(plan: ResearchPlan) -> str:
    """Hash a canonical JSON representation of the plan."""

    canonical = json.dumps(
        plan.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_initial_state(
    run_id: str,
    plan: ResearchPlan,
) -> DurableResearchState:
    normalized_run_id = normalize_run_id(run_id)

    actual_roles = {assignment.role for assignment in plan.assignments}
    required_roles = set(REQUIRED_ROLES)

    if actual_roles != required_roles:
        raise ValueError(
            "durable research requires exactly the required roles: "
            "papers, documentation_code, and skeptic"
        )

    return DurableResearchState(
        run_id=normalized_run_id,
        plan=plan.model_dump(mode="json"),
        plan_fingerprint=plan_fingerprint(plan),
        results={},
        attempts={},
        routes={},
        attempt_records={},
        status="running",
        final_report="",
    )
