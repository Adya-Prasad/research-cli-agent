"""Application-owned timeout and retry policy."""

from typing import Self

from pydantic import BaseModel, Field, model_validator

from research_agent.orchestration.models import WorkerResult

_DEFAULT_RETRYABLE_ERROR_CODES = frozenset(
    {
        "SourceTimeout",
        "SourceRateLimited",
        "SourceUnavailable",
    }
)


class WorkerExecutionPolicy(BaseModel):
    """Bound one worker's attempts, deadline, and retry delay.

    `max_attempts` includes the initial attempt. Therefore,
    max_attempts=2 permits one initial attempt and one retry.
    """

    model_config = {"frozen": True}

    max_attempts: int = Field(default=2, ge=1, le=5)
    timeout_seconds: float = Field(
        default=15.0,
        gt=0,
        le=300,
    )
    initial_backoff_seconds: float = Field(
        default=0.25,
        ge=0,
        le=60,
    )
    max_backoff_seconds: float = Field(
        default=2.0,
        ge=0,
        le=120,
    )
    jitter_ratio: float = Field(
        default=0.20,
        ge=0,
        le=1,
    )
    retryable_error_codes: frozenset[str] = _DEFAULT_RETRYABLE_ERROR_CODES

    @model_validator(mode="after")
    def validate_backoff_bounds(self) -> Self:
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError(
                "max_backoff_seconds must be greater than or equal to initial_backoff_seconds"
            )
        return self

    def should_retry(
        self,
        result: WorkerResult,
        *,
        attempts_used: int,
    ) -> bool:
        """Return whether another attempt is authorized."""

        if attempts_used < 1:
            raise ValueError("attempts_used must be at least 1")

        if attempts_used >= self.max_attempts:
            return False

        if result.status == "timed_out":
            return True

        if result.status != "failed":
            return False

        return result.error_code in self.retryable_error_codes

    def backoff_seconds(
        self,
        attempts_used: int,
        *,
        random_fraction: float = 0.5,
    ) -> float:
        """Calculate bounded exponential backoff with jitter.

        `random_fraction` must be in [0, 1]. Injecting it makes
        randomness controllable in tests. A value of 0.5 applies
        no positive or negative jitter.
        """

        if attempts_used < 1:
            raise ValueError("attempts_used must be at least 1")

        if not 0.0 <= random_fraction <= 1.0:
            raise ValueError("random_fraction must be between 0 and 1")

        exponential_delay = self.initial_backoff_seconds * (2 ** (attempts_used - 1))
        bounded_delay = min(
            exponential_delay,
            self.max_backoff_seconds,
        )

        centered_random = (2.0 * random_fraction) - 1.0
        jitter_multiplier = 1.0 + self.jitter_ratio * centered_random

        return min(
            self.max_backoff_seconds,
            max(0.0, bounded_delay * jitter_multiplier),
        )
