import pytest

from research_agent.durable.policy import WorkerExecutionPolicy
from research_agent.orchestration.models import (
    WorkerResult,
    WorkerStatus,
)


def _result(
    status: WorkerStatus,
    error_code: str | None = None,
) -> WorkerResult:
    return WorkerResult(
        task_id="papers",
        role="papers",
        status=status,
        error_code=error_code,
        started_at=1.0,
        finished_at=2.0,
    )


@pytest.mark.parametrize(
    "error_code",
    [
        "SourceTimeout",
        "SourceRateLimited",
        "SourceUnavailable",
    ],
)
def test_policy_retries_transient_provider_failures(
    error_code: str,
) -> None:
    policy = WorkerExecutionPolicy(max_attempts=2)
    result = _result("failed", error_code)

    assert policy.should_retry(
        result,
        attempts_used=1,
    )


def test_policy_retries_worker_timeout() -> None:
    policy = WorkerExecutionPolicy(max_attempts=2)

    assert policy.should_retry(
        _result("timed_out", "worker_timeout"),
        attempts_used=1,
    )


@pytest.mark.parametrize(
    "result",
    [
        _result("completed"),
        _result("budget_exhausted", "tool_budget_exhausted"),
        _result("failed", "SourcePolicyRejected"),
        _result("failed", "ValueError"),
    ],
)
def test_policy_does_not_retry_permanent_outcomes(
    result: WorkerResult,
) -> None:
    policy = WorkerExecutionPolicy(max_attempts=3)

    assert not policy.should_retry(
        result,
        attempts_used=1,
    )


def test_policy_stops_after_total_attempt_budget() -> None:
    policy = WorkerExecutionPolicy(max_attempts=2)
    result = _result("failed", "SourceUnavailable")

    assert policy.should_retry(result, attempts_used=1)
    assert not policy.should_retry(result, attempts_used=2)


def test_backoff_is_exponential_and_bounded() -> None:
    policy = WorkerExecutionPolicy(
        max_attempts=5,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=4.0,
        jitter_ratio=0.0,
    )

    assert policy.backoff_seconds(1) == 1.0
    assert policy.backoff_seconds(2) == 2.0
    assert policy.backoff_seconds(3) == 4.0
    assert policy.backoff_seconds(4) == 4.0


def test_backoff_supports_deterministic_jitter() -> None:
    policy = WorkerExecutionPolicy(
        initial_backoff_seconds=1.0,
        max_backoff_seconds=4.0,
        jitter_ratio=0.25,
    )

    assert policy.backoff_seconds(
        1,
        random_fraction=0.0,
    ) == pytest.approx(0.75)

    assert policy.backoff_seconds(
        1,
        random_fraction=0.5,
    ) == pytest.approx(1.0)

    assert policy.backoff_seconds(
        1,
        random_fraction=1.0,
    ) == pytest.approx(1.25)


def test_backoff_rejects_invalid_attempt_number() -> None:
    policy = WorkerExecutionPolicy()

    with pytest.raises(ValueError, match="attempts_used"):
        policy.backoff_seconds(0)


def test_backoff_rejects_invalid_random_fraction() -> None:
    policy = WorkerExecutionPolicy()

    with pytest.raises(ValueError, match="random_fraction"):
        policy.backoff_seconds(
            1,
            random_fraction=1.1,
        )


def test_policy_rejects_inverted_backoff_bounds() -> None:
    with pytest.raises(
        ValueError,
        match="max_backoff_seconds",
    ):
        WorkerExecutionPolicy(
            initial_backoff_seconds=2.0,
            max_backoff_seconds=1.0,
        )
