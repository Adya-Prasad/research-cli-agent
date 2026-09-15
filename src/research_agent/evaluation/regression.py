"""Absolute and baseline-relative regression gates."""

from research_agent.evaluation.models import (
    EvalBaseline,
    EvalSummary,
    RegressionDecision,
    RegressionThresholds,
)


def evaluate_regression(
    *,
    dataset_version: str,
    summary: EvalSummary,
    thresholds: RegressionThresholds,
    baseline: EvalBaseline | None = None,
) -> RegressionDecision:
    reasons: list[str] = []

    if (
        summary.trial_pass_rate
        < thresholds.minimum_trial_pass_rate
    ):
        reasons.append(
            "trial pass rate below absolute threshold"
        )

    if (
        summary.consistent_task_rate
        < thresholds.minimum_consistent_task_rate
    ):
        reasons.append(
            "consistent task rate below absolute threshold"
        )

    if (
        summary.mean_citation_integrity
        < thresholds.minimum_citation_integrity
    ):
        reasons.append(
            "citation integrity below absolute threshold"
        )

    if (
        thresholds.maximum_p95_latency_ms is not None
        and summary.p95_latency_ms
        > thresholds.maximum_p95_latency_ms
    ):
        reasons.append(
            "p95 latency exceeds absolute threshold"
        )

    maximum_cost = (
        thresholds.maximum_nominal_cost_microusd
    )
    if (
        maximum_cost is not None
        and (
            summary.total_nominal_cost_microusd is None
            or summary.total_nominal_cost_microusd
            > maximum_cost
        )
    ):
        reasons.append(
            "nominal cost is unavailable or exceeds threshold"
        )

    if baseline is not None:
        if baseline.dataset_version != dataset_version:
            reasons.append(
                "baseline dataset version does not match"
            )
        else:
            pass_rate_drop = (
                baseline.summary.trial_pass_rate
                - summary.trial_pass_rate
            )

            if (
                pass_rate_drop
                > thresholds.maximum_pass_rate_drop
            ):
                reasons.append(
                    "trial pass rate regressed from baseline"
                )

    return RegressionDecision(
        passed=not reasons,
        reasons=tuple(reasons),
    )