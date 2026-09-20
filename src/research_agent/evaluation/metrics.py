"""Aggregate trial metrics without hiding task-level variance."""

import math

from research_agent.evaluation.models import (
    EvalSummary,
    TrialArtifact,
)


def _percentile(
    values: list[float],
    quantile: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _optional_sum(
    values: list[int | None],
) -> int | None:
    if any(value is None for value in values):
        return None

    return sum(value for value in values if value is not None)


def build_summary(
    artifacts: tuple[TrialArtifact, ...],
) -> EvalSummary:
    if not artifacts:
        raise ValueError("cannot summarize zero trial artifacts")

    grouped: dict[str, list[TrialArtifact]] = {}

    for artifact in artifacts:
        grouped.setdefault(
            artifact.task_id,
            [],
        ).append(artifact)

    passed_trials = sum(artifact.passed for artifact in artifacts)
    task_count = len(grouped)

    consistent_tasks = sum(all(item.passed for item in trials) for trials in grouped.values())
    tasks_with_any_success = sum(any(item.passed for item in trials) for trials in grouped.values())

    outputs = [artifact.output for artifact in artifacts if artifact.output is not None]

    latencies = [output.duration_ms for output in outputs]
    citation_scores = [output.report.ledger.verified_fraction for output in outputs]

    return EvalSummary(
        task_count=task_count,
        trial_count=len(artifacts),
        passed_trials=passed_trials,
        trial_pass_rate=(passed_trials / len(artifacts)),
        consistent_task_rate=(consistent_tasks / task_count),
        pass_at_k=(tasks_with_any_success / task_count),
        mean_citation_integrity=(
            sum(citation_scores) / len(citation_scores) if citation_scores else 0.0
        ),
        p50_latency_ms=_percentile(
            latencies,
            0.50,
        ),
        p95_latency_ms=_percentile(
            latencies,
            0.95,
        ),
        total_model_calls=sum(output.usage.model_calls for output in outputs),
        total_tool_calls=sum(output.usage.tool_calls for output in outputs),
        total_input_tokens=_optional_sum([output.usage.input_tokens for output in outputs]),
        total_output_tokens=_optional_sum([output.usage.output_tokens for output in outputs]),
        total_nominal_cost_microusd=_optional_sum(
            [output.usage.nominal_cost_microusd for output in outputs]
        ),
    )
