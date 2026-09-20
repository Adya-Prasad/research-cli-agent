"""Write machine, human, and trajectory evaluation artifacts."""

import json
from dataclasses import dataclass
from pathlib import Path

from research_agent.evaluation.models import (
    EvalBaseline,
    EvalReport,
)


@dataclass(frozen=True, slots=True)
class ReportPaths:
    json_report: Path
    markdown_summary: Path
    trajectory_jsonl: Path


def _optional(value: int | None) -> str:
    return "unavailable" if value is None else str(value)


def render_markdown(report: EvalReport) -> str:
    summary = report.summary

    lines = [
        "# Agent evaluation report",
        "",
        f"- Dataset: `{report.dataset_version}`",
        f"- Profile: `{report.profile}`",
        f"- System: `{report.system_name}`",
        f"- Tasks: {summary.task_count}",
        f"- Trials: {summary.trial_count}",
        (f"- Trial pass rate: {summary.trial_pass_rate:.3f}"),
        (f"- Consistent task rate: {summary.consistent_task_rate:.3f}"),
        f"- Pass@k: {summary.pass_at_k:.3f}",
        (f"- Mean citation integrity: {summary.mean_citation_integrity:.3f}"),
        f"- p50 latency: {summary.p50_latency_ms:.2f} ms",
        f"- p95 latency: {summary.p95_latency_ms:.2f} ms",
        f"- Model calls: {summary.total_model_calls}",
        f"- Tool calls: {summary.total_tool_calls}",
        (f"- Input tokens: {_optional(summary.total_input_tokens)}"),
        (f"- Output tokens: {_optional(summary.total_output_tokens)}"),
        (f"- Nominal cost (micro-USD): {_optional(summary.total_nominal_cost_microusd)}"),
        (f"- Regression gate: {'PASS' if report.regression.passed else 'FAIL'}"),
        "",
        "## Trial results",
        "",
        "| Task | Trial | Passed | Error |",
        "|---|---:|---:|---|",
    ]

    for artifact in report.artifacts:
        lines.append(
            "| "
            f"{artifact.task_id} | "
            f"{artifact.trial_index} | "
            f"{'yes' if artifact.passed else 'no'} | "
            f"{artifact.error or ''} |"
        )

    if report.regression.reasons:
        lines.extend(
            [
                "",
                "## Regression failures",
                "",
                *[f"- {reason}" for reason in report.regression.reasons],
            ]
        )

    return "\n".join(lines) + "\n"


def write_eval_report(
    report: EvalReport,
    output_directory: Path,
) -> ReportPaths:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = output_directory / "report.json"
    markdown_path = output_directory / "summary.md"
    trajectory_path = output_directory / "trajectories.jsonl"

    json_path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_markdown(report),
        encoding="utf-8",
    )

    rows: list[str] = []

    for artifact in report.artifacts:
        if artifact.output is None:
            continue

        for event in artifact.output.trajectory:
            rows.append(
                json.dumps(
                    {
                        "task_id": artifact.task_id,
                        "trial_id": artifact.trial_id,
                        "profile": artifact.profile,
                        "event": event.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )

    trajectory_path.write_text(
        "\n".join(rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )

    return ReportPaths(
        json_report=json_path,
        markdown_summary=markdown_path,
        trajectory_jsonl=trajectory_path,
    )


def write_baseline(
    report: EvalReport,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    baseline = EvalBaseline(
        dataset_version=report.dataset_version,
        profile=report.profile,
        summary=report.summary,
    )
    path.write_text(
        baseline.model_dump_json(indent=2),
        encoding="utf-8",
    )


def load_baseline(path: Path) -> EvalBaseline:
    return EvalBaseline.model_validate_json(path.read_text(encoding="utf-8"))
