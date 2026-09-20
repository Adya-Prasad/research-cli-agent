"""CLI entrypoint for production agent evaluations."""

import asyncio
import os
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from research_agent.evaluation.gemini import (
    GeminiStructuredGenerator,
)
from research_agent.evaluation.graders import default_graders
from research_agent.evaluation.harness import (
    EvalHarness,
    limit_eval_suite,
    load_eval_suite,
)
from research_agent.evaluation.pricing import (
    load_price_card,
)
from research_agent.evaluation.reporting import (
    load_baseline,
    write_baseline,
    write_eval_report,
)
from research_agent.evaluation.systems import (
    DeterministicResearchSystem,
    GeminiSynthesisSystem,
)
from research_agent.retrieval.embeddings import (
    SentenceTransformerEmbedder,
)
from research_agent.retrieval.lab import RetrievalLab

eval_app = typer.Typer(
    invoke_without_command=True,
    no_args_is_help=False,
    help="Evaluate complete research-agent behavior.",
)
console = Console()


class ProfileOption(str, Enum):
    DETERMINISTIC = "deterministic"
    GEMINI = "gemini"


class EvalDevice(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


@eval_app.callback()
def evaluate_agent(
    suite_path: Annotated[
        Path,
        typer.Option(
            "--suite",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = Path("evals/agent_tasks.json"),
    profile: Annotated[
        ProfileOption,
        typer.Option("--profile"),
    ] = ProfileOption.DETERMINISTIC,
    trials: Annotated[
        int | None,
        typer.Option(
            "--trials",
            min=1,
            max=5,
        ),
    ] = None,
    task_limit: Annotated[
        int | None,
        typer.Option(
            "--task-limit",
            min=1,
            help=("Evaluate only the first N tasks. Intended for smoke testing."),
        ),
    ] = None,
    corpus: Annotated[
        Path,
        typer.Option(
            "--corpus",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ] = Path("examples/corpus"),
    output: Annotated[
        Path,
        typer.Option("--output"),
    ] = Path("evals/results/latest"),
    device: Annotated[
        EvalDevice,
        typer.Option("--device"),
    ] = EvalDevice.CPU,
    model: Annotated[
        str,
        typer.Option("--model"),
    ] = "gemini-2.5-flash",
    price_card_path: Annotated[
        Path,
        typer.Option(
            "--price-card",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = Path("evals/pricing/gemini-2.5-flash.json"),
    baseline_path: Annotated[
        Path | None,
        typer.Option(
            "--baseline",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    baseline_output: Annotated[
        Path | None,
        typer.Option("--write-baseline"),
    ] = None,
    temperature: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=2.0,
        ),
    ] = 0.2,
) -> None:
    """Run task/trial/grader evaluation and emit artifacts."""

    try:
        suite = load_eval_suite(suite_path)
        baseline = load_baseline(baseline_path) if baseline_path is not None else None
        if task_limit is not None:
            suite = limit_eval_suite(
                suite,
                task_limit,
            )

        selected_device = None if device is EvalDevice.AUTO else device.value
        lab = RetrievalLab.from_corpus(
            corpus=corpus,
            embedder=SentenceTransformerEmbedder(device=selected_device),
        )
        deterministic = DeterministicResearchSystem(lab=lab)

        if profile is ProfileOption.DETERMINISTIC:
            system = deterministic
            trial_count = trials or 1
        else:
            api_key = os.getenv("GEMINI_API_KEY", "")

            if not api_key:
                raise ValueError("GEMINI_API_KEY is required for the Gemini evaluation profile")

            price_card = load_price_card(price_card_path)

            if price_card.model != model:
                raise ValueError("price-card model does not match the requested Gemini model")

            system = GeminiSynthesisSystem(
                base_system=deterministic,
                generator=GeminiStructuredGenerator(
                    api_key=api_key,
                    price_card=price_card,
                    model_name=model,
                    temperature=temperature,
                ),
            )
            trial_count = trials or 3

        report = asyncio.run(
            EvalHarness(
                system=system,
                graders=default_graders(),
                baseline=baseline,
            ).run(
                suite,
                trials_per_task=trial_count,
            )
        )

        paths = write_eval_report(
            report,
            output,
        )

        if baseline_output is not None:
            write_baseline(
                report,
                baseline_output,
            )

    except (OSError, ValueError) as exc:
        console.print(f"[bold red]Evaluation failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    summary = report.summary
    table = Table("Metric", "Value")
    table.add_row(
        "Trial pass rate",
        f"{summary.trial_pass_rate:.3f}",
    )
    table.add_row(
        "Consistent task rate",
        f"{summary.consistent_task_rate:.3f}",
    )
    table.add_row(
        "Pass@k",
        f"{summary.pass_at_k:.3f}",
    )
    table.add_row(
        "Citation integrity",
        f"{summary.mean_citation_integrity:.3f}",
    )
    table.add_row(
        "p95 latency",
        f"{summary.p95_latency_ms:.2f} ms",
    )
    table.add_row(
        "Model calls",
        str(summary.total_model_calls),
    )
    table.add_row(
        "Tool calls",
        str(summary.total_tool_calls),
    )
    table.add_row(
        "Nominal micro-USD",
        (
            str(summary.total_nominal_cost_microusd)
            if summary.total_nominal_cost_microusd is not None
            else "unavailable"
        ),
    )

    console.print(table)
    console.print(f"JSON report: {paths.json_report}")
    console.print(f"Markdown summary: {paths.markdown_summary}")
    console.print(f"Trajectories: {paths.trajectory_jsonl}")

    if not report.regression.passed:
        for reason in report.regression.reasons:
            console.print(f"[bold red]Gate failure:[/bold red] {reason}")
        raise typer.Exit(code=1)

    console.print("[bold green]Regression gate passed[/bold green]")
