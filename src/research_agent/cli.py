from enum import Enum
from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console
from rich.table import Table

from research_agent.domain import AgentResult
from research_agent.errors import AgentError
from research_agent.loop import AgentLoop
from research_agent.memory.cli import memory_app
from research_agent.models.demo import DemoModel
from research_agent.retrieval.embeddings import SentenceTransformerEmbedder
from research_agent.retrieval.evaluation import evaluate_recall, load_cases
from research_agent.retrieval.lab import RetrievalLab, RetrievalMode
from research_agent.retrieval.models import SearchHit
from research_agent.tools import ToolRegistry, WordCountTool

app = typer.Typer(
    no_args_is_help=True,
    help="Run the research CLI agent and retrieval laboratory.",
)
console = Console()

app.add_typer(
    memory_app,
    name="memory",
)

class SearchMode(str, Enum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


class DeviceMode(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


def normalize_words(words: list[str], parameter_name: str) -> str:
    text = " ".join(words).strip()
    if not text:
        raise typer.BadParameter(
            "must not be empty",
            param_hint=parameter_name,
        )
    return text


def render_agent_result(
    result: AgentResult,
    show_trace: bool,
) -> None:
    console.print(f"[bold green]{result.answer}[/bold green]")

    if not show_trace:
        return

    table = Table("Step", "Event", "Detail")
    for event in result.trace:
        table.add_row(
            str(event.step),
            event.event,
            event.detail,
        )
    console.print(table)


def build_retrieval_lab(
    corpus: Path,
    device: DeviceMode,
) -> RetrievalLab:
    selected_device = None if device is DeviceMode.AUTO else device.value
    embedder = SentenceTransformerEmbedder(device=selected_device)

    return RetrievalLab.from_corpus(
        corpus=corpus,
        embedder=embedder,
    )


def render_search_hits(
    lab: RetrievalLab,
    hits: list[SearchHit],
) -> None:
    table = Table(
        "Rank",
        "Score",
        "Source",
        "Chunk",
        "Text",
    )

    for hit in hits:
        chunk = lab.chunk_for(hit.chunk_id)
        table.add_row(
            str(hit.rank),
            f"{hit.score:.5f}",
            chunk.source_path.as_posix(),
            str(chunk.chunk_index),
            chunk.text,
        )

    console.print(table)


@app.command()
def demo(
    text: Annotated[
        list[str],
        typer.Argument(help="Text processed by the demo agent."),
    ],
    trace: Annotated[
        bool,
        typer.Option("--trace/--no-trace"),
    ] = True,
) -> None:
    """Run the bounded agent loop with a deterministic model."""
    query = normalize_words(text, "text")

    agent = AgentLoop(
        model=DemoModel(),
        registry=ToolRegistry([WordCountTool()]),
        max_steps=4,
    )

    try:
        result = agent.run(query)
    except AgentError as exc:
        console.print(f"[bold red]Agent failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    render_agent_result(result, show_trace=trace)


@app.command()
def retrieve(
    query: Annotated[
        list[str],
        typer.Argument(help="Research query."),
    ],
    corpus: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Directory containing Markdown or text documents.",
        ),
    ] = Path("examples/corpus"),
    mode: Annotated[
        SearchMode,
        typer.Option(help="dense, bm25, or hybrid"),
    ] = SearchMode.HYBRID,
    top_k: Annotated[
        int,
        typer.Option(min=1, max=50),
    ] = 5,
    device: Annotated[
        DeviceMode,
        typer.Option(help="auto, cpu, or cuda"),
    ] = DeviceMode.AUTO,
) -> None:
    """Search the local corpus without invoking a chat model."""
    text = normalize_words(query, "query")

    try:
        lab = build_retrieval_lab(corpus, device)
        hits = lab.search(
            text,
            mode=cast(RetrievalMode, mode.value),
            top_k=top_k,
        )
    except ValueError as exc:
        console.print(f"[bold red]Retrieval failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    render_search_hits(lab, hits)


@app.command()
def evaluate_retrieval(
    corpus: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ] = Path("examples/corpus"),
    cases: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = Path("evals/retrieval_cases.json"),
    top_k: Annotated[
        int,
        typer.Option(min=1, max=50),
    ] = 2,
    device: Annotated[
        DeviceMode,
        typer.Option(help="auto, cpu, or cuda"),
    ] = DeviceMode.AUTO,
) -> None:
    """Compare dense, BM25, and hybrid Recall at k."""
    try:
        lab = build_retrieval_lab(corpus, device)
        labeled_cases = load_cases(cases)
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]Evaluation failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    modes: tuple[RetrievalMode, ...] = (
        "dense",
        "bm25",
        "hybrid",
    )
    table = Table(
        "Method",
        f"Recall@{top_k}",
        "Queries",
    )

    for mode in modes:
        report = evaluate_recall(
            lab.retriever(mode),
            labeled_cases,
            k=top_k,
        )
        table.add_row(
            mode,
            f"{report.mean_recall:.3f}",
            str(report.case_count),
        )

    console.print(table)


if __name__ == "__main__":
    app()