"""CLI commands for explicit durable-memory operations."""

import os
from enum import Enum
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from research_agent.memory.errors import MemoryPolicyRejected, MemoryProviderError
from research_agent.memory.models import MemoryCandidate
from research_agent.memory.policy import MemoryPolicy
from research_agent.memory.service import MemoryService
from research_agent.memory.supermemory import SupermemoryStore

memory_app = typer.Typer(
    no_args_is_help=True,
    help="Manage policy-approved long term memory",
)
console = Console()

class MemoryKindOption(str, Enum):
    USER_PREFERENCE = "user_preference"
    DURABLE_FACT = "durable_fact"
    RESEARCH_INTEREST = "research_interest"
    SESSION_SUMMARY = "session_summary"

def _join_word(words: list[str], parameter_name: str) -> str:
    text = " ".join(words).strip()
    if not text:
        raise typer.BadParameter(f"{parameter_name} cannot be empty")
    return text

def build_memory_service() -> MemoryService:
    api_key = os.environ.get("SUPERMEMORY_API_KEY", "").strip()
    if not api_key:
        raise MemoryProviderError("SUPERMEMORY_API_KEY is not configured")
    return MemoryService(
        policy=MemoryPolicy(),
        store=SupermemoryStore.from_api_key(api_key),
    )

@memory_app.command()
def remember(
    content: Annotated[
        list[str],
        typer.Argument(help="Durable information to remember."),
    ],
    user_id: Annotated[
        str, 
        typer.Option(help="Stable local user identifier"),
    ] = "local_research",

    session_id: Annotated[
        str,
        typer.Option(help="Stable session identifier"),
    ] = "manual-session",
    kind: Annotated[
        MemoryKindOption,
        typer.Option(help="Allowlisted durable-memory category.")
    ] = MemoryKindOption.RESEARCH_INTEREST,
) -> None:
    """Persist one explicit, policy-approved memory."""
    candidate = MemoryCandidate(
        user_id=user_id,
        session_id=session_id,
        kind = kind.value,
        content = _join_word(content, "content"),
    )
    try:
        receipt = build_memory_service().remember(candidate)
    except MemoryPolicyRejected as exc:
        console.print(
            f"[bold red]Memory rejected:[/bold red] {exc.reason} "
        )
        raise typer.Exit(code=1) from exc
    except MemoryProviderError as exc:
        console.print(
            f"[bold red]Memory provider failed:[/bold red] {exc}"
        )
        raise typer.Exit(code=1) from exc

    console.print("[bold green]Memory accepted[/bold green]")
    console.print(f"ID: {receipt.memory_id}")
    console.print(f"Provider status: {receipt.status}")

@memory_app.command()
def recall(
    query: Annotated[
        list[str],
        typer.Argument(help="Semantic memory query."),
    ],
    user_id: Annotated[
        str,
        typer.Option(help="Stable local user identifier."),
    ] = "local-researcher",
    top_k: Annotated[
        int,
        typer.Option(min=1, max=20),
    ] = 5,
) -> None:

    """Retrieve long-term memory for one user."""
    try: 
        hits = build_memory_service().recall(
            user_id=user_id,
            query=_join_word(query, "query"),
            top_k=top_k,
        )
    except MemoryProviderError as exc:
        console.print(
            f"[bold red]Memory provider failed:[/bold red]{exc}"
        )
        raise typer.Exit(code=1) from exc

    if not hits:
        console.print("[yellow]No memories found. [/yellow]")
        return

    table = Table()
    table.add_column("Rank")
    table.add_column("Score")
    table.add_column("Kind")
    table.add_column("Session")
    table.add_column("Memory ID", overflow="fold")
    table.add_column("content")
    for hit in hits:
        table.add_row(
            str(hit.rank),
            f"{hit.score:.4f}",
            hit.kind,
            hit.session_id,
            hit.memory_id,
            hit.content
        )

    console.print(table)

@memory_app.command()
def forget(
    memory_id: Annotated[
        str, 
        typer.Argument(help="Logical memory ID returned by remember"),
    ],
    user_id: Annotated[
        str,
        typer.Option(help="Owner of the memory"),
    ] = "local-researcher",
) -> None:
    """Delete one owned long term memory"""
    try:
        deleted = build_memory_service().forget(
            user_id=user_id,
            memory_id=memory_id,
        )
    except MemoryProviderError as exc:
        console.print(
            f"[bold red]Memory provider failed: [/bold red]{exc}"
        )
        raise typer.Exit(code=1) from exc

    if not deleted:
        console.print(
            "[bold red]Memory not found or ownership mismatch.[/bold red]"
        )
        raise typer.Exit(code=1)

    console.print("[bold green]Memory deleted[/bold green]")