from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from research_agent.loop import AgentLoop
from research_agent.models.demo import DemoModel
from research_agent.tools import ToolRegistry, WordCountTool

app = typer.Typer(no_args_is_help=True, help="Run the research cli agent")

console = Console()


@app.command()
def demo(text: Annotated[list[str], typer.Argument(...)]) -> None:
    query = " ".join(text)

    """Run the minimal agent with a determinsitic model"""
    model = DemoModel()
    registry = ToolRegistry([WordCountTool()])

    agent = AgentLoop(model=model, registry=registry, max_steps=4)

    table = Table("step", "Event", "Detail")

    result = agent.run(query)

    for event in result.trace:
        table.add_row(
            str(event.step),
            event.event,
            event.detail,
        )

    console.print(table)


if __name__ == "__main__":
    app()
