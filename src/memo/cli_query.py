"""`memo query` command group — saved queries.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(query_group)`.
"""

from __future__ import annotations

import json

import click
from rich.table import Table

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config

# -- query composition commands ------------------------------------------------


@click.group(name="query")
def query_group() -> None:
    """Query composition and saved queries."""
    pass


@query_group.command(name="save")
@click.argument("name")
@click.argument("query_text")
@click.option("--type", "type_filter", help="Filter by memory type")
@click.option("--tags", "tags_filter", multiple=True, help="Filter by tags")
@click.option("--date-from", help="Start date (ISO format)")
@click.option("--date-to", help="End date (ISO format)")
@click.option(
    "--mode",
    "search_mode",
    type=click.Choice(["vec", "bm25", "hybrid"]),
    default="hybrid",
    help="Search mode (default: hybrid)",
)
@click.option("--limit", type=int, default=10, help="Result limit")
@click.option("--description", help="Query description")
@click.option("--execute", is_flag=True, help="Execute the query after saving")
def query_save(
    name: str,
    query_text: str,
    type_filter: str | None,
    tags_filter: tuple[str, ...],
    date_from: str | None,
    date_to: str | None,
    search_mode: str,
    limit: int,
    description: str | None,
    execute: bool,
) -> None:
    """Save a query for reuse.

    Example: memo query save "MLX decisions" "MLX" --type decision --execute
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    if execute:
        result = mem.query_composer.compose_and_save(
            name=name,
            query_text=query_text,
            type_filter=type_filter,
            tags_filter=list(tags_filter),
            date_from=date_from,
            date_to=date_to,
            search_mode=search_mode,
            limit=limit,
            description=description,
        )
        console.print(f"[green]Saved and executed query '{name}'[/green]")
        console.print(f"Results: {result.count}")
    else:
        mem.query_composer.query_store.save_query(
            name=name,
            query_text=query_text,
            type_filter=type_filter,
            tags_filter=list(tags_filter),
            date_from=date_from,
            date_to=date_to,
            search_mode=search_mode,
            limit=limit,
            description=description,
        )
        console.print(f"[green]Saved query '{name}'[/green]")


@query_group.command(name="list")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def query_list(as_json: bool) -> None:
    """List all saved queries.

    Example: memo query list
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    queries = mem.query_composer.query_store.list_queries()

    if as_json:
        click.echo(json.dumps([q.__dict__ for q in queries], indent=2))
        return

    if not queries:
        console.print("[dim]No saved queries[/dim]")
        return

    table = Table(title="Saved Queries")
    table.add_column("Name", style="cyan")
    table.add_column("Query Text", style="yellow")
    table.add_column("Type Filter", style="green")
    table.add_column("Mode", style="magenta")
    table.add_column("Description", style="dim")

    for q in queries[:20]:
        table.add_row(
            q.name,
            q.query_text[:40],
            q.type_filter or "—",
            q.search_mode,
            q.description or "—",
        )

    console.print(table)
    if len(queries) > 20:
        console.print(f"[dim]...and {len(queries) - 20} more[/dim]")


@query_group.command(name="run")
@click.argument("name")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def query_run(name: str, as_json: bool) -> None:
    """Execute a saved query.

    Example: memo query run "MLX decisions"
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    query = mem.query_composer.query_store.get_query(name)
    if not query:
        console.print(f"[yellow]Query '{name}' not found[/yellow]")
        return

    result = mem.query_composer.execute_query(query)

    if as_json:
        # Convert results to dict format
        results_dict = [r.__dict__ for r in result.results]
        click.echo(
            json.dumps(
                {
                    "query_name": result.query_name,
                    "count": result.count,
                    "executed_at": result.executed_at,
                    "results": results_dict,
                },
                indent=2,
            )
        )
        return

    console.print(f"[bold]Query: {name}[/bold]")
    console.print(f"Results: {result.count}")
    console.print()

    for r in result.results[:10]:
        console.print(f"  [cyan]{r.id[:8]}[/cyan] {r.title}")
        console.print(f"    {r.body[:100]}")

    if len(result.results) > 10:
        console.print(f"  [dim]...and {len(result.results) - 10} more[/dim]")


@query_group.command(name="delete")
@click.argument("name")
@click.confirmation_option(prompt="Delete this saved query?")
def query_delete(name: str) -> None:
    """Delete a saved query.

    Example: memo query delete "MLX decisions"
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    success = mem.query_composer.query_store.delete_query(name)

    if success:
        console.print(f"[green]Deleted query '{name}'[/green]")
    else:
        console.print(f"[yellow]Query '{name}' not found[/yellow]")
