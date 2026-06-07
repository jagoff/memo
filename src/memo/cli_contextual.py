"""`memo contextual` command group — context-aware search + preferences.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(contextual_group)`.
"""

from __future__ import annotations

import json

import click
from rich.table import Table

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config

# -- contextual recall commands -----------------------------------------------


@click.group(name="contextual")
def contextual_group() -> None:
    """Contextual recall with conversation history and preference learning."""
    pass


@contextual_group.command(name="search")
@click.argument("query")
@click.option("--limit", type=int, default=10, help="Max results (default: 10)")
@click.option(
    "--mode",
    type=click.Choice(["vec", "bm25", "hybrid"]),
    default="hybrid",
    help="Search mode (default: hybrid)",
)
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def contextual_search(query: str, limit: int, mode: str, as_json: bool) -> None:
    """Search with contextual re-ranking based on conversation history.

    Example: memo contextual search "MLX performance"
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    results = mem.contextual.search_with_context(
        query=query,
        limit=limit,
        mode=mode,
    )

    if as_json:
        click.echo(json.dumps([r.__dict__ for r in results], indent=2))
        return

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    table = Table(title=f"Contextual Search Results for '{query}'")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="yellow")
    table.add_column("Original Score", style="dim")
    table.add_column("Contextual Score", style="green")
    table.add_column("Boost Factors", style="magenta")

    for r in results[:20]:
        boosts = ", ".join(f"{k}={v:.2f}" for k, v in r.boost_factors.items())
        table.add_row(
            r.memoria_id[:8],
            r.title[:40],
            f"{r.original_score:.3f}",
            f"{r.contextual_score:.3f}",
            boosts or "—",
        )

    console.print(table)
    if len(results) > 20:
        console.print(f"[dim]...and {len(results) - 20} more[/dim]")


@contextual_group.command(name="record-search")
@click.argument("query")
@click.argument("memoria_ids", nargs=-1, required=True)
def contextual_record_search(query: str, memoria_ids: tuple[str, ...]) -> None:
    """Record a search in the conversation history for learning.

    Example: memo contextual record-search "MLX" abc123 def456
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    mem.contextual.record_search(query, list(memoria_ids))
    console.print(f"[green]Recorded search with {len(memoria_ids)} recalled memorias[/green]")


@contextual_group.command(name="record-click")
@click.argument("memoria_id")
def contextual_record_click(memoria_id: str) -> None:
    """Record that the user clicked/viewed a memoria (for preference learning).

    Example: memo contextual record-click abc123
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    mem.contextual.record_click(memoria_id)
    console.print(f"[green]Recorded click for memoria {memoria_id[:8]}[/green]")


@contextual_group.command(name="preferences")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def contextual_preferences(as_json: bool) -> None:
    """Show learned user preferences for memory recall.

    Example: memo contextual preferences
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    prefs = mem.contextual.context.get_preferences()

    if as_json:
        click.echo(json.dumps(prefs.__dict__, indent=2))
        return

    console.print("[bold]User Preferences[/bold]")
    console.print()

    console.print(f"[yellow]Recency Weight:[/yellow] {prefs.recency_weight:.2f}")
    console.print(f"[yellow]Diversity Weight:[/yellow] {prefs.diversity_weight:.2f}")
    console.print(f"[yellow]Last Updated:[/yellow] {prefs.last_updated or 'Never'}")
    console.print()

    console.print("[yellow]Preferred Memory Types:[/yellow]")
    if prefs.preferred_types:
        for type_, score in sorted(prefs.preferred_types.items(), key=lambda x: x[1], reverse=True):
            console.print(f"  {type_}: {score:.2f}")
    else:
        console.print("  [dim]No preferences learned yet[/dim]")
    console.print()

    console.print("[yellow]Preferred Entities:[/yellow]")
    if prefs.preferred_entities:
        for entity, score in sorted(
            prefs.preferred_entities.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            console.print(f"  {entity}: {score:.2f}")
        if len(prefs.preferred_entities) > 10:
            console.print(f"  [dim]...and {len(prefs.preferred_entities) - 10} more[/dim]")
    else:
        console.print("  [dim]No preferences learned yet[/dim]")


@contextual_group.command(name="history")
@click.option(
    "--limit", type=int, default=10, help="Number of recent prompts to show (default: 10)"
)
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def contextual_history(limit: int, as_json: bool) -> None:
    """Show recent conversation history used for contextual recall.

    Example: memo contextual history --limit 5
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    history = mem.contextual.context.get_recent_context(n=limit)

    if as_json:
        click.echo(json.dumps([c.__dict__ for c in history], indent=2))
        return

    if not history:
        console.print("[dim]No conversation history yet[/dim]")
        return

    console.print(f"[bold]Recent {len(history)} Prompts[/bold]")
    console.print()

    for i, ctx in enumerate(history, 1):
        console.print(f"[cyan]{i}. {ctx.timestamp}[/cyan]")
        console.print(f"   Prompt: {ctx.prompt[:80]}")
        console.print(f"   Recalled: {len(ctx.recalled_memorias)} memoria(s)")
        console.print()


@contextual_group.command(name="reset-preferences")
@click.confirmation_option(prompt="This will reset all learned preferences. Continue?")
def contextual_reset_preferences() -> None:
    """Reset all learned user preferences.

    Example: memo contextual reset-preferences
    """
    cfg = Config.from_env()

    # Reset preferences file
    prefs_file = cfg.state_dir / "user_preferences.json"
    if prefs_file.is_file():
        prefs_file.unlink()

    # Reload will create fresh defaults
    console.print("[green]Preferences reset successfully[/green]")
