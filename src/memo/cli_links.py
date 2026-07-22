"""`memo links` command group — backlinks / outlinks / link suggestions.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(links_group)`.
"""

from __future__ import annotations

import json

import click
from rich.table import Table

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config

# -- cross-reference commands -------------------------------------------------


@click.group(name="links")
def links_group() -> None:
    """Cross-reference and backlink system for memories."""
    pass


@links_group.command(name="backlinks")
@click.argument("memory_id")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def links_backlinks(memory_id: str, as_json: bool) -> None:
    """Show all memories that reference this one.

    Example: memo links backlinks abc123
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    # crossref stores full 32-char ids; resolve a user-supplied prefix
    # (e.g. the 12-char id printed by `memo list`) before the exact-match query.
    rec = mem.get(memory_id)
    if rec is None:
        console.print(f"[red]not found: {memory_id}[/red]")
        raise SystemExit(1)
    memory_id = rec.id

    backlinks = mem.crossref.get_backlinks(memory_id)

    if as_json:
        click.echo(json.dumps([b.__dict__ for b in backlinks], indent=2))
        return

    if not backlinks:
        console.print(f"[dim]No backlinks found for memory {memory_id[:8]}[/dim]")
        return

    table = Table(title=f"Backlinks to {memory_id[:8]}")
    table.add_column("Source ID", style="cyan")
    table.add_column("Link Type", style="yellow")
    table.add_column("Context", style="dim")

    for bl in backlinks[:20]:
        table.add_row(
            bl.source_id[:8],
            bl.link_type,
            bl.context[:60] if bl.context else "—",
        )

    console.print(table)
    if len(backlinks) > 20:
        console.print(f"[dim]...and {len(backlinks) - 20} more[/dim]")


@links_group.command(name="outlinks")
@click.argument("memory_id")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def links_outlinks(memory_id: str, as_json: bool) -> None:
    """Show all memories that this one references.

    Example: memo links outlinks abc123
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    # crossref stores full 32-char ids; resolve a user-supplied prefix
    # (e.g. the 12-char id printed by `memo list`) before the exact-match query.
    rec = mem.get(memory_id)
    if rec is None:
        console.print(f"[red]not found: {memory_id}[/red]")
        raise SystemExit(1)
    memory_id = rec.id

    outlinks = mem.crossref.get_outlinks(memory_id)

    if as_json:
        click.echo(json.dumps([o.__dict__ for o in outlinks], indent=2))
        return

    if not outlinks:
        console.print(f"[dim]No outlinks found for memory {memory_id[:8]}[/dim]")
        return

    console.print(f"[bold]Outlinks from {memory_id[:8]}[/bold]")
    console.print()

    for ol in outlinks:
        console.print(f"  [cyan]{ol.target}[/cyan]")


@links_group.command(name="suggest")
@click.argument("content")
@click.option("--title", help="Title of the memory being saved")
@click.option("--tags", multiple=True, help="Tags of the memory being saved")
@click.option("--limit", type=int, default=5, help="Max suggestions (default: 5)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def links_suggest(
    content: str, title: str | None, tags: tuple[str, ...], limit: int, as_json: bool
) -> None:
    """Suggest links to existing memories based on content.

    Example: memo links suggest "MLX performance optimization" --title "MLX"
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    suggestions = mem.link_suggester.suggest_links(
        content=content,
        title=title or "",
        tags=list(tags),
        limit=limit,
    )

    if as_json:
        click.echo(json.dumps([s.__dict__ for s in suggestions], indent=2))
        return

    if not suggestions:
        console.print("[dim]No link suggestions found[/dim]")
        return

    table = Table(title="Link Suggestions")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="yellow")
    table.add_column("Similarity", style="green")
    table.add_column("Reason", style="dim")

    for s in suggestions:
        table.add_row(
            s.memory_id[:8],
            s.title[:40],
            f"{s.similarity:.3f}",
            s.reason,
        )

    console.print(table)


@links_group.command(name="format")
@click.argument("memory_id")
@click.option("--title", help="Display title for the link")
def links_format(memory_id: str, title: str | None) -> None:
    """Format a memory ID as a wikilink.

    Example: memo links format abc123 --title "My Memory"
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    wikilink = mem.link_suggester.format_wikilink(memory_id, title)
    click.echo(wikilink)


@links_group.command(name="reindex")
@click.confirmation_option(prompt="This will rebuild the entire cross-reference index. Continue?")
def links_reindex() -> None:
    """Rebuild the cross-reference index from all memories.

    Example: memo links reindex
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    # Clear existing index (truncate, not unlink — the crossref tables may
    # share the main DB file under single_db, where unlinking would nuke it).
    mem.crossref.reset()

    # Re-index all memories
    all_records = mem.list(limit=10000)
    indexed = 0

    for rec in all_records:
        body = rec.body or ""
        if body:
            # index_source (delete-then-insert, incl. typed `- rel [[target]]`
            # edges), matching save/update/reindex — index_wikilinks is
            # append-only and drops typed edges, leaving stale rows.
            mem.crossref.index_source(rec.id, body)
            indexed += 1

    console.print(f"[green]Reindexed {indexed} memories[/green]")
