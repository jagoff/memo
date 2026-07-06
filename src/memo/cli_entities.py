"""Knowledge-graph verbs for the memo CLI — extract-entities / entities / entity.

Extracted from cli_memory.py (god-module decomposition). Registered onto the
root group in cli.py.
"""

from __future__ import annotations

import json
import sys

import click
from rich.table import Table

from memo.cli_common import _resolved, console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config


@click.command(name="extract-entities")
@click.option("--all", "all_", is_flag=True, help="Process every memory in the store.")
@click.option(
    "--id",
    "id_",
    default=None,
    multiple=True,
    help="Repeatable. Process specific memory id(s) (full or prefix).",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-extract even if memory already has entity links (default skips).",
)
@click.option("--json", "as_json", is_flag=True)
def extract_entities(all_: bool, id_: tuple[str, ...], force: bool, as_json: bool) -> None:
    """Extract named entities (person/project/technology/file/org/concept)
    from memory bodies via the configured helper LLM and write them to the graph DB.

    Cost: ~0.5-1s per memory. 223-doc corpus ≈ 2-4 min.
    """

    if not all_ and not id_:
        click.echo("pass --all or one or more --id <prefix>", err=True)
        sys.exit(2)

    mem = _get_memory(Config.from_env())
    resolved_ids: list[str] | None = None
    if id_:
        resolved_ids = []
        for raw in id_:
            r = _resolved(lambda raw=raw: mem.resolve_id(raw))
            if r is None:
                console.print(f"[red]not found:[/red] {raw}")
                sys.exit(1)
            resolved_ids.append(r)

    counts = mem.extract_entities(
        ids=resolved_ids,
        all_=all_,
        skip_already_indexed=not force,
    )
    if as_json:
        click.echo(json.dumps(counts, indent=2))
        return
    console.print(
        f"processed: [cyan]{counts['processed']}[/cyan]  "
        f"entities: [green]{counts['entities_extracted']}[/green]  "
        f"links: [green]{counts['links_written']}[/green]  "
        f"skipped: [dim]{counts['skipped']}[/dim]  "
        f"errors: [red]{counts['errors']}[/red]",
    )


@click.command()
@click.option("--limit", default=30, type=int, show_default=True)
@click.option(
    "--type",
    "type_",
    default=None,
    type=click.Choice(["person", "project", "technology", "file", "org", "concept"]),
    help="Filter by entity type.",
)
@click.option("--json", "as_json", is_flag=True)
def entities(limit: int, type_: str | None, as_json: bool) -> None:
    """Top entities by mention count."""

    mem = _get_memory(Config.from_env())
    rows = mem.graph.top_entities(limit=limit, type_=type_)
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    if not rows:
        console.print("[dim]no entities indexed — run `memo extract-entities --all` first[/dim]")
        return
    tbl = Table(show_lines=False, expand=True)
    tbl.add_column("count", justify="right", width=6)
    tbl.add_column("type", width=12)
    tbl.add_column("name", overflow="fold")
    tbl.add_column("first_seen", width=10)
    tbl.add_column("last_seen", width=10)
    for r in rows:
        tbl.add_row(
            str(r["mention_count"]),
            r["type"],
            r["name"],
            (r["first_seen"] or "")[:10],
            (r["last_seen"] or "")[:10],
        )
    console.print(tbl)


@click.command()
@click.argument("name")
@click.option(
    "--type",
    "type_",
    default=None,
    type=click.Choice(["person", "project", "technology", "file", "org", "concept"]),
)
@click.option("--json", "as_json", is_flag=True)
def entity(name: str, type_: str | None, as_json: bool) -> None:
    """Memories that mention an entity."""

    mem = _get_memory(Config.from_env())
    ids = mem.graph.entity_memories(name, type_=type_)
    if as_json:
        click.echo(json.dumps(ids, indent=2))
        return
    if not ids:
        console.print(f"[dim]no memories mention {name!r}{f' ({type_})' if type_ else ''}[/dim]")
        return
    console.print(f"[bold]{len(ids)}[/bold] memory(s) mention [cyan]{name}[/cyan]:")
    for mid in ids[:50]:
        rec = mem.store.get(mid)
        if rec:
            console.print(f"  · [{mid[:8]}] {rec['title'][:60]} [dim]({rec['updated'][:10]})[/dim]")
    if len(ids) > 50:
        console.print(f"  · …and {len(ids) - 50} more")
