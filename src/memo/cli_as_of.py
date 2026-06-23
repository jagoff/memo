"""`memo as-of` command group — time-travel search/ask/list.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(as_of_group)`.
"""

from __future__ import annotations

import json

import click
from rich.panel import Panel
from rich.table import Table

from memo.cli_common import _parse_as_of_date, console
from memo.config import Config


@click.group(name="as-of")
def as_of_group() -> None:
    """Time-machine — query the corpus as it existed at any past date.

    Subcommands: `search`, `ask`, `list`. All take `--date YYYY-MM-DD`
    (or a full ISO timestamp). The snapshot is reconstructed by
    replaying `history.db` events in reverse from "now".
    """
    pass


@as_of_group.command(name="search")
@click.argument("query")
@click.option("--date", "as_of", required=True, help="YYYY-MM-DD or full ISO 8601.")
@click.option("--limit", default=10, type=int, show_default=True)
@click.option("--type", "type_", default=None, help="Filter by record type.")
@click.option(
    "--mode", default="hybrid", type=click.Choice(["hybrid", "vec", "bm25"]), show_default=True
)
@click.option("--json", "as_json", is_flag=True)
def as_of_search(
    query: str,
    as_of: str,
    limit: int,
    type_: str | None,
    mode: str,
    as_json: bool,
) -> None:
    """Search the corpus as it existed on a past date."""
    from memo.memory import Memory
    from memo.time_machine import reconstruct

    mem = Memory(Config.from_env())
    snap = reconstruct(mem, as_of=_parse_as_of_date(as_of))
    hits = snap.search(query, limit=limit, mode=mode)
    if type_:
        hits = [h for h in hits if h.type == type_]

    if as_json:
        click.echo(
            json.dumps(
                {
                    "as_of": snap.as_of.isoformat(),
                    "snapshot_size": len(snap),
                    "results": [h.to_dict() for h in hits],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if not hits:
        console.print(f"[dim]no results in snapshot @ {snap.as_of.date().isoformat()}[/dim]")
        return
    tbl = Table(
        show_lines=False,
        expand=True,
        title=f"snapshot @ {snap.as_of.date().isoformat()} · {len(snap)} memorias existed",
    )
    tbl.add_column("score", justify="right", width=6)
    tbl.add_column("type", width=10)
    tbl.add_column("title", overflow="fold")
    tbl.add_column("tags", overflow="fold")
    for h in hits:
        tbl.add_row(
            f"{h.score:.3f}" if h.score is not None else "—",
            h.type,
            h.title,
            ", ".join(h.tags) or "—",
        )
    console.print(tbl)


@as_of_group.command(name="ask")
@click.argument("question")
@click.option("--date", "as_of", required=True, help="YYYY-MM-DD or full ISO 8601.")
@click.option("--k", default=5, type=int, show_default=True)
@click.option("--json", "as_json", is_flag=True)
def as_of_ask(question: str, as_of: str, k: int, as_json: bool) -> None:
    """RAG question against a past snapshot of the corpus."""
    from memo.memory import Memory
    from memo.time_machine import reconstruct

    mem = Memory(Config.from_env())
    snap = reconstruct(mem, as_of=_parse_as_of_date(as_of))
    out = snap.ask(question, k=k)

    if as_json:
        click.echo(json.dumps(out, ensure_ascii=False, indent=2))
        return

    console.print(
        Panel.fit(
            out["answer"] or "[dim](no answer)[/dim]",
            title=f"✓ as-of {snap.as_of.date().isoformat()} ({len(snap)} memorias in scope)",
            border_style="magenta",
        )
    )
    if out.get("sources"):
        console.print("\n[dim]sources:[/dim]")
        for s in out["sources"]:
            console.print(f"  [bold]{s['id_short']}[/bold]  {s['title']}  [dim]({s['type']})[/dim]")


@as_of_group.command(name="list")
@click.option("--date", "as_of", required=True, help="YYYY-MM-DD or full ISO 8601.")
@click.option("--type", "type_", default=None)
@click.option("--limit", default=20, type=int, show_default=True)
@click.option("--json", "as_json", is_flag=True)
def as_of_list(as_of: str, type_: str | None, limit: int, as_json: bool) -> None:
    """List memorias that existed in a past snapshot (most-recent first)."""
    from memo.memory import Memory
    from memo.time_machine import reconstruct

    mem = Memory(Config.from_env())
    snap = reconstruct(mem, as_of=_parse_as_of_date(as_of))
    rows = snap.list(type_=type_)[:limit]

    if as_json:
        click.echo(
            json.dumps(
                {
                    "as_of": snap.as_of.isoformat(),
                    "snapshot_size": len(snap),
                    "records": [
                        {
                            "id": r.id,
                            "title": r.title,
                            "type": r.type,
                            "tags": r.tags,
                            "updated": r.updated,
                        }
                        for r in rows
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if not rows:
        console.print(f"[dim]empty snapshot @ {snap.as_of.date().isoformat()}[/dim]")
        return
    tbl = Table(
        show_lines=False,
        expand=True,
        title=f"snapshot @ {snap.as_of.date().isoformat()} · {len(snap)} memorias",
    )
    tbl.add_column("id", width=10)
    tbl.add_column("type", width=10)
    tbl.add_column("title", overflow="fold")
    tbl.add_column("updated", width=12)
    for r in rows:
        tbl.add_row(r.id[:8], r.type, r.title, (r.updated or "—")[:10])
    console.print(tbl)
