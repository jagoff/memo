"""`memo feedback` command group — relevance feedback capture.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(feedback_group)`.
"""

from __future__ import annotations

import json

import click

from memo.cli_common import console
from memo.config import Config


@click.group(name="feedback")
def feedback_group() -> None:
    """Per-source 👍/👎 feedback used to teach the retriever which
    memories to surface (or hide) for queries similar to the one being
    voted on."""


@feedback_group.command(name="record")
@click.argument("source_id")
@click.option("--query", "query_text", required=True, help="Query text the feedback applies to.")
@click.option(
    "--rating",
    required=True,
    type=click.Choice(["up", "down"]),
    help="up = boost, down = exclude for similar queries.",
)
@click.option("--as-json", is_flag=True)
def feedback_record_cmd(source_id: str, query_text: str, rating: str, as_json: bool) -> None:
    """Record a 👍/👎 vote on SOURCE_ID for QUERY. Embeds QUERY so future
    semantically-similar queries inherit the vote.

    SOURCE_ID may be a full meta.id or a unique prefix (>= 4 chars)."""
    from memo.memory import Memory

    cfg = Config.from_env()
    mem = Memory(cfg)
    try:
        rid = mem.feedback_record(source_id, query_text=query_text, rating=rating)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc
    finally:
        mem.close()
    if as_json:
        click.echo(json.dumps(rid, ensure_ascii=False))
        return
    console.print(
        f"[green]ok[/green] feedback_id={rid['feedback_id']} "
        f"source={rid['source_id'][:8]} rating={rid['rating']}"
    )


@feedback_group.command(name="flag")
@click.argument("source_id")
@click.option(
    "--kind",
    required=True,
    type=click.Choice(["outdated", "wrong"]),
    help="outdated = archive stale memory; wrong = archive (+ supersede if a replacement is given).",
)
@click.option(
    "--superseded-by",
    "superseded_by",
    default=None,
    help="For --kind wrong: id (prefix ok) of the memory that replaces this one.",
)
@click.option("--as-json", is_flag=True)
def feedback_flag_cmd(source_id: str, kind: str, superseded_by: str | None, as_json: bool) -> None:
    """Flag SOURCE_ID as outdated/wrong → route to the lifecycle (reversible
    archive), not the retriever. Use `memo feedback record` for 👍/👎 ranking
    votes instead.

    SOURCE_ID may be a full meta.id or a unique prefix."""
    from memo.memory import Memory

    cfg = Config.from_env()
    mem = Memory(cfg)
    try:
        res = mem.feedback_flag(source_id, kind=kind, superseded_by=superseded_by)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc
    finally:
        mem.close()
    if as_json:
        click.echo(json.dumps(res, ensure_ascii=False))
        return
    tail = f" → {res['superseded_by'][:8]}" if res.get("superseded_by") else ""
    console.print(
        f"[green]ok[/green] {res['action']} {res['source_id'][:8]} (kind={res['kind']}){tail}"
    )


@feedback_group.command(name="list")
@click.option("--source", "source_id", default=None, help="Filter by source id (prefix ok).")
@click.option("--limit", default=50, type=int)
@click.option("--as-json", is_flag=True)
def feedback_list_cmd(source_id: str | None, limit: int, as_json: bool) -> None:
    """List recorded feedback rows, newest first."""
    from memo.memory import Memory

    cfg = Config.from_env()
    mem = Memory(cfg)
    try:
        rows = mem.feedback_list(source_id=source_id, limit=limit)
    finally:
        mem.close()
    if as_json:
        click.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    if not rows:
        console.print("[dim]no feedback recorded[/dim]")
        return
    for r in rows:
        rating_disp = "👍" if int(r["rating"]) > 0 else "👎"
        console.print(
            f"{rating_disp} [cyan]{r['source_id'][:8]}[/cyan] "
            f"[dim]{r['created_at']}[/dim]  q={r['query_text']!r}"
        )


@feedback_group.command(name="clear")
@click.argument("source_id")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt.")
def feedback_clear_cmd(source_id: str, yes: bool) -> None:
    """Drop ALL feedback rows for SOURCE_ID. Cannot be undone."""
    if not yes and not click.confirm(f"Drop all feedback for {source_id}?"):
        return
    from memo.memory import Memory

    cfg = Config.from_env()
    mem = Memory(cfg)
    try:
        n = mem.feedback_clear(source_id)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc
    finally:
        mem.close()
    console.print(f"[green]ok[/green] deleted {n} row(s)")
