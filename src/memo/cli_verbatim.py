"""`memo verbatim` — lexical turn-level transcript index (Total Recall F1).

The index is explicit and opt-in; search is a pure FTS5 lookup and never
constructs a `Memory` or enters automatic recall.
"""

from __future__ import annotations

import json
from typing import Any

import click

from memo.cli_common import console, log_cli_consult
from memo.config import Config


@click.group(name="verbatim")
def verbatim_group() -> None:
    """Lexical (FTS5, no embeddings) turn-level index over transcript JSONL."""


@verbatim_group.command(name="index")
@click.option("--dry-run", is_flag=True, help="Report what would be indexed without writing.")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def index_cmd(dry_run: bool, as_json: bool) -> None:
    """Incrementally index transcript turns into the verbatim FTS5 store."""
    from memo.verbatim_index import run_verbatim_index_pass

    cfg = Config.from_env()
    result = run_verbatim_index_pass(cfg, dry_run=dry_run)

    if as_json:
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))
        return

    console.print(f"[bold]verbatim index:[/bold] {result.get('status')}")
    console.print(f"  sessions indexed: {result.get('sessions_indexed', 0)}")
    console.print(f"  turns indexed:    {result.get('turns_indexed', 0)}")
    console.print(f"  skipped unchanged: {result.get('skipped_unchanged', 0)}")
    console.print(f"  pruned:           {result.get('pruned', 0)}")
    if result.get("status") == "error":
        console.print(f"  [red]error:[/red] {result.get('error')}")


@verbatim_group.command(name="search")
@click.argument("query")
@click.option("--session", "session_id", default=None, help="Restrict to one session id.")
@click.option("--since", default=None, help="ISO8601 lower bound on turn timestamp.")
@click.option("--limit", default=10, type=click.IntRange(1, 100), show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
@click.option(
    "--source",
    default=None,
    help="Attribute this consult in `memo usefulness` (falls back to MEMO_SOURCE).",
)
def search_cmd(
    query: str,
    session_id: str | None,
    since: str | None,
    limit: int,
    as_json: bool,
    source: str | None,
) -> None:
    """Lexical search over explicitly indexed transcript turns."""
    import time

    from memo.store.turn_store import TurnStore

    cfg = Config.from_env()
    t0 = int(time.time() * 1000)
    hits: list[dict[str, Any]] = []
    if cfg.verbatim_db.is_file():
        store = TurnStore(cfg.verbatim_db)
        try:
            hits = store.search(query, limit=limit, session_id=session_id, since=since)
        finally:
            store.close()
    log_cli_consult(cfg, verb="verbatim_search", query=query, hits=hits, t0_ms=t0, source=source)

    if as_json:
        click.echo(json.dumps(hits, ensure_ascii=False, indent=2))
        return
    if not hits:
        console.print("[dim]no results[/dim]")
        return
    for hit in hits:
        console.print(
            f"[dim][{str(hit.get('session_id') or '')[:8]}][/dim] "
            f"turn {hit.get('turn_idx')} [bold]{hit.get('role')}[/bold] "
            f"[dim]{hit.get('ts') or ''}[/dim]  {hit.get('snippet') or ''}"
        )


@verbatim_group.command(name="status")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def status_cmd(as_json: bool) -> None:
    """Report private index statistics and watermark metadata."""
    import time

    from memo.store.turn_store import TurnStore
    from memo.verbatim_index import _WATERMARK_FILE

    cfg = Config.from_env()
    stats = {"sessions": 0, "turns": 0}
    if cfg.verbatim_db.is_file():
        store = TurnStore(cfg.verbatim_db)
        try:
            stats = store.stats()
        finally:
            store.close()

    watermark_path = cfg.state_dir / _WATERMARK_FILE
    watermark: dict[str, Any] = {"exists": False}
    if watermark_path.is_file():
        metadata = watermark_path.stat()
        watermark = {
            "exists": True,
            "path": str(watermark_path),
            "size_bytes": metadata.st_size,
            "age_seconds": max(0, int(time.time() - metadata.st_mtime)),
        }
    result = {**stats, "db_path": str(cfg.verbatim_db), "watermark": watermark}

    if as_json:
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))
        return
    console.print("[bold]verbatim status:[/bold]")
    console.print(f"  sessions: {stats['sessions']}")
    console.print(f"  turns:    {stats['turns']}")
    console.print(f"  db path:  {cfg.verbatim_db}")
    if watermark.get("exists"):
        console.print(
            f"  watermark: {watermark['path']} "
            f"({watermark['size_bytes']}b, {watermark['age_seconds']}s old)"
        )
    else:
        console.print("  watermark: [dim](none — index never ran)[/dim]")


__all__ = ["verbatim_group"]
