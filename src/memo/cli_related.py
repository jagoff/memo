"""`memo related` — on-demand associative recall from the CLI."""

from __future__ import annotations

import json

import click

from memo.cli_common import get_memory as _get_memory
from memo.config import Config
from memo.server_related import related_for


@click.command(name="related")
@click.argument("query_or_id")
@click.option("--hops", type=int, default=2, help="Graph hops to expand (default 2).")
@click.option("--limit", type=int, default=5, help="Max related memories (default 5).")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def related(query_or_id: str, hops: int, limit: int, as_json: bool) -> None:
    """Memories structurally connected to a memory id or query, via the graph."""
    mem = _get_memory(Config.from_env())
    hits = related_for(mem, query_or_id, hops=hops, limit=limit)
    if as_json:
        click.echo(json.dumps(hits, indent=2))
        return
    if not hits:
        click.echo("No related memories found.")
        return
    for h in hits:
        click.echo(f"[{h['id'][:8]}] {h['title']}  — via {h['via']} ({h['activation']})")
