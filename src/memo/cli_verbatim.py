"""`memo verbatim` — lexical turn-level transcript index (Total Recall F1).

Only the `index` subcommand ships here (Task V3). `search`/`status` land in
Task V4 in this same module.
"""

from __future__ import annotations

import json

import click

from memo.cli_common import console
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


__all__ = ["verbatim_group"]
