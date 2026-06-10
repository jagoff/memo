"""`memo sync` command group — multi-machine sync via audit-log replay.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(sync_group)`.

The sync model is pull-only: a machine replays the events missing from its
local store that exist in a remote `history.db`. There is no file diff and no
push (the remote machine pulls instead).
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config


def _resolve_remote_history_db(remote: str | None) -> Path | None:
    """Map a ``--remote`` arg to the remote machine's ``history.db``.

    Accepts either a direct path to a ``.db`` file or a memo state dir that
    contains ``history.db``.
    """
    if not remote:
        return None
    p = Path(remote)
    return p if p.suffix == ".db" else p / "history.db"


@click.group(name="sync")
def sync_group() -> None:
    """Multi-machine sync — replay a remote machine's audit log locally."""
    pass


@sync_group.command(name="diff")
@click.option("--remote", help="Path to remote memo state dir")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def sync_diff(remote: str | None, as_json: bool) -> None:
    """Not supported in the replay sync model (no precomputed diff).

    Use `memo sync pull` to apply missing remote events.
    """
    msg = "replay sync model has no precomputed diff; use `memo sync pull`"
    if as_json:
        click.echo(json.dumps({"error": msg}, indent=2))
        return
    console.print(f"[yellow]{msg}[/yellow]")


@sync_group.command(name="push")
@click.option("--remote", help="Path to remote memo state dir")
def sync_push(remote: str | None) -> None:
    """Not supported in the replay sync model (pull-only).

    Sync is pull-only: the remote machine pulls from this one instead.
    """
    console.print(
        "[yellow]replay sync model is pull-only; the remote machine pulls instead[/yellow]"
    )


@sync_group.command(name="pull")
@click.option("--remote", required=True, help="Path to remote memo state dir")
def sync_pull(remote: str) -> None:
    """Pull remote changes by replaying the remote audit log.

    Example: memo sync pull --remote /path/to/remote/memo
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    remote_db = _resolve_remote_history_db(remote)
    assert remote_db is not None  # --remote is required
    diff = mem.sync.sync_from_remote(remote_db)

    console.print("[bold]Pull Sync[/bold]")
    console.print(f"Applied: {diff.applied}")
    console.print(f"Conflicts: {diff.conflicts}")
    console.print(f"Errors: {diff.errors}")


@sync_group.command(name="both")
@click.option("--remote", required=True, help="Path to remote memo state dir")
def sync_both(remote: str) -> None:
    """Sync from a remote machine (replay model alias for pull).

    In the replay model "both directions" is achieved by each machine pulling
    the other's audit log; from this side that is a pull.

    Example: memo sync both --remote /path/to/remote/memo
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    remote_db = _resolve_remote_history_db(remote)
    assert remote_db is not None  # --remote is required
    diff = mem.sync.sync_from_remote(remote_db)

    console.print("[bold]Sync (replay)[/bold]")
    console.print(f"Applied: {diff.applied}")
    console.print(f"Conflicts: {diff.conflicts}")
    console.print(f"Errors: {diff.errors}")
