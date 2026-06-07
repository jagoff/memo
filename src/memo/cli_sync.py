"""`memo sync` command group — vault sync (diff/push/pull/both).

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(sync_group)`.
"""

from __future__ import annotations

import json

import click

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config


@click.group(name="sync")
def sync_group() -> None:
    """Multi-vault sync — sync between vaults."""
    pass


@sync_group.command(name="diff")
@click.option("--remote", help="Path to remote vault")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def sync_diff(remote: str | None, as_json: bool) -> None:
    """Compute diff between local and remote vaults.

    Example: memo sync diff --remote /path/to/remote/memo
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path

    from memo.sync import SyncManager

    remote_path = Path(remote) if remote else None

    sync_mgr = SyncManager(mem, remote_path=remote_path)
    diff = sync_mgr.compute_diff()

    if as_json:
        click.echo(json.dumps(diff.__dict__, indent=2))
        return

    console.print("[bold]Sync Diff[/bold]")
    console.print()
    console.print(f"New: {len(diff.new)}")
    console.print(f"Modified: {len(diff.modified)}")
    console.print(f"Deleted: {len(diff.deleted)}")
    console.print(f"Conflicts: {len(diff.conflicts)}")


@sync_group.command(name="push")
@click.option("--remote", help="Path to remote vault")
def sync_push(remote: str | None) -> None:
    """Push local changes to remote vault.

    Example: memo sync push --remote /path/to/remote/memo
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path

    from memo.sync import SyncManager

    remote_path = Path(remote) if remote else None

    sync_mgr = SyncManager(mem, remote_path=remote_path)
    diff = sync_mgr.sync(direction="push")

    console.print("[bold]Push Sync[/bold]")
    console.print(f"Modified: {len(diff.modified)}")
    console.print(f"Deleted: {len(diff.deleted)}")


@sync_group.command(name="pull")
@click.option("--remote", help="Path to remote vault")
def sync_pull(remote: str | None) -> None:
    """Pull remote changes to local vault.

    Example: memo sync pull --remote /path/to/remote/memo
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path

    from memo.sync import SyncManager

    remote_path = Path(remote) if remote else None

    sync_mgr = SyncManager(mem, remote_path=remote_path)
    diff = sync_mgr.sync(direction="pull")

    console.print("[bold]Pull Sync[/bold]")
    console.print(f"New: {len(diff.new)}")
    console.print(f"Modified: {len(diff.modified)}")


@sync_group.command(name="both")
@click.option("--remote", help="Path to remote vault")
def sync_both(remote: str | None) -> None:
    """Sync both directions (bidirectional).

    Example: memo sync both --remote /path/to/remote/memo
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path

    from memo.sync import SyncManager

    remote_path = Path(remote) if remote else None

    sync_mgr = SyncManager(mem, remote_path=remote_path)
    diff = sync_mgr.sync(direction="both")

    console.print("[bold]Bidirectional Sync[/bold]")
    console.print(f"New: {len(diff.new)}")
    console.print(f"Modified: {len(diff.modified)}")
    console.print(f"Deleted: {len(diff.deleted)}")
    console.print(f"Conflicts: {len(diff.conflicts)}")
