"""`memo version` command group — memoria version history / diff / rollback.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(version_group)`.
"""

from __future__ import annotations

import json

import click
from rich.table import Table

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config

# -- versioning commands ------------------------------------------------------


@click.group(name="version")
def version_group() -> None:
    """Memory versioning — track changes, visualize diffs, rollback."""
    pass


@version_group.command(name="history")
@click.argument("memoria_id")
@click.option("--limit", type=int, default=10,
              help="Max versions to show (default: 10)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def version_history(memoria_id: str, limit: int, as_json: bool) -> None:
    """Show version history for a memoria.

    Example: memo version history abc123
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    versions = mem.versioning.get_version_history(memoria_id, limit=limit)

    if as_json:
        click.echo(json.dumps([v.__dict__ for v in versions], indent=2))
        return

    if not versions:
        console.print(f"[dim]No version history for memoria {memoria_id[:8]}[/dim]")
        return

    console.print(f"[bold]Version History for {memoria_id[:8]}[/bold]")
    console.print()

    table = Table()
    table.add_column("Version ID", style="cyan")
    table.add_column("Timestamp", style="dim")
    table.add_column("Title", style="yellow")
    table.add_column("Type", style="green")
    table.add_column("Reason", style="magenta")

    for v in versions[:20]:
        table.add_row(
            str(v.version_id),
            v.timestamp[:19],
            v.title[:40],
            v.type,
            v.reason or "—",
        )

    console.print(table)
    if len(versions) > 20:
        console.print(f"[dim]...and {len(versions) - 20} more[/dim]")


@version_group.command(name="diff")
@click.argument("memoria_id")
@click.option("--version-a", type=int, help="First version ID (default: latest)")
@click.option("--version-b", type=int, help="Second version ID (default: latest-1)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def version_diff(memoria_id: str, version_a: int | None, version_b: int | None, as_json: bool) -> None:
    """Show diff between two versions of a memoria.

    Example: memo version diff abc123 --version-a 1 --version-b 2
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    diff = mem.versioning.diff_versions(memoria_id, version_a, version_b)

    if as_json:
        click.echo(json.dumps(diff.__dict__ if diff else None, indent=2))
        return

    if diff is None:
        console.print("[yellow]Could not generate diff[/yellow]")
        return

    console.print(f"[bold]Diff for {memoria_id[:8]}[/bold]")
    console.print(f"[dim]v{diff.version_a} → v{diff.version_b}[/dim]")
    console.print()
    console.print(diff.unified_diff)


@version_group.command(name="rollback")
@click.argument("memoria_id")
@click.argument("version_id", type=int)
@click.option("--reason", help="Reason for the rollback")
@click.confirmation_option(prompt="This will restore the memoria to the specified version. Continue?")
def version_rollback(memoria_id: str, version_id: int, reason: str | None) -> None:
    """Rollback a memoria to a previous version.

    Example: memo version rollback abc123 1 --reason "Mistake in update"
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    success = mem.versioning.rollback_to_version(memoria_id, version_id, reason)

    if success:
        console.print(f"[green]Rolled back {memoria_id[:8]} to version {version_id}[/green]")
    else:
        console.print("[red]Failed to rollback[/red]")
