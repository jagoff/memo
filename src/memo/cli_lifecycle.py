"""`memo lifecycle` command group — archival / promotion / expiration.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(lifecycle_group)`.
"""

from __future__ import annotations

import json

import click
from rich.table import Table

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config

# -- lifecycle management commands ---------------------------------------------


@click.group(name="lifecycle")
def lifecycle_group() -> None:
    """Memory lifecycle management — archival, promotion, expiration."""
    pass


@lifecycle_group.command(name="report")
@click.option("--limit", type=int, default=100,
              help="Max memorias to analyze (default: 100)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def lifecycle_report(limit: int, as_json: bool) -> None:
    """Generate a lifecycle report on the corpus.

    Shows statistics on archival candidates, promotion/demotion candidates,
    expiration candidates, and access patterns.

    Example: memo lifecycle report --limit 50
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    report = mem.lifecycle.get_lifecycle_report(limit=limit)

    if as_json:
        click.echo(json.dumps(report, indent=2))
        return

    console.print("[bold]Lifecycle Report[/bold]")
    console.print()
    console.print(f"Total memorias: {report['total']}")
    console.print(f"Average access count: {report['avg_access_count']:.2f}")
    console.print()
    console.print(f"[yellow]Archive candidates:[/yellow] {report['archive_candidates']}")
    console.print(f"[yellow]Promotion candidates:[/yellow] {report['promotion_candidates']}")
    console.print(f"[yellow]Demotion candidates:[/yellow] {report['demotion_candidates']}")
    console.print(f"[yellow]Expiration candidates:[/yellow] {report['expiration_candidates']}")
    console.print(f"[yellow]Never accessed:[/yellow] {report['never_accessed']}")


@lifecycle_group.command(name="apply")
@click.option("--dry-run", is_flag=True,
              help="Show what would happen without applying changes")
@click.option("--limit", type=int, default=100,
              help="Max memorias to process (default: 100)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
@click.confirmation_option(prompt="This will archive/delete memorias based on lifecycle rules. Continue?")
def lifecycle_apply(dry_run: bool, limit: int, as_json: bool) -> None:
    """Apply lifecycle rules to the corpus.

    Archives inactive memorias, expires temporary memories, and reports
    promotion/demotion candidates. Use --dry-run first to preview.

    Example: memo lifecycle apply --dry-run
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    if dry_run:
        console.print("[yellow]Dry run mode - no changes will be applied[/yellow]")
        console.print()

    actions = mem.lifecycle.apply_lifecycle_rules(dry_run=dry_run, limit=limit)

    if as_json:
        click.echo(json.dumps(actions, indent=2))
        return

    console.print("[bold]Lifecycle Actions[/bold]")
    console.print()
    console.print(f"[green]Archived:[/green] {actions['archived']}")
    console.print(f"[green]Promoted:[/green] {actions['promoted']}")
    console.print(f"[yellow]Demoted:[/yellow] {actions['demoted']}")
    console.print(f"[red]Expired:[/red] {actions['expired']}")
    console.print(f"[red]Deleted:[/red] {actions['deleted']}")
    console.print(f"[dim]Skipped:[/dim] {actions['skipped']}")


@lifecycle_group.command(name="access-count")
@click.argument("memoria_id")
def lifecycle_access_count(memoria_id: str) -> None:
    """Show access count for a specific memoria.

    Example: memo lifecycle access-count abc123
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    count = mem.lifecycle.get_access_count(memoria_id)
    console.print(f"Access count: {count}")


@lifecycle_group.command(name="list-inactive")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def lifecycle_list_inactive(as_json: bool) -> None:
    """List all archived/inactive memorias.

    Example: memo lifecycle list-inactive
    """
    cfg = Config.from_env()
    inactive_dir = cfg.memory_dir / "inactive"

    if not inactive_dir.is_dir():
        console.print("[dim]No inactive memorias found[/dim]")
        return

    files = list(inactive_dir.glob("*.md"))

    if as_json:
        inactive_data = []
        for f in files:
            inactive_data.append({
                "id": f.stem,
                "path": str(f),
            })
        click.echo(json.dumps(inactive_data, indent=2))
        return

    table = Table(title="Inactive Memorias")
    table.add_column("ID", style="cyan")
    table.add_column("Path", style="dim")

    for f in files[:50]:
        table.add_row(f.stem[:8], str(f.name))

    console.print(table)
    if len(files) > 50:
        console.print(f"[dim]...and {len(files) - 50} more[/dim]")
