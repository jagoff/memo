"""`memo config` command group — inspect + validate feature flags.

Surfaces the central `flags.py` registry: list every documented `MEMO_*`
flag with its type/default/group, show which are currently active, and
validate the environment for misconfigured or unknown vars.

Registered onto the root group in cli.py via `cli.add_command(config_group)`.
"""

from __future__ import annotations

import json

import click
from rich.table import Table

from memo import flags
from memo.cli_common import console


@click.group(name="config")
def config_group() -> None:
    """Inspect + validate memo's MEMO_* configuration flags."""
    pass


@config_group.command(name="flags")
@click.option("--group", "group_filter", default=None, help="Filter by subsystem group.")
@click.option("--active", is_flag=True, help="Only flags currently set in the environment.")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def config_flags(group_filter: str | None, active: bool, as_json: bool) -> None:
    """List documented MEMO_* flags (type, default, group, active value).

    Example: memo config flags --group recall
    """
    active_vals = flags.active_flags()
    rows = []
    for name, spec in flags.REGISTRY.items():
        if group_filter and spec.group != group_filter:
            continue
        if active and name not in active_vals:
            continue
        rows.append(
            {
                "flag": name,
                "group": spec.group,
                "kind": spec.kind,
                "default": spec.default,
                "active": active_vals.get(name),
                "help": spec.help,
            }
        )

    if as_json:
        click.echo(json.dumps(rows, indent=2, default=str))
        return

    table = Table(title="MEMO_* flags")
    table.add_column("flag", style="cyan", no_wrap=True)
    table.add_column("group", style="magenta")
    table.add_column("kind")
    table.add_column("default", style="dim")
    table.add_column("active", style="green")
    for r in rows:
        table.add_row(
            r["flag"],
            r["group"],
            r["kind"],
            "" if r["default"] is None else str(r["default"]),
            "" if r["active"] is None else str(r["active"]),
        )
    console.print(table)
    console.print(f"[dim]{len(rows)} flag(s); {len(active_vals)} active in env[/dim]")


@config_group.command(name="validate")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def config_validate(as_json: bool) -> None:
    """Parse every set MEMO_* flag; report misconfigured or unknown vars.

    Exit code 1 if any problems are found. Example: memo config validate
    """
    problems = flags.validate()
    active_vals = flags.active_flags()

    if as_json:
        click.echo(json.dumps({"active": len(active_vals), "problems": problems}, indent=2))
    elif not problems:
        console.print(f"[green]✓[/green] {len(active_vals)} flag(s) set, all valid")
    else:
        console.print(f"[red]✗ {len(problems)} problem(s):[/red]")
        for p in problems:
            console.print(f"  [yellow]{p['flag']}[/yellow]={p['value']!r} — {p['error']}")

    if problems:
        raise SystemExit(1)
