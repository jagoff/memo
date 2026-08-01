"""`memo coordinate` command group — live cross-agent collision scan.

Registered onto the root group in cli.py via `cli.add_command(coordinate_group)`.
Spec: docs/SPECS/2026-07-31-cross-agent-coordination-design.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import click
from rich.table import Table

from memo.cli_common import _short, console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config
from memo.coordination import Collision, CoordinationStore, coordination_db_path, scan_collisions


@click.group(name="coordinate")
def coordinate_group() -> None:
    """Detect live collisions between active agent sessions.

    `scan` runs one gather → candidate → LLM-judge pass now and stores
    confirmed collisions in a sidecar DB. `status` lists open/delivered
    collisions. `resolve` closes one once the conflict no longer applies.
    Directives are auto-delivered into each agent's next turn by the
    recall hook (`<memo-coordination>` block).
    """


def _collision_table(rows: list[Collision], *, title: str) -> Table:
    table = Table(title=title)
    table.add_column("id")
    table.add_column("sev")
    table.add_column("kind")
    table.add_column("resource")
    table.add_column("a", justify="right")
    table.add_column("b", justify="right")
    table.add_column("status")
    table.add_column("directive a / b")
    for row in rows:
        table.add_row(
            row.id,
            row.severity,
            row.kind,
            _short(row.resource, 40),
            row.session_a[:8],
            row.session_b[:8],
            row.status,
            _short(f"{row.directive_a} / {row.directive_b}", 70),
        )
    return table


@coordinate_group.command(name="scan")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def coordinate_scan(as_json: bool) -> None:
    """Run one collision scan now and print confirmed collisions.

    Example: memo coordinate scan
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)
    result = scan_collisions(mem, cfg)
    payload = asdict(result)
    if as_json:
        click.echo(json.dumps(payload, indent=2))
        return
    table = Table(title="Coordination scan")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key, value in payload.items():
        table.add_row(key.replace("_", " "), str(value))
    console.print(table)
    with CoordinationStore(coordination_db_path(cfg)) as store:
        rows = store.list_collisions(statuses=("open", "delivered"))
    if rows:
        console.print(_collision_table(rows, title="Active collisions"))


@coordinate_group.command(name="status")
@click.option("--limit", type=int, default=20, help="Max rows (default: 20)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def coordinate_status(limit: int, as_json: bool) -> None:
    """List open/delivered collisions from the sidecar DB."""
    cfg = Config.from_env()
    with CoordinationStore(coordination_db_path(cfg)) as store:
        rows = store.list_collisions(statuses=("open", "delivered"), limit=limit)
    if as_json:
        click.echo(json.dumps([asdict(r) for r in rows], indent=2))
        return
    if not rows:
        console.print("[green]No active collisions.[/green]")
        return
    console.print(_collision_table(rows, title="Active collisions"))


@coordinate_group.command(name="resolve")
@click.argument("collision_id")
def coordinate_resolve(collision_id: str) -> None:
    """Mark a collision as resolved (either agent or the user)."""
    cfg = Config.from_env()
    with CoordinationStore(coordination_db_path(cfg)) as store:
        resolved = store.resolve(collision_id)
    if resolved:
        console.print(f"[green]collision {collision_id} resolved.[/green]")
    else:
        console.print(f"[red]collision {collision_id} not found.[/red]")
