"""`memo embed-daemon` command group — embedding daemon status.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(embed_daemon_group)`.
"""

from __future__ import annotations

import json
import sys

import click
from rich.table import Table

from memo.cli_common import console


@click.group(name="embed-daemon")
def embed_daemon_group() -> None:
    """Inspect the recall daemon's shared embedder sidecar.

    The recall daemon doubles as a shared MLX embedder so peers
    (synapse, memflow, other memo CLIs) can reuse one warm model.
    These commands surface its metrics without opening the socket
    by hand. See `memo.embedder_client` for the in-process client.
    """


@embed_daemon_group.command(name="status")
def embed_daemon_status() -> None:
    """Show whether the daemon is alive plus model + uptime."""
    from memo.embedder_client import status as _status

    info = _status()
    if info is None:
        console.print("[red]daemon: not running[/red]")
        sys.exit(1)
    uptime = info.get("uptime_s")
    uptime_str = f"{uptime}s" if isinstance(uptime, int) else "?"
    console.print(
        f"[green]daemon: running[/green] "
        f"model={info.get('model', '?')} dims={info.get('dims', '?')} "
        f"uptime={uptime_str}"
    )


@embed_daemon_group.command(name="stats")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def embed_daemon_stats(as_json: bool) -> None:
    """Per-op request counters + latency p50/p95/p99."""
    from memo.embedder_client import stats as _stats

    info = _stats()
    if info is None:
        console.print("[red]daemon: not running[/red]")
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(info, indent=2))
        return

    uptime = info.get("uptime_s")
    uptime_str = f"{uptime}s" if isinstance(uptime, int) else "?"
    console.print(
        f"[bold]Embedder daemon stats[/bold] "
        f"model={info.get('model', '?')} dims={info.get('dims', '?')} "
        f"uptime={uptime_str}"
    )
    ops = info.get("ops") or {}
    if not ops:
        console.print("[dim]no requests recorded yet[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("op")
    table.add_column("count", justify="right")
    table.add_column("errors", justify="right")
    table.add_column("samples", justify="right")
    table.add_column("p50 ms", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("p99 ms", justify="right")
    for op, row in sorted(ops.items()):
        def _fmt(v: float | None) -> str:
            return "-" if v is None else f"{v:.1f}"
        table.add_row(
            op,
            str(row.get("count", 0)),
            str(row.get("errors", 0)),
            str(row.get("samples", 0)),
            _fmt(row.get("p50_ms")),
            _fmt(row.get("p95_ms")),
            _fmt(row.get("p99_ms")),
        )
    console.print(table)
