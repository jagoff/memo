"""`memo health` — operational health snapshot.

Thin CLI over `memo.health_report.build_health_report`: corpus size,
index dims, embedder profile, health-score coverage, and warnings.
"""

from __future__ import annotations

import json as _json

import click

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config
from memo.health_report import build_health_report


def _fmt_bytes(n: int | None) -> str:
    if not n:
        return "—"
    units = ["B", "KB", "MB", "GB"]
    size = float(n)
    for u in units:
        if size < 1024 or u == units[-1]:
            return f"{size:.0f}{u}" if u == "B" else f"{size:.1f}{u}"
        size /= 1024
    return f"{n}B"


@click.command(name="health")
@click.option("--probe", is_flag=True, help="Time one embed_query call (loads the embedder).")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def health(probe: bool, as_json: bool) -> None:
    """Report corpus/index/embedder health and warnings.

    Read-only. Example: memo health   |   memo health --probe --json
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)
    report = build_health_report(mem, probe_embedder=probe)

    if as_json:
        click.echo(_json.dumps(report, indent=2))
        return

    corpus = report["corpus"]
    index = report["index"]
    embedder = report["embedder"]
    health_tbl = report["health_table"]

    console.print("[bold]memo health[/bold]")
    console.print(
        f"  corpus     : {corpus['memorias']} memorias "
        f"({corpus.get('archived') or 0} archived, {_fmt_bytes(corpus.get('db_size_bytes'))})"
    )
    dims_flag = "[green]ok[/green]" if index["dims_ok"] else "[red]MISMATCH[/red]"
    console.print(
        f"  index      : vec dims {index['vec_dims']}/{index['expected_dims']} {dims_flag}, "
        f"fts {'ready' if index['fts_ready'] else 'missing'} ({index['fts_backend']})"
    )
    lat = embedder["latency_ms"]
    console.print(
        f"  embedder   : {embedder['model']} ({embedder['dims']}d)"
        + (f", {lat}ms/query" if lat is not None else "")
    )
    console.print(
        f"  health tbl : {health_tbl['tracked']} tracked "
        f"({health_tbl['low_confidence']} low-confidence, {health_tbl['high_roi']} high-ROI)"
    )
    console.print(f"  feedback   : {report['feedback']['records'] or 0} signals")

    warnings = report.get("warnings") or []
    if warnings:
        console.print()
        for w in warnings:
            console.print(f"  [yellow]⚠ {w}[/yellow]")
    else:
        console.print("\n  [green]✓ no warnings[/green]")
