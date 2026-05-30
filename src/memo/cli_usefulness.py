"""`memo usefulness` — is memo actually consulted, and by whom?

memo is only worth keeping if the system reads it. This command turns the
consult ring buffer (`recall.log`, written by the Claude Code recall-hook AND
by every instrumented MCP tool — see `server._log_consult`) into a per-consumer
report: how often each layer (Claude Code / synapse / memflow / agents) consults
memo, its hit rate, and — the part that answers "no sé si los demás lo usan" —
which expected consumers are **silent** (zero consults).

Pure read over existing telemetry; no hot-path cost. The aggregation lives in
`dashboard.consult_breakdown` / `dashboard.recall_health`; this module only
wires it to flags and stdout.
"""

from __future__ import annotations

import json as _json

import click

from .config import Config
from .dashboard import consult_breakdown, recall_health


def _age(ts: str | None) -> str:
    from .dashboard import _human_age

    return _human_age(ts)


@click.command(name="usefulness")
@click.option("--limit", default=500, show_default=True, help="Consult-log rows to sample.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def usefulness(*, limit: int = 500, as_json: bool = False) -> None:
    """Report how useful memo is: who consults it, hit rate, and silent gaps."""
    cfg = Config.from_env()
    state_dir = cfg.state_dir
    health = recall_health(state_dir, limit=limit)
    breakdown = consult_breakdown(state_dir, limit=limit)

    if as_json:
        click.echo(_json.dumps({"recall_hook": health, "by_consumer": breakdown},
                               ensure_ascii=False, indent=2))
        return

    consumers = breakdown["consumers"]
    silent = breakdown["silent"]

    if not consumers:
        click.echo("No consults recorded yet — memo has not been read.")
        click.echo("(The recall-hook logs Claude Code consults; MCP tools log when "
                   "callers pass `source=`.)")
        return

    click.echo(f"memo usefulness — {breakdown['sampled']} consults sampled\n")
    click.echo(f"  {'consumer':<16} {'consults':>8} {'fired':>6} {'bail':>5} "
               f"{'hit%':>6} {'top':>6}  last")
    click.echo("  " + "-" * 62)
    for c in consumers:
        hit = f"{c['hit_rate']*100:.0f}" if c["hit_rate"] is not None else "—"
        top = f"{c['median_top_score']:.2f}" if c["median_top_score"] is not None else "—"
        click.echo(f"  {c['consumer']:<16} {c['consults']:>8} {c['fired']:>6} "
                   f"{c['bailed']:>5} {hit:>6} {top:>6}  {_age(c['last_seen'])}")

    if silent:
        click.echo(f"\n⚠ Expected consumers with ZERO consults: {', '.join(silent)}")
        click.echo("  These layers are NOT reading memo as source-of-truth.")

    # One-line verdict.
    total = breakdown["sampled"]
    n = len(consumers)
    click.echo(f"\nmemo consulted {total}× across {n} consumer(s); "
               f"recall-hook hit_rate={health.get('hit_rate')}.")
