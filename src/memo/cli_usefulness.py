"""`memo usefulness` — is memo actually consulted, and by whom?

memo is only worth keeping if the system reads it. This command turns the
consult ring buffer (`recall.log`, written by the Claude Code recall-hook AND
by every instrumented MCP tool — see `server._log_consult`) into a per-consumer
report: how often each configured client (Claude Code, Codex, or another
attributed agent) consults memo, its hit rate, and — the part that answers "do
the others actually use it?" — which expected consumers are **silent** (zero
consults).

Pure read over existing telemetry; no hot-path cost. The aggregation lives in
`dashboard.consult_breakdown` / `dashboard.recall_health`; this module only
wires it to flags and stdout.
"""

from __future__ import annotations

import json as _json

import click

from .config import Config
from .dashboard import consult_breakdown, reask_stats, recall_health


def _age(ts: str | None) -> str:
    from .dashboard import _human_age

    return _human_age(ts)


def _render_usefulness(*, limit: int = 500, as_json: bool = False) -> None:
    cfg = Config.from_env()
    state_dir = cfg.state_dir
    health = recall_health(state_dir, limit=limit)
    breakdown = consult_breakdown(state_dir, limit=limit)

    if as_json:
        click.echo(
            _json.dumps(
                {"recall_hook": health, "by_consumer": breakdown}, ensure_ascii=False, indent=2
            )
        )
        return

    consumers = breakdown["consumers"]
    silent = breakdown["silent"]

    if not consumers:
        click.echo("No consults recorded yet — memo has not been read.")
        click.echo(
            "(The recall-hook logs Claude Code consults; MCP tools log when "
            "callers pass `source=`.)"
        )
        return

    click.echo(f"memo usefulness — {breakdown['sampled']} consults sampled\n")
    composite_threshold = health.get("composite_score_threshold", 0.85)
    click.echo(
        f"  composite ranking: % of fired recalls whose top final score "
        f"is > {composite_threshold:.2f}"
    )
    click.echo(
        f"  {'consumer':<16} {'consults':>8} {'fired':>6} {'bail':>5} "
        f"{'hit%':>6} {'comp>%':>8} {'grnd%':>6} {'comp50':>6}  last"
    )
    click.echo("  " + "-" * 80)
    for c in consumers:
        hit = f"{c['hit_rate'] * 100:.0f}" if c["hit_rate"] is not None else "—"
        composite_rate = c.get("top_composite_score_rate", c.get("strong_hit_rate"))
        composite = f"{composite_rate * 100:.0f}" if composite_rate is not None else "—"
        # grounded% = outcome-based "actually used in the answer" (— until the
        # Stop-hook grounding detector has correlatable data for this consumer).
        grnd = f"{c['grounded_rate'] * 100:.0f}" if c.get("grounded_rate") is not None else "—"
        median_composite = c.get("median_top_composite_score", c.get("median_top_score"))
        top = f"{median_composite:.2f}" if median_composite is not None else "—"
        click.echo(
            f"  {c['consumer']:<16} {c['consults']:>8} {c['fired']:>6} "
            f"{c['bailed']:>5} {hit:>6} {composite:>8} {grnd:>6} "
            f"{top:>6}  {_age(c['last_seen'])}"
        )

    if silent:
        click.echo(f"\n⚠ Expected consumers with ZERO consults: {', '.join(silent)}")
        click.echo("  These layers are NOT reading memo as source-of-truth.")

    # "Used", not just "shown": explicit fetch-through of a surfaced memory.
    # Lower bound — the model usually consumes injected recall text inline
    # without a fetch, so true usefulness is at least this.
    ref_rate = health.get("referenced_rate")
    if ref_rate is not None:
        click.echo(
            f"\nreferenced_rate={ref_rate} "
            f"({health.get('referenced')}/{health.get('surfaced')} surfaced memories "
            f"later fetched — lower bound on 'used')."
        )

    # Outcome-based successor: the answer actually USED the surfaced memory
    # (Stop-hook grounding detector, lexical+embedding). Only correlatable rows
    # (session_id+turn) count, so it's null until new sessions accrue.
    g_rate = health.get("grounded_rate")
    if g_rate is not None:
        click.echo(
            f"grounded_rate={g_rate} "
            f"({health.get('grounded')}/{health.get('grounded_surfaced')} surfaced memories "
            f"used in the answer — outcome-based, not just shown)."
        )

    # Re-derivations memo prevented (a grounded recall not re-asked next turns).
    reask = reask_stats(state_dir, limit=limit)
    if reask.get("considered"):
        click.echo(
            f"reask_avoided={reask['reask_avoided']}/{reask['considered']} "
            f"(grounded recalls the user did NOT have to ask again — see `memo roi`)."
        )

    # One-line summary. hit% = fired that returned anything; the composite rate
    # uses the final ranking score (semantic similarity plus ranking boosts).
    total = breakdown["sampled"]
    n = len(consumers)
    composite_rate = health.get("top_composite_score_rate", health.get("strong_hit_rate"))
    click.echo(
        f"\nmemo consulted {total}× across {n} consumer(s); recall-hook "  # noqa: RUF001
        f"hit_rate={health.get('hit_rate')} "
        f"top_composite_score_rate={composite_rate}."
    )


@click.group(name="usefulness", invoke_without_command=True)
@click.option("--limit", default=500, show_default=True, help="Consult-log rows to sample.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
@click.pass_context
def usefulness(ctx: click.Context, *, limit: int = 500, as_json: bool = False) -> None:
    """Report how useful memo is: who consults it, hit rate, and silent gaps."""
    if ctx.invoked_subcommand is not None:
        return
    _render_usefulness(limit=limit, as_json=as_json)


@usefulness.command(name="doctor")
@click.option("--limit", default=500, show_default=True, help="Rows/signals to sample.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def usefulness_doctor(limit: int = 500, as_json: bool = False) -> None:
    """Diagnose memo adoption and trust signals."""
    from memo.usefulness_doctor import build_report, format_text_report

    cfg = Config.from_env()
    report = build_report(cfg, limit=limit)
    if as_json:
        click.echo(_json.dumps(report, ensure_ascii=False, indent=2))
        return
    click.echo(format_text_report(report))
