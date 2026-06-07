"""`memo roi` — make memo's value undeniable.

Pure read over the recall→use→outcome ledger (recall.log + grounding.log +
usage.log). Turns the raw signals into the numbers a user feels:
  - grounding rate (answers that actually USED a surfaced memoria) — per client
  - re-derivations prevented (reask_avoided)
  - estimated time saved (conservative, clearly labeled an estimate)
  - per-client value table + silent gaps

No new MCP tool — this is a CLI/observability surface only (memo stays the store).
"""

from __future__ import annotations

import json as _json

import click

from .config import Config
from .dashboard import (
    consult_breakdown,
    grounding_log_path,
    read_grounding_log,
    reask_stats,
    recall_health,
)


def _secs(env: str, default: int) -> int:
    from memo.flags import flag_int

    return flag_int(env) or default


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def compute_roi(state_dir, *, limit: int = 500, window_turns: int = 4) -> dict:
    """Aggregate the ROI numbers from the ledgers. Returns a plain dict so the
    same computation backs both the human table and `--json`."""
    health = recall_health(state_dir, limit=limit)
    breakdown = consult_breakdown(state_dir, limit=limit)
    reask = reask_stats(state_dir, window_turns=window_turns, limit=limit)

    # Per-client action counts from grounding.log (downstream_action set).
    actions: dict[str, dict[str, int]] = {}
    for g in read_grounding_log(state_dir):
        client = g.get("client") or "unknown"
        a = actions.setdefault(client, {"grounded": 0, "actions": 0})
        score = g.get("used_score")
        from .dashboard import GROUNDED_SCORE

        if isinstance(score, (int, float)) and float(score) >= GROUNDED_SCORE:
            a["grounded"] += 1
        if g.get("downstream_action"):
            a["actions"] += 1

    grounded_total = int(health.get("grounded") or 0)
    reask_avoided = int(reask.get("reask_avoided") or 0)
    secs_grounded = _secs("MEMO_ROI_SECS_PER_GROUNDED", 30)
    secs_reask = _secs("MEMO_ROI_SECS_PER_REASK", 120)
    time_saved_s = grounded_total * secs_grounded + reask_avoided * secs_reask

    return {
        "grounded_rate": health.get("grounded_rate"),
        "grounded": grounded_total,
        "grounded_surfaced": health.get("grounded_surfaced"),
        "referenced_rate": health.get("referenced_rate"),
        "reask": reask,
        "time_saved_seconds": time_saved_s,
        "time_saved_human": _fmt_duration(time_saved_s),
        "secs_per_grounded": secs_grounded,
        "secs_per_reask": secs_reask,
        "by_consumer": breakdown["consumers"],
        "silent": breakdown["silent"],
        "actions_by_client": actions,
        "sampled": breakdown["sampled"],
    }


@click.command(name="roi")
@click.option("--limit", default=500, show_default=True, help="Consult-log rows to sample.")
@click.option("--window-turns", default=4, show_default=True, help="Re-ask detection window.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def roi(*, limit: int = 500, window_turns: int = 4, as_json: bool = False) -> None:
    """Show the value memo generated: grounding, re-derivations prevented, time saved."""
    cfg = Config.from_env()
    data = compute_roi(cfg.state_dir, limit=limit, window_turns=window_turns)

    if as_json:
        click.echo(_json.dumps(data, ensure_ascii=False, indent=2))
        return

    if not data["by_consumer"]:
        click.echo("No consults recorded yet — memo has not been read.")
        return

    g_rate = data["grounded_rate"]
    g_line = (
        f"{g_rate * 100:.0f}% grounded "
        f"({data['grounded']}/{data['grounded_surfaced']} surfaced memorias used in the answer)"
        if g_rate is not None
        else "grounding: no correlatable data yet (needs new sessions post-P0)"
    )
    reask = data["reask"]
    click.echo("memo ROI — value generated\n")
    click.echo(f"  grounding         {g_line}")
    if reask.get("reask_avoided") is not None:
        click.echo(
            f"  re-derivations    {reask['reask_avoided']} prevented "
            f"({reask['considered']} grounded recalls, {reask['reask']} re-asked)"
        )
    click.echo(
        f"  estimated time    ~{data['time_saved_human']} saved "
        f"(est: {data['secs_per_grounded']}s/grounded + {data['secs_per_reask']}s/re-ask avoided)"
    )

    # Per-client value table.
    click.echo(f"\n  {'client':<16} {'consults':>8} {'hit%':>6} {'grnd%':>6} {'act':>5}  last")
    click.echo("  " + "-" * 60)
    actions = data["actions_by_client"]
    for c in data["by_consumer"]:
        name = c["consumer"]
        hit = f"{c['hit_rate'] * 100:.0f}" if c.get("hit_rate") is not None else "—"
        grnd = f"{c['grounded_rate'] * 100:.0f}" if c.get("grounded_rate") is not None else "—"
        act = actions.get(name, {}).get("actions", 0)
        act_s = str(act) if act else "—"
        from .dashboard import _human_age

        click.echo(
            f"  {name:<16} {c['consults']:>8} {hit:>6} {grnd:>6} {act_s:>5}  "
            f"{_human_age(c.get('last_seen'))}"
        )

    if data["silent"]:
        click.echo(f"\n  ⚠ wired but silent (not reading memo): {', '.join(data['silent'])}")

    if not grounding_log_path(cfg.state_dir).is_file():
        click.echo(
            "\n  (grounding.log empty — run a few Claude Code turns so the Stop-hook "
            "detector can populate outcome data.)"
        )
