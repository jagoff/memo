"""`memo gaps` + `memo outcome` — The Outcome Loop CLI surface.

``memo gaps``     — knowledge-seeking prompts memo could NOT answer (what to
                    capture next). Pure read over recall.log + grounding.log.
``memo outcome``  — reconcile memory_health.roi_score from real grounding
                    outcomes (promote memorias that ground answers, demote ones
                    that surface but never help). The single write in the loop.
"""

from __future__ import annotations

import json as _json

import click

from .config import Config


@click.command(name="gaps")
@click.option("--limit", default=2000, show_default=True, help="Recall-log rows to scan.")
@click.option("--min-count", default=1, show_default=True, help="Minimum cluster size to report.")
@click.option("--top", default=20, show_default=True, help="Max gap clusters to show.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def gaps(*, limit: int, min_count: int, top: int, as_json: bool) -> None:
    """What memo could NOT answer — knowledge gaps to fill."""
    from .outcome import detect_gaps

    cfg = Config.from_env()
    found = detect_gaps(cfg.state_dir, limit=limit, min_count=min_count)

    if as_json:
        click.echo(_json.dumps(found[:top], ensure_ascii=False, indent=2))
        return
    if not found:
        click.echo("No gaps detected — memo answered everything it was asked. ✅")
        return
    click.echo(f"memo gaps — {len(found)} topic(s) memo could not answer\n")
    click.echo(f"  {'times':>5}  {'reason':<30} question")
    click.echo("  " + "-" * 78)
    for g in found[:top]:
        reason = ", ".join(g["reasons"])[:30]
        prompt = (g["prompt"] or "")[:60]
        click.echo(f"  {g['count']:>5}  {reason:<30} {prompt}")
    click.echo("\nCapture these topics with `memo save` so they stop being gaps.")


@click.command(name="outcome")
@click.option("--apply", "do_apply", is_flag=True, help="Write the reconciled roi_score (default: dry-run).")
@click.option(
    "--archive-dead",
    is_flag=True,
    help="Also reversibly forget dead-weight memorias (surfaced ≥ "
    "MEMO_OUTCOME_DEAD_MIN_SURFACED times, never grounded). Reverse with `memo unforget`.",
)
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def outcome(*, do_apply: bool, archive_dead: bool, as_json: bool) -> None:
    """Reconcile roi_score from real grounding outcomes (the learning step)."""
    from .flags import flag_bool, flag_int
    from .memory import Memory
    from .outcome import compute_utilities, dead_weight, reconcile_roi

    cfg = Config.from_env()
    min_surfaced = flag_int("MEMO_OUTCOME_DEAD_MIN_SURFACED") or 8

    if not do_apply:
        u = compute_utilities(cfg.state_dir)
        scored = len(u["by_prefix"])
        helpful = sum(1 for v in u["by_prefix"].values() if v["grounded"] > 0)
        mem = Memory(cfg)
        dead = dead_weight(mem, min_surfaced=min_surfaced)
        if as_json:
            click.echo(_json.dumps({"dry_run": True, "dead_weight": dead, **u}, ensure_ascii=False, indent=2))
            return
        click.echo(
            f"outcome (dry-run) — {scored} memorias with history; "
            f"{helpful} grounded ≥1 answer; baseline grounded={u['prior_mean']}."
        )
        if dead:
            click.echo(
                f"  dead weight: {len(dead)} memoria(s) surfaced >={min_surfaced}x without "
                f"ever grounding (candidates to archive with --apply --archive-dead)."
            )
        click.echo("Run with --apply to write roi_score from these outcomes.")
        if not flag_bool("MEMO_OUTCOME_RANKING_ENABLED"):
            click.echo(
                "Note: MEMO_OUTCOME_RANKING_ENABLED is OFF — ranking still uses "
                "access-based roi. Enable it so outcomes take precedence."
            )
        return

    mem = Memory(cfg)
    res = reconcile_roi(mem)
    archived: list[str] = []
    if archive_dead:
        for d in dead_weight(mem, min_surfaced=min_surfaced):
            if mem.forget(d["id"], reason=f"outcome: surfaced {d['surfaced']}x without grounding") is not None:
                archived.append(d["id"])
    if as_json:
        click.echo(_json.dumps({**res, "archived": archived}, ensure_ascii=False, indent=2))
        return
    click.echo(
        f"outcome applied — roi_score updated for {res['updated']} memoria(s) "
        f"(baseline grounded={res['prior_mean']}, range [{res['floor']},{res['cap']}])."
    )
    if archive_dead:
        click.echo(f"  dead weight archived (reversible with `memo unforget`): {len(archived)}")
    if not flag_bool("MEMO_OUTCOME_RANKING_ENABLED"):
        click.echo(
            "Note: MEMO_OUTCOME_RANKING_ENABLED is OFF — wrote roi_score but the "
            "access-boost will overwrite it. Enable the flag so the outcome takes precedence."
        )
