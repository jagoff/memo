"""`memo graduation` — inspect and control the graduation controller."""
from __future__ import annotations

import click

from memo.config import Config
from memo.graduation import ledger, overlay_ops
from memo.graduation.registry import default_candidates


@click.group(name="graduation")
def graduation_group() -> None:
    """Shadow-prove and flip dark flags (see MEMO_GRADUATION_CONTROLLER_ENABLED)."""


@graduation_group.command(name="status")
def status_cmd() -> None:
    cfg = Config.from_env()
    for c in default_candidates():
        s = ledger.streak(cfg.state_dir, c.flag)
        live = "live" if overlay_ops.is_flipped_on(cfg.state_dir, c.flag) else "—"
        mode = "auto" if c.auto_flip else "report-only"
        click.echo(f"{c.flag:<34} streak {s}/{c.k}  overlay:{live}  [{mode}]")


@graduation_group.command(name="explain")
def explain_cmd() -> None:
    from memo.graduation.controller import run_graduation_controller
    from memo.memory import Memory

    cfg = Config.from_env()
    receipt = run_graduation_controller(cfg, Memory(cfg), dry_run=True)
    for r in receipt["candidates"]:
        if r["status"] == "vetoed":
            click.echo(f"{r['flag']:<34} vetoed (MEMO_* env var set)")
            continue
        click.echo(
            f"{r['flag']:<34} would:{r['status']:<12} "
            f"Δprec {r.get('delta_prec', 0.0):+.3f}  streak {r.get('streak', 0)}/{r.get('k', 0)}"
        )


@graduation_group.command(name="revert")
@click.argument("flag")
def revert_cmd(flag: str) -> None:
    cfg = Config.from_env()
    was_on = overlay_ops.is_flipped_on(cfg.state_dir, flag)
    overlay_ops.revert(cfg.state_dir, flag)
    click.echo(f"{flag}: {'reverted (overlay flip removed)' if was_on else 'nothing to revert'}")
