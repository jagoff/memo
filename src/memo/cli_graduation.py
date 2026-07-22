"""`memo graduation` — inspect and control the graduation controller."""

from __future__ import annotations

import click

from memo.config import Config
from memo.graduation import ledger, overlay_ops
from memo.graduation.registry import all_candidates


@click.group(name="graduation")
def graduation_group() -> None:
    """Shadow-prove and flip dark flags."""


@graduation_group.command(name="status")
def status_cmd() -> None:
    cfg = Config.from_env()
    for c in all_candidates():
        s = ledger.streak(cfg.state_dir, c.flag)
        live_val = overlay_ops.overlay_value(cfg.state_dir, c.flag)
        live = str(live_val) if live_val is not None else "—"
        mode = "auto" if c.auto_flip else "report-only"
        click.echo(f"{c.flag:<34} streak {s}/{c.k}  overlay:{live}  [{mode}]")


@graduation_group.command(name="revert")
@click.argument("flag")
def revert_cmd(flag: str) -> None:
    cfg = Config.from_env()
    was_on = overlay_ops.is_flipped_on(cfg.state_dir, flag)
    overlay_ops.revert(cfg.state_dir, flag)
    click.echo(f"{flag}: {'reverted (overlay flip removed)' if was_on else 'nothing to revert'}")
