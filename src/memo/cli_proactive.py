"""`memo digest` — human pull surface for the proactive engine.

Prints the grouped digest (reliability/continuity/déjà-vu/health/ROI nudges).
`--dismiss <id>` / `--snooze <kind> [--days N]` record anti-annoyance state
(`ProactiveStore.dismiss` / `snooze_kind`) plus a `dismissed` feedback signal
that feeds `ProactiveStore.kind_multipliers` on future runs. Respects
`MEMO_PROACTIVE_ENABLED` — off by default (dark flag), prints a one-line hint
instead of the digest.

Registered onto the root group in cli.py via `cli.add_command(digest)`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import click

from memo.cli_common import console
from memo.config import Config
from memo.flags import flag_bool

if TYPE_CHECKING:
    from memo.proactive.nudge import Nudge
    from memo.proactive.store import ProactiveStore


def record_acted_if_matches(
    store: ProactiveStore, *, command_line: str, now: str, window_min: int = 30
) -> None:
    """Frictionless acted-feedback: record `acted` when a run command matches a nudge.

    If an active candidate's `action` equals `command_line` verbatim and it was
    surfaced within `window_min` minutes of `now`, record an `acted` outcome for
    it. No explicit user marking required — running the suggested action IS the
    signal.
    """

    def _parse(ts: str) -> datetime:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))

    now_dt = _parse(now)
    for n in store.active_candidates(now):
        if n.action == command_line and now_dt - _parse(n.created_at) <= timedelta(
            minutes=window_min
        ):
            store.record_feedback(n.id, n.kind, "acted", now)


def _record_dismiss_feedback(
    store: ProactiveStore, dismiss_id: str, active: list[Nudge], now: str
) -> None:
    match = next((n for n in active if n.id == dismiss_id), None)
    store.dismiss(dismiss_id, now)
    if match is not None:
        store.record_feedback(dismiss_id, match.kind, "dismissed", now)


def _record_snooze_feedback(
    store: ProactiveStore, snooze_kind: str, days: int, active: list[Nudge], now: str
) -> None:
    until = (datetime.now(UTC) + timedelta(days=days)).isoformat()
    store.snooze_kind(snooze_kind, until)
    for n in active:
        if n.kind == snooze_kind:
            store.record_feedback(n.id, n.kind, "dismissed", now)


@click.command(name="digest")
@click.option("--dismiss", "dismiss_id", default=None, help="Dismiss a single nudge by id.")
@click.option(
    "--snooze",
    "snooze_kind_",
    default=None,
    help="Snooze an entire nudge kind (e.g. reliability, continuity).",
)
@click.option(
    "--days", default=7, type=int, show_default=True, help="Snooze duration, used with --snooze."
)
def digest(dismiss_id: str | None, snooze_kind_: str | None, days: int) -> None:
    """Print the grouped proactive digest.

    `--dismiss <id>` hides one nudge permanently; `--snooze <kind>` hides an
    entire kind for `--days` (default 7). Both feed a `dismissed` outcome
    into `ProactiveStore.kind_multipliers`, damping that kind's future rank.
    """
    if not flag_bool("MEMO_PROACTIVE_ENABLED"):
        console.print("proactive engine disabled (set MEMO_PROACTIVE_ENABLED=1)")
        return

    from memo.proactive.engine import compute_routed
    from memo.proactive.store import ProactiveStore
    from memo.proactive.surfaces import render_digest

    cfg = Config.from_env()
    now_dt = datetime.now(UTC)
    now = now_dt.isoformat()
    day = now_dt.date().isoformat()

    with ProactiveStore(cfg.state_dir / "proactive.db") as store:
        if dismiss_id or snooze_kind_:
            active = store.active_candidates(now)
            if dismiss_id:
                _record_dismiss_feedback(store, dismiss_id, active, now)
            if snooze_kind_:
                _record_snooze_feedback(store, snooze_kind_, days, active, now)

        routed = compute_routed(store, now=now, day=day)
    console.print(render_digest(routed))
