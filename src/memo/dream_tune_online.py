"""Phase-1 online proof loop for the recall self-tuner.

Phase 0 stamped every grounding row with the tuned-params version live when the
recall happened. This module uses that attribution to judge a tuner change by
the grounding accumulated under its NEW params version (a later, out-of-sample
cohort) — not by the offline label eval that motivated it. A change whose
realized online grounded-fraction regresses is reverted; every verdict lands in
a durable ledger that `memo dream status` surfaces.

MLX-free: operates only over grounding.log + two JSON sidecars under
``state_dir/dream/``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memo.dashboard import GROUNDED_SCORE, read_grounding_log
from memo.tuned_overlay import params_version

PENDING_FILE = "tune_pending.json"
LEDGER_FILE = "tune_ledger.jsonl"


def _dream_dir(state_dir: Path) -> Path:
    return Path(state_dir) / "dream"


def pending_path(state_dir: Path) -> Path:
    return _dream_dir(state_dir) / PENDING_FILE


def ledger_path(state_dir: Path) -> Path:
    return _dream_dir(state_dir) / LEDGER_FILE


def cohort_fraction(
    rows: list[dict[str, Any]], params_version: str, *, threshold: float = GROUNDED_SCORE
) -> tuple[float, int]:
    """(grounded fraction, cohort size) over rows stamped with ``params_version``.

    A row is grounded when its numeric ``used_score`` is >= ``threshold``. Rows
    without a numeric ``used_score`` are ignored. Empty cohort → ``(0.0, 0)``.
    """
    scores = [
        float(r["used_score"])
        for r in rows
        if r.get("params_version") == params_version
        and isinstance(r.get("used_score"), (int, float))
    ]
    n = len(scores)
    if n == 0:
        return 0.0, 0
    grounded = sum(1 for s in scores if s >= threshold)
    return grounded / n, n


def online_fraction(
    state_dir: Path,
    params_version: str,
    *,
    threshold: float = GROUNDED_SCORE,
    limit: int = 4000,
) -> tuple[float, int]:
    """cohort_fraction over the live grounding.log for ``params_version``."""
    rows = read_grounding_log(Path(state_dir), limit=limit)
    return cohort_fraction(rows, params_version, threshold=threshold)


def read_pending(state_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads(pending_path(state_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_pending(state_dir: Path, record: dict[str, Any]) -> None:
    p = pending_path(state_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2), encoding="utf-8")


_DEFAULT_KNOB = "MEMO_RECALL_MIN_SIM"


def record_pending(
    state_dir: Path,
    *,
    knob: str,
    value_before: Any,
    value_after: Any,
    offline_before: dict[str, float],
    offline_after: dict[str, float],
    version_before: str,
) -> None:
    """Record an applied tuner change for a later night to resolve online.

    Generic over which knob changed. Must be called AFTER the overlay write (so
    ``params_version`` reflects the new config) and given the ``version_before``
    captured BEFORE that write. ``online_before`` is the pre-apply grounded
    fraction of the old-version cohort. Field names ``floor_before/after`` are
    kept for ledger back-compat (they hold the knob's before/after value)."""
    write_pending(
        state_dir,
        {
            "knob": knob,
            "floor_before": value_before,
            "floor_after": value_after,
            "version_before": version_before,
            "version_after": params_version(state_dir),
            "offline_before": offline_before,
            "offline_after": offline_after,
            "online_before": online_fraction(state_dir, version_before)[0],
        },
    )


def clear_pending(state_dir: Path) -> None:
    pending_path(state_dir).unlink(missing_ok=True)


COOLDOWN_FILE = "tune_cooldown"


def cooldown_path(state_dir: Path) -> Path:
    return _dream_dir(state_dir) / COOLDOWN_FILE


def set_revert_cooldown(state_dir: Path) -> None:
    """Mark that an online revert happened this cycle. Other tuner passes hold
    the overlay steady for the rest of the cycle — no new apply lands in the same
    cycle a change was reverted (prevents the reverting pass and a co-gated pass
    from re-applying the just-reverted value)."""
    p = cooldown_path(state_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("1", encoding="utf-8")


def clear_revert_cooldown(state_dir: Path) -> None:
    cooldown_path(state_dir).unlink(missing_ok=True)


def in_revert_cooldown(state_dir: Path) -> bool:
    return cooldown_path(state_dir).exists()


def append_ledger(state_dir: Path, entry: dict[str, Any]) -> None:
    p = ledger_path(state_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _read_jsonl_tail(path: Path, *, limit: int) -> list[dict[str, Any]]:
    """Read the last ``limit`` lines of a JSONL file, skipping corrupt/blank
    lines. Missing file → ``[]``."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            out.append(json.loads(stripped))
        except json.JSONDecodeError:
            continue
    return out


def read_ledger(state_dir: Path, *, limit: int = 50) -> list[dict[str, Any]]:
    return _read_jsonl_tail(ledger_path(state_dir), limit=limit)


def resolve_pending(
    state_dir: Path,
    *,
    min_cohort: int,
    eps: float,
    live_version: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Resolve a prior night's applied change against its out-of-sample cohort.

    ``status`` is one of:
      - ``"none"``      — no pending change.
      - ``"waiting"``   — the new-version cohort is still smaller than
        ``min_cohort``; the pending record is kept for a later night.
      - ``"confirmed"`` — realized online fraction held (>= before - eps); the
        change stays. Pending cleared, ledger appended.
      - ``"reverted"``  — realized online fraction regressed (< before - eps).
        Pending cleared, ledger appended; the return carries ``offline_before``
        so the CALLER can roll back the overlay and restore the offline baseline.

    No overlay side effects here — only grounding.log reads + sidecar writes.
    """
    pending = read_pending(state_dir)
    if not pending:
        return {"status": "none"}

    version_after = str(pending.get("version_after", ""))
    online_after, n_after = online_fraction(state_dir, version_after)
    if n_after < min_cohort:
        # Version drift: a co-running overlay writer (e.g. the graph-weight pass,
        # gated by the same flag) moved the live config off version_after, so its
        # cohort can never fill — expire honestly instead of waiting forever. Only
        # genuinely wait while the applied config is still the live one.
        if live_version is not None and live_version != version_after:
            entry = {
                "resolved_ts": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
                "verdict": "expired",
                "reason": "version_drift",
                "knob": pending.get("knob", _DEFAULT_KNOB),
                "floor_before": pending.get("floor_before"),
                "floor_after": pending.get("floor_after"),
                "version_before": pending.get("version_before"),
                "version_after": version_after,
                "n_after": n_after,
            }
            append_ledger(state_dir, entry)
            clear_pending(state_dir)
            return {"status": "expired", **entry}
        return {
            "status": "waiting",
            "version_after": version_after,
            "n_after": n_after,
            "min_cohort": min_cohort,
        }

    online_before = float(pending.get("online_before", 0.0))
    realized = round(online_after - online_before, 4)
    verdict = "reverted" if realized < -eps else "confirmed"
    entry = {
        "resolved_ts": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
        "verdict": verdict,
        "knob": pending.get("knob", _DEFAULT_KNOB),
        "floor_before": pending.get("floor_before"),
        "floor_after": pending.get("floor_after"),
        "version_before": pending.get("version_before"),
        "version_after": version_after,
        "offline_before": pending.get("offline_before"),
        "offline_after": pending.get("offline_after"),
        "online_before": round(online_before, 4),
        "online_after": round(online_after, 4),
        "n_after": n_after,
        "realized_delta": realized,
    }
    append_ledger(state_dir, entry)
    clear_pending(state_dir)
    return {"status": verdict, **entry}


def graduation_streak(entries: list[dict[str, Any]]) -> int:
    """Count the trailing run of graduating verdicts: consecutive newest-first
    ledger entries with verdict ``"confirmed"`` and a non-negative
    ``realized_delta``. Any ``reverted``/``expired`` (or a negative delta) breaks
    the streak. ``entries`` is oldest→newest (as ``read_ledger`` returns).
    Covers any tuned knob, not only min_sim."""
    streak = 0
    for e in reversed(entries):
        if e.get("verdict") == "confirmed" and float(e.get("realized_delta", 0.0)) >= 0.0:
            streak += 1
        else:
            break
    return streak


def graduation_status(state_dir: Path, *, k: int) -> dict[str, Any]:
    """Read-only graduation readiness: is the proof loop's trailing
    confirmed-streak >= k (any tuned knob)? Never flips any flag — reports only."""
    streak = graduation_streak(read_ledger(state_dir, limit=max(k * 4, 20)))
    return {"streak": streak, "k": k, "graduated": streak >= k}


def has_unresolved_pending(state_dir: Path) -> bool:
    """True when the proof loop has an applied-but-not-yet-resolved change in
    flight (any tuned knob). Other tuner passes consult this to hold the overlay
    steady — one change per proof cycle — so the pending's grounding cohort is
    not orphaned."""
    return read_pending(state_dir) is not None
