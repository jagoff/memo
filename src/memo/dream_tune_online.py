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


def clear_pending(state_dir: Path) -> None:
    pending_path(state_dir).unlink(missing_ok=True)


def append_ledger(state_dir: Path, entry: dict[str, Any]) -> None:
    p = ledger_path(state_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_ledger(state_dir: Path, *, limit: int = 50) -> list[dict[str, Any]]:
    try:
        lines = ledger_path(state_dir).read_text(encoding="utf-8").splitlines()
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
    the streak. ``entries`` is oldest→newest (as ``read_ledger`` returns)."""
    streak = 0
    for e in reversed(entries):
        if e.get("verdict") == "confirmed" and float(e.get("realized_delta", 0.0)) >= 0.0:
            streak += 1
        else:
            break
    return streak


def graduation_status(state_dir: Path, *, k: int) -> dict[str, Any]:
    """Read-only graduation readiness: is the min_sim proof loop's trailing
    confirmed-streak >= k? Never flips any flag — reports only."""
    streak = graduation_streak(read_ledger(state_dir, limit=max(k * 4, 20)))
    return {"streak": streak, "k": k, "graduated": streak >= k}
