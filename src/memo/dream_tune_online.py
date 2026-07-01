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
