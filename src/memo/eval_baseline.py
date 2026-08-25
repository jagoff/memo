"""Phase-0 baseline snapshot.

Freezes {offline recall metrics, online grounded count over 7d + 30d, active
tuned-params version} so later self-improvement changes can be attributed to a
config and compared against a known-good point. Written by `memo eval baseline`.

The online window used to also carry a "tokens" figure derived from
``grounded × MEMO_ROI_TOKENS_PER_GROUNDED`` — that hardcoded-constant estimate
was retired (see CHANGELOG); only the real, physical grounded count remains.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memo import token_ledger
from memo.tuned_overlay import params_version

SNAPSHOT_SCHEMA = "memo.eval_baseline.v1"


def snapshot_path(state_dir: Path) -> Path:
    return Path(state_dir) / "eval" / "baseline_snapshot.json"


def _window(daily: list[dict[str, Any]], days: int) -> dict[str, int]:
    recent = daily[-days:] if days < len(daily) else daily
    return {"grounded": sum(int(d.get("grounded", 0)) for d in recent)}


def build_baseline_snapshot(
    state_dir: Path,
    offline: dict[str, float],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Assemble the baseline snapshot. ``offline`` is the recall metrics the
    caller already computed (precision_at_k / noise_at_k); online numbers come
    from the durable token ledger; the version pins the active overlay."""
    sd = Path(state_dir)
    token_ledger.roll_up(sd)
    summary = token_ledger.summarize(sd, days_back=30)
    daily: list[dict[str, Any]] = summary.get("daily", [])
    ts = (now or datetime.now(UTC)).isoformat(timespec="seconds")
    return {
        "schema": SNAPSHOT_SCHEMA,
        "ts": ts,
        "params_version": params_version(sd),
        "offline": {
            "precision_at_k": float(offline.get("precision_at_k", 0.0)),
            "noise_at_k": float(offline.get("noise_at_k", 0.0)),
        },
        "online": {
            "window_7d": _window(daily, 7),
            "window_30d": _window(daily, 30),
        },
    }
