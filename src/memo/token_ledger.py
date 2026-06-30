"""Durable token-savings ledger — the data behind `memo tokens`.

grounding.log is the per-row, timestamped record of *grounded* recalls (a
surfaced memory the answer actually USED — i.e. a re-derivation memo prevented).
But that log is capped (`cap=1000`, ~12 days of history) and rotates, so an
all-time "tokens saved" total read from it alone would plateau the moment old
rows scroll out.

This module folds the grounded events into a tiny per-day file
(`state_dir/token_savings_daily.json`, one row per local day) BEFORE they
rotate, so the historic total is:

  * **durable** — old days survive grounding.log eviction, and
  * **monotonic** — a day's grounded count never decreases (max-merge), so the
    historic line only ever rises as memo accumulates more grounded memories.

Tokens saved are derived at read time: ``grounded × MEMO_ROI_TOKENS_PER_GROUNDED``
(default 350 — the same per-grounded estimate `memo roi` uses). Storing the raw
grounded count (not the token product) keeps the durable signal physical and
lets the tunable rate re-price history without rewriting the ledger.

Pure stdlib + `memo.dashboard` (leaf log readers) + `memo.flags` — no MLX, no
`memo.memory` import, so it is cheap enough to roll up on every Stop hook.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from .dashboard import GROUNDED_SCORE, read_grounding_log
from .flags import flag_int

LEDGER_SCHEMA = "memo.token_savings.daily.v1"
_DEFAULT_TOKENS_PER_GROUNDED = 350


def ledger_path(state_dir: Path) -> Path:
    return state_dir / "token_savings_daily.json"


def _tokens_per_grounded() -> int:
    v = flag_int("MEMO_ROI_TOKENS_PER_GROUNDED")
    return _DEFAULT_TOKENS_PER_GROUNDED if v is None else v


def _local_date(ts: str) -> str | None:
    """ISO UTC timestamp → local ``YYYY-MM-DD`` (None if unparseable).

    Day/month boundaries follow the user's local clock, not UTC, so "today"
    matches what they'd intuitively expect.
    """
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone().date().isoformat()


def read_ledger(state_dir: Path) -> dict:
    """Load the durable ledger, returning a fresh empty one on absence/corruption."""
    path = ledger_path(state_dir)
    if not path.is_file():
        return {"schema": LEDGER_SCHEMA, "days": {}}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema": LEDGER_SCHEMA, "days": {}}
    if not isinstance(doc, dict) or not isinstance(doc.get("days"), dict):
        return {"schema": LEDGER_SCHEMA, "days": {}}
    return doc


def write_ledger(state_dir: Path, ledger: dict) -> None:
    """Atomically persist the ledger (tmp file + os.replace)."""
    path = ledger_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def grounded_by_day(
    rows: list[dict],
    *,
    threshold: float = GROUNDED_SCORE,
    to_day: Callable[[str], str | None] = _local_date,
) -> dict[str, int]:
    """Count grounded rows (``used_score >= threshold``) per local day."""
    out: dict[str, int] = {}
    for r in rows:
        score = r.get("used_score")
        if not isinstance(score, (int, float)) or float(score) < threshold:
            continue
        day = to_day(r.get("ts", ""))
        if day is None:
            continue
        out[day] = out.get(day, 0) + 1
    return out


def roll_up(state_dir: Path, *, limit: int = 4000) -> dict:
    """Fold grounded events from grounding.log into the durable ledger.

    Monotonic per day: a day's stored grounded count is ``max(stored, observed)``,
    so eviction from the rolling grounding.log can never shrink a day already
    captured, while a still-in-window day keeps climbing as new rows land.
    Returns the updated ledger.
    """
    ledger = read_ledger(state_dir)
    days: dict = ledger.setdefault("days", {})
    observed = grounded_by_day(read_grounding_log(state_dir, limit=limit))
    for day, n in observed.items():
        prev = int(days.get(day, {}).get("grounded", 0))
        if n >= prev:
            days[day] = {"grounded": n}
    if observed:
        write_ledger(state_dir, ledger)
    return ledger


def _month_of(day_key: str) -> str:
    return day_key[:7]


def summarize(
    state_dir: Path,
    *,
    today: date | None = None,
    days_back: int = 30,
    months_back: int = 6,
) -> dict:
    """Read-only roll of the durable ledger into the numbers + chart series.

    Does NOT roll up — callers run :func:`roll_up` first when they want fresh
    data. ``today`` is injectable for deterministic tests.
    """
    ledger = read_ledger(state_dir)
    days: dict[str, dict] = ledger.get("days", {})
    tpg = _tokens_per_grounded()
    today = today or datetime.now().astimezone().date()
    today_key = today.isoformat()
    month_prefix = today.strftime("%Y-%m")

    def _grounded(day_key: str) -> int:
        return int(days.get(day_key, {}).get("grounded", 0))

    def _bucket(grounded: int) -> dict:
        return {"grounded": grounded, "tokens": grounded * tpg}

    today_g = _grounded(today_key)
    month_g = sum(_grounded(k) for k in days if k.startswith(month_prefix))
    historic_g = sum(int(d.get("grounded", 0)) for d in days.values())

    # Continuous daily series ending today (gap days filled with 0 for the chart).
    daily = []
    for i in range(days_back - 1, -1, -1):
        d = today - timedelta(days=i)
        key = d.isoformat()
        daily.append({"date": key, **_bucket(_grounded(key))})

    # Monthly series — group ledger days by YYYY-MM, last `months_back` months.
    months: dict[str, int] = {}
    for k, rec in days.items():
        months[_month_of(k)] = months.get(_month_of(k), 0) + int(rec.get("grounded", 0))
    monthly = [
        {"month": m, **_bucket(months[m])} for m in sorted(months)[-months_back:]
    ]

    # Growth: this month vs the immediately preceding calendar month.
    prev_month = (today.replace(day=1) - timedelta(days=1)).strftime("%Y-%m")
    prev_g = months.get(prev_month, 0)
    this_g = months.get(month_prefix, 0)
    if prev_g > 0:
        pct = round((this_g - prev_g) / prev_g * 100, 1)
        up: bool | None = this_g >= prev_g
    else:
        pct = None
        up = None
    growth = {
        "this_month_tokens": this_g * tpg,
        "prev_month_tokens": prev_g * tpg,
        "pct": pct,
        "up": up,
    }

    return {
        "tpg": tpg,
        "today": {"date": today_key, **_bucket(today_g)},
        "month": {"month": month_prefix, **_bucket(month_g)},
        "historic": _bucket(historic_g),
        "daily": daily,
        "monthly": monthly,
        "growth": growth,
        "ledger_path": str(ledger_path(state_dir)),
    }
