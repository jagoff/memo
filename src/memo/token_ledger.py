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

Two physical signals feed the ledger, one per agent class:

  * **grounded** (Claude Code) — a surfaced memory the answer USED, scored by the
    Stop-hook grounding pass over the transcript. Strong signal.
  * **consults** (every other agent: codex/opencode/devin/synapse/memflow/...) —
    a memo search that returned >=1 hit, logged in recall.log with its
    ``source``/``client``. These agents read memo over MCP/CLI/socket and we
    never see their answer, so we cannot ground them; a productive consult is the
    honest proxy for "a re-derivation memo prevented". Weaker signal.

Tokens saved are derived at read time:
``grounded × MEMO_ROI_TOKENS_PER_GROUNDED (350) + consults × MEMO_ROI_TOKENS_PER_CONSULT (200)``.
Storing the raw counts (not the token product) keeps the durable signal physical
and lets the tunable rates re-price history without rewriting the ledger. Consults
are attributed per-source and exclude ``claude-code`` (already counted by its
grounding), so no agent is double-counted.

Pure stdlib + `memo.dashboard` (leaf log readers) + `memo.flags` — no MLX, no
`memo.memory` import, so it is cheap enough to roll up on every Stop hook.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from .dashboard import (
    GROUNDED_SCORE,
    read_grounding_log,
    read_recall_hook_log,
    read_recall_log,
)
from .flags import flag_int

LEDGER_SCHEMA = "memo.token_savings.daily.v1"
_DEFAULT_TOKENS_PER_GROUNDED = 350
_DEFAULT_TOKENS_PER_CONSULT = 200

# Grounded rows are attributed to Claude Code — grounding runs only from Claude
# Code's Stop hook over its transcript, and every grounding.log row confirms
# client="claude-code". Other agents (codex/opencode/devin/synapse/memflow) never
# reach grounding; their savings ride the consult signal below, so their consults
# in recall.log must NOT double-count against a client that also grounds.
_GROUNDED_CLIENT = "claude-code"
_CONSULT_EXCLUDE = frozenset({_GROUNDED_CLIENT})


def ledger_path(state_dir: Path) -> Path:
    return state_dir / "token_savings_daily.json"


def _ledger_lock_path(state_dir: Path) -> Path:
    """Sidecar lock file for the roll_up read-merge-write.

    A sidecar (not the ledger itself) because write_ledger publishes via
    os.replace — an flock on the ledger would ride the OLD inode and stop
    excluding the moment a writer replaced the file.
    """
    return state_dir / "token_savings_daily.json.lock"


def _tokens_per_grounded() -> int:
    v = flag_int("MEMO_ROI_TOKENS_PER_GROUNDED")
    return _DEFAULT_TOKENS_PER_GROUNDED if v is None else v


def _tokens_per_consult() -> int:
    v = flag_int("MEMO_ROI_TOKENS_PER_CONSULT")
    return _DEFAULT_TOKENS_PER_CONSULT if v is None else v


def _consult_source(row: dict) -> str | None:
    """The agent behind a recall.log consult row: ``source`` (CLI/socket path)
    or ``client`` (MCP clientInfo handshake). None when unattributed."""
    src = row.get("source") or row.get("client")
    if not isinstance(src, str):
        return None
    src = src.strip()
    return src or None


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
    """Atomically persist the ledger (unique tmp file + os.replace).

    The tmp name is unique per writer (mkstemp) — a fixed tmp name would let
    one concurrent writer os.replace another's half-written tmp into place,
    truncating the ledger and (via read_ledger's corruption fallback) silently
    resetting the durable history. The tmp is unlinked on failure.
    """
    path = ledger_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(ledger, ensure_ascii=False, indent=2))
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


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


def consults_by_day_client(
    rows: list[dict],
    *,
    to_day: Callable[[str], str | None] = _local_date,
    exclude: frozenset[str] = _CONSULT_EXCLUDE,
) -> dict[str, dict[str, int]]:
    """Count PRODUCTIVE consults (a recall that returned >=1 hit) per local day,
    split by the agent that made them.

    ``{day: {source: count}}``. Empty consults (health-check pings that returned
    no hits) carry no saving and are skipped. ``exclude`` drops clients already
    measured by grounding (Claude Code) so their consults never double-count.
    """
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        if not r.get("hits"):
            continue
        src = _consult_source(r)
        if src is None or src.lower() in exclude:
            continue
        day = to_day(r.get("ts", ""))
        if day is None:
            continue
        per_src = out.setdefault(day, {})
        per_src[src] = per_src.get(src, 0) + 1
    return out


def turns_by_cohort(
    rows: list[dict],
    *,
    to_day: Callable[[str], str | None] = _local_date,
) -> dict[str, dict[str, int]]:
    """Per-local-day recall-hook turn counts split by ablation cohort.

    ``on`` = turns the hook served (daemon/subprocess path); ``off`` = turns
    short-circuited by MEMO_RECALL_DISABLE (via="disabled"). Bail rows carry
    neither and are excluded — they are not comparable turns."""
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        via = r.get("via")
        if via == "disabled":
            cohort = "off"
        elif via in ("daemon", "subprocess"):
            cohort = "on"
        else:
            continue
        day = to_day(r.get("ts", ""))
        if day is None:
            continue
        out.setdefault(day, {"on": 0, "off": 0})[cohort] += 1
    return out


def roll_up(state_dir: Path, *, limit: int = 4000) -> dict:
    """Fold grounded events from grounding.log into the durable ledger.

    Monotonic per day: a day's stored grounded count is ``max(stored, observed)``,
    so eviction from the rolling grounding.log can never shrink a day already
    captured, while a still-in-window day keeps climbing as new rows land.
    Returns the updated ledger.

    Roll-ups run from every session's Stop hook plus `memo tokens`, so an
    exclusive flock around the whole read-merge-write (same pattern as
    dashboard's daily_trend) serializes concurrent writers — otherwise the
    last writer's snapshot clobbers the other's merged days (lost-update).
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    with _ledger_lock_path(state_dir).open("a+", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        ledger = read_ledger(state_dir)
        days: dict = ledger.setdefault("days", {})
        changed = False
        observed = grounded_by_day(read_grounding_log(state_dir, limit=limit))
        for day, n in observed.items():
            prev = int(days.get(day, {}).get("grounded", 0))
            if n >= prev:
                days[day] = {**days.get(day, {}), "grounded": n}
                changed = True
        consults = consults_by_day_client(read_recall_log(state_dir, limit=limit))
        for day, per_src in consults.items():
            cur = days.get(day, {})
            stored: dict = dict(cur.get("consults", {}))
            for src, n in per_src.items():
                if n > int(stored.get(src, 0)):
                    stored[src] = n
            if stored != cur.get("consults", {}):
                days[day] = {**cur, "consults": stored}
                changed = True
        cohorts = turns_by_cohort(read_recall_hook_log(state_dir, limit=limit))
        for day, c in cohorts.items():
            cur = days.get(day, {})
            merged = {
                "turns_on": max(int(cur.get("turns_on", 0)), c["on"]),
                "turns_off": max(int(cur.get("turns_off", 0)), c["off"]),
            }
            if (merged["turns_on"], merged["turns_off"]) != (
                int(cur.get("turns_on", 0)),
                int(cur.get("turns_off", 0)),
            ):
                days[day] = {**cur, **merged}
                changed = True
        if changed:
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
    tpc = _tokens_per_consult()
    today = today or datetime.now().astimezone().date()
    today_key = today.isoformat()
    month_prefix = today.strftime("%Y-%m")

    def _grounded(day_key: str) -> int:
        return int(days.get(day_key, {}).get("grounded", 0))

    def _consults(day_key: str) -> dict[str, int]:
        c = days.get(day_key, {}).get("consults", {})
        return c if isinstance(c, dict) else {}

    def _consults_total(day_key: str) -> int:
        return sum(int(n) for n in _consults(day_key).values())

    def _tokens(grounded: int, consults: int) -> int:
        return grounded * tpg + consults * tpc

    def _bucket(grounded: int, consults: int) -> dict:
        return {
            "grounded": grounded,
            "consults": consults,
            "tokens": _tokens(grounded, consults),
        }

    def _by_client(day_keys: list[str]) -> dict[str, dict]:
        """Per-agent savings over ``day_keys``: grounded → Claude Code (its Stop
        hook is the only grounding path), consults → each consulting agent."""
        agg: dict[str, dict[str, int]] = {}
        g = sum(_grounded(k) for k in day_keys)
        if g:
            agg[_GROUNDED_CLIENT] = {"grounded": g, "consults": 0}
        for k in day_keys:
            for src, n in _consults(k).items():
                e = agg.setdefault(src, {"grounded": 0, "consults": 0})
                e["consults"] += int(n)
        out = {c: {**v, "tokens": _tokens(v["grounded"], v["consults"])} for c, v in agg.items()}
        return dict(sorted(out.items(), key=lambda kv: kv[1]["tokens"], reverse=True))

    month_keys = [k for k in days if k.startswith(month_prefix)]
    all_keys = list(days)

    today_g, today_c = _grounded(today_key), _consults_total(today_key)
    month_g = sum(_grounded(k) for k in month_keys)
    month_c = sum(_consults_total(k) for k in month_keys)
    historic_g = sum(int(d.get("grounded", 0)) for d in days.values())
    historic_c = sum(_consults_total(k) for k in all_keys)

    # Continuous daily series ending today (gap days filled with 0 for the chart).
    daily = []
    for i in range(days_back - 1, -1, -1):
        d = today - timedelta(days=i)
        key = d.isoformat()
        daily.append({"date": key, **_bucket(_grounded(key), _consults_total(key))})

    # Monthly series — group ledger days by YYYY-MM, last `months_back` months.
    months: dict[str, dict[str, int]] = {}
    for k, rec in days.items():
        m = months.setdefault(_month_of(k), {"grounded": 0, "consults": 0})
        m["grounded"] += int(rec.get("grounded", 0))
        m["consults"] += _consults_total(k)
    monthly = [
        {"month": m, **_bucket(months[m]["grounded"], months[m]["consults"])}
        for m in sorted(months)[-months_back:]
    ]

    # Growth: this month's total savings vs the preceding calendar month.
    prev_month = (today.replace(day=1) - timedelta(days=1)).strftime("%Y-%m")
    prev = months.get(prev_month, {"grounded": 0, "consults": 0})
    this = months.get(month_prefix, {"grounded": 0, "consults": 0})
    prev_tok = _tokens(prev["grounded"], prev["consults"])
    this_tok = _tokens(this["grounded"], this["consults"])
    if prev_tok > 0:
        pct = round((this_tok - prev_tok) / prev_tok * 100, 1)
        up: bool | None = this_tok >= prev_tok
    else:
        pct = None
        up = None
    growth = {
        "this_month_tokens": this_tok,
        "prev_month_tokens": prev_tok,
        "pct": pct,
        "up": up,
    }

    ablation = {
        "turns_on": sum(int(d.get("turns_on", 0)) for d in days.values()),
        "turns_off": sum(int(d.get("turns_off", 0)) for d in days.values()),
    }

    return {
        "tpg": tpg,
        "tpc": tpc,
        "today": {"date": today_key, **_bucket(today_g, today_c)},
        "month": {"month": month_prefix, **_bucket(month_g, month_c)},
        "historic": _bucket(historic_g, historic_c),
        "by_client": {
            "today": _by_client([today_key]),
            "month": _by_client(month_keys),
            "historic": _by_client(all_keys),
        },
        "daily": daily,
        "monthly": monthly,
        "growth": growth,
        "ablation": ablation,
        "ledger_path": str(ledger_path(state_dir)),
    }
