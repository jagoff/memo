"""Per-request measurement against a real control arm.

memo's existing token meter reads `output_tokens` alone, which is why it cannot
see its own input cost or its effect on the prompt cache. The proxy sits where
the provider's own `usage` is visible, so this module records all four counters
and compares treated requests against an uncompressed holdout.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

LEDGER_SCHEMA = "memo.proxy.requests.v1"

_log = logging.getLogger(__name__)
_HOLDOUT_BUCKETS = 10_000

# Weights that turn the three Messages-API usage counters into one
# "prompt cost" in token-equivalents, from Anthropic's prompt-caching
# pricing (https://docs.claude.com/en/docs/build-with-claude/prompt-caching,
# 5-minute cache tier, the default): an uncached input token bills at the
# base rate (1x); writing a new block into the cache costs a one-time
# CACHE_CREATION_WEIGHT premium; reading a block already in the cache costs
# only CACHE_READ_WEIGHT. `input_tokens` alone is the UNCACHED REMAINDER
# ONLY -- on this machine's real Claude Code traffic (4,938 requests) it
# takes only the values 1 or 2, mean 1.96, about 0.001% of the real prompt
# (total prompt = input_tokens + cache_creation_tokens + cache_read_tokens).
# A ratio over `input_tokens` alone is a ratio of noise: a prefix transform
# (e.g. `toolschemas`, zone=ZONE_PREFIX) edits the cached HEAD and so its
# entire effect lands in the two counters that ratio ignores.
CACHE_CREATION_WEIGHT = 1.25
CACHE_READ_WEIGHT = 0.1


@dataclass
class Record:
    request_key: str
    holdout: bool
    # The real Claude Code session id (or the per-process fallback) the arm
    # was actually assigned on -- see `is_holdout` callers. Optional/blank
    # for older rows and for hand-built test Records that don't care about
    # session identity; `summarize` only counts non-blank values toward the
    # session-level counts below.
    session_key: str = ""
    transforms: list[str] = field(default_factory=list)
    est_saved_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    retrieved: int = 0
    # Per-transform share of est_saved_tokens (from TransformPlan.saved_by) —
    # `transforms` alone cannot support an honest split: it lists every
    # ENABLED transform that ran, whether or not it saved anything.
    saved_by: dict[str, int] = field(default_factory=dict)
    # False marks a request that reached the measurement path without its
    # body actually being rewritten -- e.g. a passthrough recorded while
    # MEMO_PROXY_ENABLED=0 (what `memo proxy off` sets). Such a request is
    # byte-identical to a control request but is NOT part of the holdout
    # sampling scheme either, so it must count toward neither arm (see
    # `summarize`). Defaults to True so the overwhelming majority of
    # callers -- every genuinely-treated request -- never need to say so.
    rewritten: bool = True


def is_holdout(request_key: str, frac: float) -> bool:
    """Stable, unbiased assignment: the same request is always on the same arm."""
    if frac <= 0.0:
        return False
    if frac >= 1.0:
        return True
    digest = hashlib.sha256(request_key.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % _HOLDOUT_BUCKETS
    return bucket < int(frac * _HOLDOUT_BUCKETS)


def usage_from_response(body: dict) -> dict[str, int]:
    usage = body.get("usage") if isinstance(body, dict) else None
    usage = usage if isinstance(usage, dict) else {}

    def _int(key: str) -> int:
        value = usage.get(key)
        return value if isinstance(value, int) else 0

    return {
        "input_tokens": _int("input_tokens"),
        "output_tokens": _int("output_tokens"),
        "cache_creation_tokens": _int("cache_creation_input_tokens"),
        "cache_read_tokens": _int("cache_read_input_tokens"),
    }


def ledger_path(state_dir: Path) -> Path:
    return Path(state_dir) / "proxy" / "requests.jsonl"


def append(state_dir: Path, record: Record) -> None:
    """Append one row. A measurement failure never propagates to a request."""
    path = ledger_path(state_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {"schema": LEDGER_SCHEMA, **asdict(record)}
        with path.open("a", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except Exception:
        _log.warning("proxy: could not append measurement row")


def summarize(state_dir: Path) -> dict:
    """Treated vs holdout on real provider counters. None means 'no data yet'."""
    treated: list[dict] = []
    holdout: list[dict] = []
    passthrough = 0
    skipped = 0
    path = ledger_path(state_dir)
    if path.is_file():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    skipped += 1
                    continue
                if row.get("holdout"):
                    holdout.append(row)
                elif row.get("rewritten", True):
                    treated.append(row)
                else:
                    # Recorded but never actually rewritten (proxy disabled)
                    # -- byte-identical to a control request yet not part of
                    # the holdout sample either; belongs to neither arm.
                    passthrough += 1
            except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
                skipped += 1
                continue

    def _mean(rows: list[dict], key: str) -> float | None:
        if not rows:
            return None
        total = 0
        for r in rows:
            val = r.get(key)
            if isinstance(val, int):
                total += val
        return total / len(rows)

    def _prompt_cost(row: dict) -> float:
        """One row's whole prompt, in token-equivalents -- see the module's
        weight constants for where CACHE_CREATION_WEIGHT/CACHE_READ_WEIGHT
        come from."""

        def _num(key: str) -> int:
            val = row.get(key)
            return val if isinstance(val, int) else 0

        return (
            _num("input_tokens")
            + CACHE_CREATION_WEIGHT * _num("cache_creation_tokens")
            + CACHE_READ_WEIGHT * _num("cache_read_tokens")
        )

    def _mean_cost(rows: list[dict]) -> float | None:
        if not rows:
            return None
        return sum(_prompt_cost(r) for r in rows) / len(rows)

    mean_t = _mean(treated, "input_tokens")
    mean_h = _mean(holdout, "input_tokens")
    mean_cache_creation_t = _mean(treated, "cache_creation_tokens")
    mean_cache_creation_h = _mean(holdout, "cache_creation_tokens")
    mean_cache_read_t = _mean(treated, "cache_read_tokens")
    mean_cache_read_h = _mean(holdout, "cache_read_tokens")
    mean_cost_t = _mean_cost(treated)
    mean_cost_h = _mean_cost(holdout)

    # The headline: a cost ratio over the WHOLE prompt (input + weighted
    # cache creation/read), not `input_tokens` alone -- see the module
    # docstring for why the latter is a ratio of noise.
    saving = None
    if mean_cost_t is not None and mean_cost_h not in (None, 0):
        saving = round((mean_cost_h - mean_cost_t) / mean_cost_h, 6)

    # Per-transform breakdown. `transforms` lists every ENABLED transform a
    # row ran, whether or not it actually saved anything — crediting the
    # row's whole `est_saved_tokens` scalar to every name in that list would
    # inflate `total_saved` by however many transforms ran and report a flat
    # 1/N share for each, real or not. `saved_by` (from TransformPlan) is the
    # honest per-transform split; a name absent from a row's `saved_by`
    # earned nothing from that row, full stop.
    by_transform: dict[str, dict] = {}
    for row in treated:
        names = row.get("transforms") or []
        saved_by = row.get("saved_by")
        saved_by = saved_by if isinstance(saved_by, dict) else {}
        for name in names:
            agg = by_transform.setdefault(name, {"n": 0, "est_saved_tokens": 0})
            agg["n"] += 1
            contrib = saved_by.get(name)
            if isinstance(contrib, int):
                agg["est_saved_tokens"] += contrib

    total_saved = sum(v["est_saved_tokens"] for v in by_transform.values())
    for v in by_transform.values():
        v["share"] = round(v["est_saved_tokens"] / total_saved, 4) if total_saved else None

    retrieved = 0
    for r in treated:
        val = r.get("retrieved")
        if isinstance(val, int):
            retrieved += val

    def _distinct_sessions(rows: list[dict]) -> int:
        """How many DISTINCT sessions an arm represents. Since holdout
        assignment is per-session (defect 2 -- `server.py` calls
        `meter.is_holdout(session_key, ...)`), every request in one holdout
        session lands in the same arm: `len(rows)` (a request count)
        overstates the effective, independent sample size. Rows with no
        `session_key` (hand-built test Records, or a row written before this
        field existed) are excluded rather than folded into a fake shared
        session."""
        return len({r.get("session_key") for r in rows if r.get("session_key")})

    return {
        "n_treated": len(treated),
        "n_holdout": len(holdout),
        "n_treated_sessions": _distinct_sessions(treated),
        "n_holdout_sessions": _distinct_sessions(holdout),
        "n_passthrough": passthrough,
        "mean_input_treated": mean_t,
        "mean_input_holdout": mean_h,
        # Per-arm cache counters. The design doc's cache rule is
        # unfalsifiable without them: they are what actually moved when a
        # prefix transform ran, and `measured_saving_frac` folds them into
        # one ratio (see the module-level weight constants) rather than
        # leaving them invisible.
        "mean_cache_creation_treated": mean_cache_creation_t,
        "mean_cache_creation_holdout": mean_cache_creation_h,
        "mean_cache_read_treated": mean_cache_read_t,
        "mean_cache_read_holdout": mean_cache_read_h,
        "mean_prompt_cost_treated": mean_cost_t,
        "mean_prompt_cost_holdout": mean_cost_h,
        "measured_saving_frac": saving,
        "by_transform": by_transform,
        "retrieved": retrieved,
        "skipped": skipped,
    }
