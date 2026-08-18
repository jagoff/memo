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


@dataclass
class Record:
    request_key: str
    holdout: bool
    transforms: list[str] = field(default_factory=list)
    est_saved_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    retrieved: int = 0


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
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except Exception:
        _log.warning("proxy: could not append measurement row")


def summarize(state_dir: Path) -> dict:
    """Treated vs holdout on real provider counters. None means 'no data yet'."""
    treated: list[dict] = []
    holdout: list[dict] = []
    skipped = 0
    path = ledger_path(state_dir)
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                skipped += 1
                continue
            (holdout if row.get("holdout") else treated).append(row)

    def _mean(rows: list[dict], key: str) -> float | None:
        if not rows:
            return None
        return sum(int(r.get(key) or 0) for r in rows) / len(rows)

    mean_t = _mean(treated, "input_tokens")
    mean_h = _mean(holdout, "input_tokens")
    saving = None
    if mean_t is not None and mean_h not in (None, 0):
        saving = round((mean_h - mean_t) / mean_h, 6)

    by_transform: dict[str, int] = {}
    for row in treated:
        for name in row.get("transforms") or []:
            by_transform[name] = by_transform.get(name, 0) + 1

    return {
        "n_treated": len(treated),
        "n_holdout": len(holdout),
        "mean_input_treated": mean_t,
        "mean_input_holdout": mean_h,
        "measured_saving_frac": saving,
        "by_transform": by_transform,
        "retrieved": sum(int(r.get("retrieved") or 0) for r in treated),
        "skipped": skipped,
    }
