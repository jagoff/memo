"""Recall-hook latency metrics — cheap jsonl stamping + percentile summaries.

The recall hook appends ONE line per execution (BOTH paths: warm daemon socket
and cold subprocess fallback — instrumenting only one would make the data lie)
to ``state_dir/recall_metrics.jsonl``::

    {"ts": "<iso8601>", "total_ms": 123.4, "path": "daemon", "hits": 3}

Constraints (UserPromptSubmit hook path, 5s budget):

- stamping is open-append-close (O_APPEND semantics, no locks);
- rotation is stat-gated: the file is only re-read once its size suggests
  ~``MAX_LINES`` entries, then atomically rewritten (tmp + ``os.replace``)
  keeping the newest ``KEEP_LINES``;
- everything degrades silently — a failed stamp must NEVER break the hook;
- stdlib-only imports, nothing heavy.

``memo stats`` reads the file back via :func:`summarize` (p50/p95/p99 of
``total_ms`` split by path over a trailing window). Known issue this
instrumentation exists to surface: daemon tail latency up to ~53s from lock
contention, previously invisible.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

_log = logging.getLogger(__name__)

MAX_LINES = 5000
KEEP_LINES = 2500
# A metrics line is ~90 bytes; re-read (count) the file only once st_size
# suggests we are near MAX_LINES. One os.stat per stamp, no full read.
_SIZE_TRIP_BYTES = 90 * MAX_LINES

# ``build_system_message`` renders "🧠 memo · N: titles" — N == hit count.
_SYSMSG_COUNT_RE = re.compile(r"·\s*(\d+):")
# Short-id citations rendered per hit: "[a1b2c3d4] title".
_HIT_ID_RE = re.compile(r"\[([0-9a-f]{8})\]")
# Literal example id inside recall_logic.CITE_INSTRUCTION — not a real hit.
_CITE_PLACEHOLDER = "a1b2c3d4"


def metrics_path(state_dir: Path) -> Path:
    return state_dir / "recall_metrics.jsonl"


def stamp(state_dir: Path, *, total_ms: float, path: str, hits: int) -> None:
    """Append one metrics line. Silent on ANY failure — the hook must survive."""
    try:
        from memo.flags import flag_bool

        if not flag_bool("MEMO_RECALL_METRICS"):
            return
        target = metrics_path(state_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "total_ms": round(float(total_ms), 1),
            "path": path,
            "hits": int(hits),
        }
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        _maybe_rotate(target)
    except Exception as exc:  # broad by design — never break the hook
        _log.debug("recall metrics stamp failed: %s", exc)


def _maybe_rotate(
    target: Path,
    *,
    max_lines: int = MAX_LINES,
    keep_lines: int = KEEP_LINES,
    size_trip_bytes: int = _SIZE_TRIP_BYTES,
) -> None:
    """Trim to the newest ``keep_lines`` once past ~``max_lines`` entries.

    Cheap: a single ``os.stat`` per call; the file is read (and atomically
    rewritten) only when its size trips the threshold AND the line count
    actually exceeds ``max_lines``.
    """
    try:
        if target.stat().st_size < size_trip_bytes:
            return
        lines = target.read_text(encoding="utf-8").splitlines()
        if len(lines) <= max_lines:
            return
        tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        tmp.write_text("\n".join(lines[-keep_lines:]) + "\n", encoding="utf-8")
        os.replace(tmp, target)
    except Exception as exc:
        _log.debug("recall metrics rotate failed: %s", exc)


def count_hits(hook_output: str) -> int:
    """Best-effort hit count from a daemon-path hook output JSON string.

    The daemon returns the final hook JSON opaquely, so the client can't see
    ``len(relevant)`` directly. Prefer the systemMessage counter ("🧠 memo ·
    N: …"); fall back to counting unique short-id citations in the injected
    context (approximate: associative-nudge ids also match). 0 on anything
    unparseable or empty.
    """
    try:
        parsed = json.loads(hook_output)
        if not isinstance(parsed, dict):
            return 0
        hook_specific = parsed.get("hookSpecificOutput")
        if not isinstance(hook_specific, dict):
            return 0
        sysmsg = parsed.get("systemMessage")
        if isinstance(sysmsg, str):
            m = _SYSMSG_COUNT_RE.search(sysmsg)
            if m:
                return int(m.group(1))
        context = hook_specific.get("additionalContext")
        if not isinstance(context, str):
            return 0
        ids = set(_HIT_ID_RE.findall(context))
        ids.discard(_CITE_PLACEHOLDER)
        return len(ids)
    except Exception:
        return 0


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile (``pct`` in 0–100). 0.0 on an empty sequence."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil((pct / 100.0) * len(ordered))
    return ordered[min(max(rank, 1), len(ordered)) - 1]


def summarize(state_dir: Path, *, days: int = 7) -> dict[str, dict[str, float | int]]:
    """p50/p95/p99/count of ``total_ms`` per path over the trailing window.

    Returns ``{}`` when the file is missing/empty or has no recent entries —
    ``memo stats`` omits the section in that case. Malformed lines are skipped.
    """
    target = metrics_path(state_dir)
    if not target.is_file():
        return {}
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _log.debug("recall metrics read failed: %s", exc)
        return {}
    cutoff = datetime.now(UTC) - timedelta(days=days)
    by_path: dict[str, list[float]] = {}
    for line in lines:
        try:
            entry = json.loads(line)
            ts = datetime.fromisoformat(str(entry["ts"]))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if ts < cutoff:
                continue
            by_path.setdefault(str(entry["path"]), []).append(float(entry["total_ms"]))
        except Exception:  # noqa: S112  # malformed lines are expected noise — skip
            continue
    return {
        path: {
            "count": len(vals),
            "p50": percentile(vals, 50),
            "p95": percentile(vals, 95),
            "p99": percentile(vals, 99),
        }
        for path, vals in by_path.items()
        if vals
    }
