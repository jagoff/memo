from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

_STATS_SAMPLE_CAP = 1024
_STATS_DEFAULT_PERSIST_INTERVAL_S = 60.0


def _percentile(sorted_values: list[float], pct: int) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


class _DaemonStats:
    def __init__(self, started_at: float, model: str, dims: int) -> None:
        self._started_at = started_at
        self._model = model
        self._dims = dims
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}
        self._errors: dict[str, int] = {}
        self._latencies: dict[str, deque[float]] = {}
        self._last_request_ts: float | None = None

    def record(self, op: str, latency_ms: float, *, error: bool = False) -> None:
        with self._lock:
            self._counts[op] = self._counts.get(op, 0) + 1
            if error:
                self._errors[op] = self._errors.get(op, 0) + 1
            buf = self._latencies.get(op)
            if buf is None:
                buf = deque(maxlen=_STATS_SAMPLE_CAP)
                self._latencies[op] = buf
            buf.append(latency_ms)
            self._last_request_ts = time.time()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            ops: dict[str, dict[str, Any]] = {}
            for op, count in self._counts.items():
                lat = sorted(self._latencies.get(op) or [])
                ops[op] = {
                    "count": count,
                    "errors": self._errors.get(op, 0),
                    "samples": len(lat),
                    "p50_ms": _percentile(lat, 50),
                    "p95_ms": _percentile(lat, 95),
                    "p99_ms": _percentile(lat, 99),
                }
            last_request_ts = self._last_request_ts
            total_requests = sum(self._counts.values())
            total_errors = sum(self._errors.values())
        return {
            "started_at": self._started_at,
            "uptime_s": int(time.time() - self._started_at),
            "model": self._model,
            "dims": self._dims,
            "last_request_ts": last_request_ts,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "ops": ops,
        }


def _stats_file(state_dir: Path) -> Path:
    return state_dir / "embed_daemon_stats.json"


def _stats_persister(
    state_dir: Path,
    stats: _DaemonStats,
    interval_s: float,
    shutdown_event: threading.Event,
) -> None:
    target = _stats_file(state_dir)
    while not shutdown_event.is_set():
        time.sleep(interval_s)
        try:
            snap = stats.snapshot()
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(snap, indent=2))
            tmp.replace(target)
        except (OSError, ValueError, TypeError) as exc:
            from memo.flags import flag_bool

            if flag_bool("MEMO_RECALL_DEBUG"):
                print(f"# recall-daemon: stats persist failed: {exc}", file=sys.stderr)
