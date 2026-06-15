from __future__ import annotations

import logging
from typing import Any

from memo.memory import Memory

_log = logging.getLogger("memo.server")


def now_ms() -> int:
    import time

    return int(time.time() * 1000)


def log_consult(
    memory: Memory,
    *,
    tool: str,
    query: str,
    hits: list[dict[str, Any]],
    t0_ms: int,
    source: str = "",
) -> None:
    """Record an MCP consult into the shared recall ring buffer."""
    try:
        from memo.dashboard import append_recall_log

        append_recall_log(
            memory.cfg.state_dir,
            prompt=query or "",
            hits=hits or [],
            via=f"mcp:{tool}",
            source=(source or "").strip().lower() or None,
            latency_ms=now_ms() - t0_ms,
        )
    except Exception as exc:
        _log.warning("consult recall-log write failed for %s: %s", tool, exc)
