"""Shared helpers for the MCP server modules (split from server.py)."""

from __future__ import annotations

from typing import Any

from memo.memory import Memory


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)


def _log_consult(
    memory: Memory,
    *,
    tool: str,
    query: str,
    hits: list[dict[str, Any]],
    t0_ms: int,
    source: str = "",
) -> None:
    """Record an MCP consult into the shared recall ring buffer so memo's
    usefulness is observable for EVERY consumer, not just the Claude Code
    recall-hook (see `memo usefulness`). Best-effort — telemetry must never
    break a tool call."""
    try:
        from memo.dashboard import append_recall_log

        append_recall_log(
            memory.cfg.state_dir,
            prompt=query or "",
            hits=hits or [],
            via=f"mcp:{tool}",
            source=(source or "").strip().lower() or None,
            latency_ms=_now_ms() - t0_ms,
        )
    except Exception:
        pass
