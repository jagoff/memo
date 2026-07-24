"""MCP profile surface built on Memo's existing durable state."""

from __future__ import annotations

from typing import Any

from memo.memory import Memory
from memo.memory_profile import build_memory_profile
from memo.server_annotations import READ_ONLY, annotated_tool
from memo.server_common import log_consult, now_ms


def register(server: Any, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_profile(
        scope: str = "current",
        limit: int = 8,
        budget_chars: int = 4000,
        cwd: str | None = None,
        source: str = "",
    ) -> dict[str, Any]:
        """Return bounded stable and active memory with evidence metadata."""
        t0 = now_ms()
        try:
            payload = build_memory_profile(
                memory,
                scope=scope,
                limit=limit,
                budget_chars=budget_chars,
                cwd=cwd,
            )
        except ValueError as exc:
            return {
                "schema": "memo.error.v1",
                "error": {"code": "invalid_scope", "message": str(exc)},
                "scope": scope,
            }
        hits = [
            item
            for section in ("stable", "active")
            for item in payload.get(section, [])
            if isinstance(item, dict)
        ]
        log_consult(memory, tool="profile", query=scope, hits=hits, t0_ms=t0, source=source)
        return payload
