"""MCP prompts — pinnable ambient recall for MCP-only clients.

Claude Code gets ambient recall via hooks; every other MCP client gets a
passive toolbox. These prompts give those clients a one-click `briefing`
(session-start context) and `recall <topic>` (on-demand recall block).
Fail-open: a broken index degrades to a one-line message, never an error.
See docs/SPECS/2026-07-28-mcp-ambient-recall-design.md.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

from memo.server_common import log_consult, now_ms

_log = logging.getLogger(__name__)

_SOURCE = "mcp-prompt"


def _briefing_text(memory: Any) -> str:
    """Compose the compact briefing text.

    Transcribed from `memo_unified_briefing` (`server_core_search.py`) — same
    `memo.briefing` calls and flag reads — minus the tool's dict envelope and
    consult logging (the prompt logs its own consult). `cwd` is not available
    to a no-argument prompt, so operational lines are unbiased by project.
    """
    from memo.briefing import (
        compact_text,
        memo_native_briefing_lines,
        operational_briefing_lines,
    )
    from memo.flags import flag_int

    loops_n = max(1, flag_int("MEMO_BRIEFING_LOOPS_N") or 5)
    loops_days = max(1, flag_int("MEMO_BRIEFING_LOOPS_DAYS") or 7)
    raw_lines: list[str] = memo_native_briefing_lines(
        memory, loops_n=loops_n, loops_days=loops_days
    )
    raw_lines.extend(operational_briefing_lines(memory, None))
    return compact_text("\n".join(raw_lines), max_chars=900)


def register(server: Any, memory: Any) -> None:
    """Register the briefing/recall prompts (all profiles — prompts ≠ tools)."""

    @server.prompt(
        name="briefing",
        description="Load memo's startup briefing (durable facts, decisions, "
        "operational state) into the conversation.",
    )
    def briefing() -> str:
        t0 = now_ms()
        try:
            text = _briefing_text(memory)
        except Exception as exc:
            _log.debug("briefing prompt failed", exc_info=True)
            return f"memo unavailable: {type(exc).__name__}"
        with suppress(Exception):
            log_consult(memory, tool="briefing", query="", hits=[], t0_ms=t0, source=_SOURCE)
        return "Context from memo (briefing):\n" + text

    @server.prompt(
        name="recall",
        description="Pull memo's most relevant memories for a topic into the conversation.",
    )
    def recall(topic: str) -> str:
        t0 = now_ms()
        try:
            hits = memory.search(topic, limit=5, mode="hybrid")
        except Exception as exc:
            _log.debug("recall prompt failed", exc_info=True)
            return f"memo unavailable: {type(exc).__name__}"
        with suppress(Exception):
            log_consult(memory, tool="recall", query=topic, hits=hits, t0_ms=t0, source=_SOURCE)
        header = f"Context from memo (recall: {topic}):\n"
        if not hits:
            return header + "(no matching memories)"
        lines = []
        for h in hits:
            body = (getattr(h, "body", "") or "")[:400]
            lines.append(f"[{h.id[:8]}] {h.title} ({h.type})\n{body}")
        return header + "\n---\n".join(lines)
