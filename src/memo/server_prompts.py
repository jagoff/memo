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


def register(server: Any, memory: Any) -> None:
    """Register the briefing/recall prompts (all profiles — prompts ≠ tools)."""

    @server.prompt(
        name="briefing",
        description="Load memo's startup briefing (durable facts, decisions, "
        "operational state) into the conversation.",
    )
    def briefing() -> str:
        # cwd=None: a no-argument prompt has no cwd, so operational lines are
        # unbiased by project.
        from memo.briefing import compose_unified_briefing

        t0 = now_ms()
        try:
            text = compose_unified_briefing(memory, None)
        except Exception as exc:
            _log.debug("briefing prompt failed", exc_info=True)
            return f"memo unavailable: {type(exc).__name__}"
        with suppress(Exception):
            log_consult(
                memory, tool="briefing", query="briefing", hits=[], t0_ms=t0, source=_SOURCE
            )
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
            log_consult(
                memory,
                tool="recall",
                query=topic,
                hits=[h.to_dict() for h in hits],
                t0_ms=t0,
                source=_SOURCE,
            )
        header = f"Context from memo (recall: {topic}):\n"
        if not hits:
            return header + "(no matching memories)"
        lines = []
        for h in hits:
            body = (h.body or "")[:400]
            lines.append(f"[{h.id[:8]}] {h.title} ({h.type})\n{body}")
        return header + "\n---\n".join(lines)
