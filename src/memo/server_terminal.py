"""Read-only diagnostics for disabled legacy terminal registrations."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from memo.server_annotations import READ_ONLY, annotated_tool
from memo.terminal_live import TerminalBridge


def register(server: Any, memory: Any) -> None:
    """Register terminal status diagnostics on every MCP surface profile."""

    @annotated_tool(server, **READ_ONLY)
    def memo_terminal_list() -> dict[str, Any]:
        """List deliverable terminals; empty while legacy TTY input is disabled."""
        rows = [asdict(item) for item in TerminalBridge(memory.cfg).list()]
        return {"terminals": rows, "count": len(rows)}
