"""Read-only diagnostics for disabled legacy terminal registrations."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from memo.flags import flag_bool
from memo.server_annotations import READ_ONLY, annotated_tool
from memo.terminal_live import TerminalBridge
from memo.terminal_receiver import ReceiverClient, read_capability_file


def register(server: Any, memory: Any) -> None:
    """Register terminal status diagnostics on every MCP surface profile."""

    @annotated_tool(server, **READ_ONLY)
    def memo_terminal_list() -> dict[str, Any]:
        """List deliverable terminals; empty while legacy TTY input is disabled."""
        rows = [asdict(item) for item in TerminalBridge(memory.cfg).list()]
        return {"terminals": rows, "count": len(rows)}

    if not flag_bool("MEMO_TERMINAL_RECEIVER_ENABLED"):
        return

    @annotated_tool(server)
    def memo_terminal_receiver_send(
        socket: str,
        capability_file: str,
        message_id: str,
        message: str,
        submit: bool = True,
    ) -> dict[str, Any]:
        """Send through an authenticated receiver-bound PTY session."""
        return ReceiverClient(socket, read_capability_file(capability_file)).send(
            message_id=message_id,
            text=message,
            submit=submit,
        )

    @annotated_tool(server)
    def memo_terminal_receiver_enter(
        socket: str,
        capability_file: str,
        message_id: str,
    ) -> dict[str, Any]:
        """Press Return through an authenticated receiver-bound PTY session."""
        return ReceiverClient(socket, read_capability_file(capability_file)).enter(
            message_id=message_id
        )
