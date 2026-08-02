"""MCP tools for immediate same-user delivery to registered agent terminals."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Annotated, Any

from pydantic import Field

from memo.errors import MemoError
from memo.flags import flag_str
from memo.server_annotations import READ_ONLY, WRITE, annotated_tool
from memo.terminal_live import TerminalBridge


def _reply_envelope(message: str, sender_id: str) -> str:
    encoded = json.dumps(message, ensure_ascii=False).replace("<", "\\u003c").replace(
        ">", "\\u003e"
    )
    if not sender_id:
        return (
            '<memo-live-message>\n<sender-content encoding="json">\n'
            f"{encoded}\n"
            "</sender-content>\n</memo-live-message>"
        )
    return (
        f'<memo-live-message sender="{sender_id}">\n'
        "Reply live with memo_terminal_send(to="
        f'"{sender_id}", message="<your reply>").\n'
        '<sender-content encoding="json">\n'
        f"{encoded}\n"
        "</sender-content>\n"
        "</memo-live-message>"
    )


def _sender_id(bridge: TerminalBridge, requested: str) -> str:
    if requested.strip():
        return bridge.registration_id(requested.strip())
    inherited_tty = flag_str("MEMO_AGENT_TTY").strip()
    return bridge.registration_id_for_tty(inherited_tty) if inherited_tty else ""


def register(server: Any, memory: Any) -> None:
    """Register live-terminal tools on every MCP surface profile."""

    @annotated_tool(server, **READ_ONLY)
    def memo_terminal_list() -> dict[str, Any]:
        """List live local agent terminals that can receive an immediate prompt."""
        rows = [asdict(item) for item in TerminalBridge(memory.cfg).list()]
        return {"terminals": rows, "count": len(rows)}

    @annotated_tool(server, **WRITE)
    def memo_terminal_send(
        to: Annotated[str, Field(description="Exact terminal id returned by memo_terminal_list.")],
        message: Annotated[str, Field(description="Prompt or reply to type into the target agent.")],
        submit: Annotated[
            bool,
            Field(description="Press Return after typing so the target agent receives the prompt."),
        ] = True,
        message_id: Annotated[
            str | None,
            Field(description="Optional idempotency key; retries never type twice."),
        ] = None,
        sender: Annotated[
            str,
            Field(description="Optional registered reply-to terminal id."),
        ] = "",
    ) -> dict[str, Any]:
        """Immediately type and optionally submit a prompt in one registered terminal."""
        bridge = TerminalBridge(memory.cfg)
        try:
            sender_id = _sender_id(bridge, sender)
            receipt = bridge.send(
                to,
                _reply_envelope(message, sender_id),
                sender=sender_id or None,
                submit=submit,
                message_id=message_id,
            )
        except MemoError as exc:
            return {"status": "failed", "error": str(exc)}
        return asdict(receipt)

    @annotated_tool(server, **WRITE)
    def memo_terminal_enter(
        to: Annotated[str, Field(description="Exact terminal id returned by memo_terminal_list.")],
        message_id: Annotated[
            str | None,
            Field(description="Optional idempotency key; retries never press Return twice."),
        ] = None,
        sender: Annotated[
            str,
            Field(description="Optional registered origin terminal id."),
        ] = "",
    ) -> dict[str, Any]:
        """Press Return in one registered foreground agent terminal to unblock it."""
        bridge = TerminalBridge(memory.cfg)
        try:
            sender_id = _sender_id(bridge, sender)
            receipt = bridge.enter(to, sender=sender_id or None, message_id=message_id)
        except MemoError as exc:
            return {"status": "failed", "error": str(exc)}
        return asdict(receipt)
