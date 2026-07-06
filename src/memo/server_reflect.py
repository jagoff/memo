"""MCP tool: memo_reflect — synthesize a session transcript into durable memories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from memo.server_annotations import WRITE, annotated_tool

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from memo.memory import Memory


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **WRITE)
    def memo_reflect(
        session_id: str | None = None,
        last: bool = True,
        if_due: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Synthesize a past coding session into durable memories.

        Reads the full session transcript and extracts decisions, facts, bugs, and
        follow-ups as structured memories, then saves a session arc note linking them.

        Args:
            session_id: Specific session id (full or prefix). Omit to use the most recent.
            last: When session_id is omitted, use the most recent session (default True).
            if_due: Skip silently if the session was already reflected.
            dry_run: Preview what would be saved without writing anything.
        """
        from memo.cli_transcripts import _reflect_session
        from memo.session import get_session, list_sessions

        cfg = memory.cfg

        target_id: str | None = session_id
        if not target_id:
            sessions = list_sessions(cfg.state_dir, limit=2)
            if not sessions:
                return {"status": "no_sessions"}
            target_id = sessions[0].get("session_id") or ""

        if not target_id:
            return {"status": "no_session_id"}

        if if_due:
            snap = get_session(cfg.state_dir, target_id)
            if snap and snap.get("reflected_at"):
                return {
                    "status": "already_reflected",
                    "session_id": target_id,
                    "reflected_at": snap["reflected_at"],
                }

        # Thread-safe lazy init — concurrent reflect calls share one wrapper.
        memory._ensure_chat()

        return _reflect_session(target_id, memory, cfg, dry_run=dry_run)
