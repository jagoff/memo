"""MCP tool: memo_idle_capture — run idle capture on current session."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from memo.memory import Memory


def register(server: FastMCP, memory: Memory) -> None:
    @server.tool()
    def memo_idle_capture(dry_run: bool = False) -> dict[str, Any]:
        """Run idle capture on the current session.

        Extracts insights from the most recent session after a period of inactivity.
        This is useful for agents (like opencode) that don't have Claude Code hooks.

        Call this periodically (e.g., every 30-60s) or after completing a subtask.

        Args:
            dry_run: Preview what would be saved without writing anything.
        """
        from memo.cli_capture import run_capture_incremental
        from memo.session import list_sessions

        cfg = memory.cfg

        # Get the most recent session
        sessions = list_sessions(cfg.state_dir, limit=1)
        if not sessions:
            return {"status": "no_sessions", "saved": 0, "saved_titles": []}

        sid = sessions[0].get("session_id")
        if not sid:
            return {"status": "no_session_id", "saved": 0, "saved_titles": []}

        transcript = sessions[0].get("transcript_path")
        if not transcript:
            return {"status": "no_transcript", "saved": 0, "saved_titles": []}

        if dry_run:
            return {
                "status": "dry_run",
                "session_id": sid,
                "would_capture": True,
                "transcript_path": transcript,
            }

        # Thread-safe lazy init
        memory._ensure_chat()

        # Run capture
        result = run_capture_incremental(
            transcript,
            sid,
            debug=False,
        )

        # Format response
        titles = result.get("saved_titles") or []
        n = len(titles)
        if titles:
            shown = "; ".join(t for t in titles[:3])
            if n > 3:
                shown += f"; +{n - 3} more"
            notification = f"※ auto save (idle): {shown}"
        else:
            notification = "※ auto save (idle): scanned (0 new insights)"

        return {
            "status": result.get("status"),
            "saved": n,
            "saved_titles": titles,
            "notification": notification,
            "session_id": sid,
        }