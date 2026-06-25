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
        import uuid
        from pathlib import Path

        from memo.capture import run_capture_incremental
        from memo.session import checkpoint, list_sessions

        cfg = memory.cfg

        # Get the most recent session
        sessions = list_sessions(cfg.state_dir, limit=1)
        if not sessions:
            new_sid = str(uuid.uuid4())
            checkpoint(cfg.state_dir, session_id=new_sid, cwd=str(Path.cwd()))
            sessions = [{"session_id": new_sid, "transcript_path": None, "cwd": str(Path.cwd())}]

        sid = sessions[0].get("session_id")
        if not sid:
            return {"status": "no_session_id", "saved": 0, "saved_titles": []}

        transcript_raw = sessions[0].get("transcript_path")
        if not transcript_raw:
            return {"status": "no_transcript", "saved": 0, "saved_titles": []}
        transcript = Path(transcript_raw)

        if dry_run:
            return {
                "status": "dry_run",
                "session_id": sid,
                "would_capture": True,
                "transcript_path": str(transcript),
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

        # Write pending notification so memo_pop_notification and search/ask
        # tools can surface it to the user. Also print to stderr for clients
        # that display daemon output (opencode, Claude Desktop).
        import contextlib as _contextlib
        with _contextlib.suppress(OSError):
            (cfg.state_dir / "pending_idle_notification.txt").write_text(
                notification + "\n", encoding="utf-8"
            )
        import sys as _sys
        print(notification, file=_sys.stderr)

        return {
            "status": result.get("status"),
            "saved": n,
            "saved_titles": titles,
            "notification": notification,
            "session_id": sid,
        }

    @server.tool()
    def memo_pop_notification() -> str:
        """Read and dismiss pending idle-capture notification.

        Returns the notification text (or empty string if none). The
        notification is deleted after reading, so subsequent calls return
        empty until a new capture runs. Call this periodically (e.g. after
        each subtask) to surface auto-captured insights to the user.
        """
        path = memory.cfg.state_dir / "pending_idle_notification.txt"
        if not path.exists():
            return ""
        try:
            text = path.read_text(encoding="utf-8").strip()
            path.unlink(missing_ok=True)
            return text
        except OSError:
            return ""

    @server.tool()
    def memo_start_session(
        session_id: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Start a new session for this client.

        Call this when beginning a new task or conversation. This enables
        session tracking (auto-capture, checkpoints, grounding).

        Args:
            session_id: Optional session ID (auto-generated if not provided).
            cwd: Current working directory (defaults to current directory).
        """
        import uuid
        from datetime import UTC, datetime

        from memo.session import checkpoint

        if not session_id:
            session_id = str(uuid.uuid4())
        if not cwd:
            import os
            cwd = os.getcwd()

        cfg = memory.cfg
        result = checkpoint(
            cfg.state_dir,
            session_id=session_id,
            cwd=cwd,
            prompt=None,
        )

        return {
            "status": "started",
            "session_id": session_id,
            "cwd": cwd,
            "project": result.get("project"),
            "head_commit": result.get("head_commit", "")[:12],
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
        }

    @server.tool()
    def memo_save_text(
        text: str,
        title: str | None = None,
    ) -> dict[str, Any]:
        """Save a memory from text.

        Use this to save insights, decisions, or important information
        directly without needing a transcript. Works for any LLM client
        (opencode, Claude Desktop, etc).

        Args:
            text: The content to save as a memory.
            title: Optional title (auto-generated from first line if not provided).
        """
        from memo.memory import WriteRefused

        if not text:
            return {"status": "error", "message": "text is required"}

        if not title:
            first_line = text.strip().split("\n")[0][:80]
            title = first_line

        memory._ensure_chat()

        try:
            rec = memory.save(content=text, title=title, type_="note")
        except WriteRefused as exc:
            return {"status": "refused", "conflict": exc.conflict, "message": str(exc)}

        return {
            "status": "saved",
            "saved": 1,
            "title": title,
            "ids": [rec.id],
        }


def run_idle_capture_loop() -> None:
    """Run the idle capture loop in the daemon.

    This is called by memo idle-daemon _serve.
    Runs capture every MEMO_SESSION_IDLE_CAPTURE_SECS (default 10s).
    """
    import logging
    import sys
    import time
    from pathlib import Path

    from memo.capture import run_capture_incremental
    from memo.config import Config
    from memo.flags import flag_int
    from memo.session import list_sessions

    _log = logging.getLogger("memo.idle_daemon")
    _log.setLevel(logging.INFO)
    if not _log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        _log.addHandler(h)

    cfg = Config.from_env()
    delay_secs = flag_int("MEMO_SESSION_IDLE_CAPTURE_SECS") or 10

    _log.info("idle daemon: starting (delay=%ds)", delay_secs)

    log_file = Path(cfg.state_dir / "idle_capture.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            # Get most recent session
            sessions = list_sessions(cfg.state_dir, limit=1)
            if not sessions:
                _log.debug("idle daemon: no sessions")
                time.sleep(delay_secs)
                continue

            sid = sessions[0].get("session_id")
            transcript = sessions[0].get("transcript_path")
            if not sid or not transcript:
                _log.debug("idle daemon: no sid/transcript")
                time.sleep(delay_secs)
                continue

            # Run capture
            result = run_capture_incremental(Path(transcript), sid, debug=False)
            titles = result.get("saved_titles") or []
            n = len(titles)

            # Log result
            ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            if titles:
                shown = "; ".join(t for t in titles[:3])
                if n > 3:
                    shown += f"; +{n - 3} more"
                notif = f"※ auto save (idle): {shown}"
                with open(log_file, "a") as f:
                    f.write(
                        f'{{"ts": "{ts}", "stage": "captured", "sid": "{sid}", '
                        f'"status": "{result.get("status")}", "saved": {n}}}\n'
                    )
                _log.info("idle daemon: captured %d insights", n)
            else:
                notif = "※ auto save (idle): scanned (0 new insights)"
                with open(log_file, "a") as f:
                    f.write(
                        f'{{"ts": "{ts}", "stage": "scanned", "sid": "{sid}", '
                        f'"status": "{result.get("status")}", "saved": 0}}\n'
                    )
                _log.debug("idle daemon: scanned (0 new insights)")

            # Write pending notification for headless clients (opencode, Devin)
            # to read via memo_pop_notification MCP tool.
            (cfg.state_dir / "pending_idle_notification.txt").write_text(
                notif + "\n", encoding="utf-8"
            )

        except Exception as exc:
            _log.error("idle daemon: error: %s", exc)

        time.sleep(delay_secs)