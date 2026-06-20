"""MCP tool: memo_idle_capture — run idle capture on current session."""

from __future__ import annotations

import time
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


def run_idle_capture_loop() -> None:
    """Run the idle capture loop in the daemon.

    This is called by memo idle-daemon _serve.
    Runs capture every MEMO_SESSION_IDLE_CAPTURE_SECS (default 10s).
    """
    import logging
    import sys
    from pathlib import Path

    from memo.cli_capture import run_capture_incremental
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
            with open(log_file, "a") as f:
                if titles:
                    shown = "; ".join(t for t in titles[:3])
                    if n > 3:
                        shown += f"; +{n - 3} more"
                    notif = f"※ auto save (idle): {shown}"
                    f.write(
                        f'{{"ts": "{ts}", "stage": "captured", "sid": "{sid}", '
                        f'"status": "{result.get("status")}", "saved": {n}}}\n'
                    )
                    _log.info("idle daemon: captured %d insights", n)
                    # Also write to pending notification for recall hook
                    (cfg.state_dir / "pending_idle_notification.txt").write_text(
                        notif + "\n", encoding="utf-8"
                    )
                    # Also print to stderr so user sees it in their terminal
                    print(notif, file=sys.stderr)
                else:
                    f.write(
                        f'{{"ts": "{ts}", "stage": "scanned", "sid": "{sid}", '
                        f'"status": "{result.get("status")}", "saved": 0}}\n'
                    )
                    _log.debug("idle daemon: scanned (0 new insights)")

        except Exception as exc:
            _log.error("idle daemon: error: %s", exc)

        time.sleep(delay_secs)