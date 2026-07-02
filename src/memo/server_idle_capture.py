"""MCP tool: memo_idle_capture — run idle capture on current session."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    from fastmcp import FastMCP

    from memo.memory import Memory


def _acquire_start_lock(state_dir: Path) -> int | None:
    """Take the idle daemon's non-blocking exclusive startup flock.

    `idle-daemon start` is check-then-spawn with no child-side guard, and
    `_ensure_idle_daemon` runs it on every memo-mcp startup — so concurrent
    starts would leave a duplicate capture loop running untracked (maint/ingest
    daemons guard their start the same way). Returns the lock fd, held for the
    process lifetime (released by the OS on exit), or None when another
    instance already holds it — the loser must exit at once.
    """
    import fcntl
    import os

    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "idle-daemon.pid.lock"
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(lock_fd)
        return None
    return lock_fd


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
        notification = "※ MEMO auto-saved" if titles else ""

        if titles:
            import contextlib as _contextlib

            with _contextlib.suppress(OSError):
                (cfg.state_dir / "pending_idle_notification.txt").write_text(
                    notification + "\n", encoding="utf-8"
                )
            _log.info("memo_idle_capture: %s", notification)

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

        is_git_clone = (cfg.memory_dir.parent / ".git").exists()

        return {
            "status": "started",
            "session_id": session_id,
            "cwd": cwd,
            "project": result.get("project"),
            "head_commit": result.get("head_commit", "")[:12],
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "needs_cloud_setup": not is_git_clone,
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
    import signal
    import threading
    import time
    from pathlib import Path

    from memo.capture import list_sessions_without_watermark, run_capture_incremental
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

    lock_fd = _acquire_start_lock(cfg.state_dir)
    if lock_fd is None:
        _log.info("idle daemon: another instance is running, exiting")
        return

    delay_secs = flag_int("MEMO_SESSION_IDLE_CAPTURE_SECS") or 10

    _log.info("idle daemon: starting (delay=%ds)", delay_secs)

    # launchd stops the daemon with SIGTERM. Set an Event from the handler and
    # wait on it (instead of bare time.sleep) so a stop interrupts the sleep and
    # exits the loop cleanly between iterations — never mid-write.
    shutdown_event = threading.Event()

    def _sigterm(signum: int, frame: Any) -> None:
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    log_file = Path(cfg.state_dir / "idle_capture.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Rotate log when it exceeds 256 KB — keep the last 200 lines.
    # Without rotation the log grows indefinitely (~820 KB/day at 10s interval).
    _LOG_MAX_BYTES = 256 * 1024
    _LOG_KEEP_LINES = 200

    def _maybe_rotate() -> None:
        try:
            if log_file.exists() and log_file.stat().st_size > _LOG_MAX_BYTES:
                lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
                log_file.write_text("\n".join(lines[-_LOG_KEEP_LINES:]) + "\n", encoding="utf-8")
        except Exception:  # noqa: S110
            pass

    # Track when we last scanned pending sessions (not just the current one).
    # Scan pending sessions once per minute to avoid redundant work.
    _last_pending_scan = 0.0
    _PENDING_SCAN_INTERVAL = 60.0

    while not shutdown_event.is_set():
        try:
            # Get most recent sessions (increase limit to check for pending)
            sessions = list_sessions(cfg.state_dir, limit=5)
            if not sessions:
                _log.debug("idle daemon: no sessions")
                shutdown_event.wait(timeout=delay_secs)
                continue

            sid = sessions[0].get("session_id")
            transcript = sessions[0].get("transcript_path")
            if not sid or not transcript:
                _log.debug("idle daemon: no sid/transcript")
                shutdown_event.wait(timeout=delay_secs)
                continue

            # Run capture on current session
            result = run_capture_incremental(Path(transcript), sid, debug=False)
            titles = result.get("saved_titles") or []
            n = len(titles)

            # After processing current session, check for pending sessions once per minute
            now = time.time()
            if now - _last_pending_scan >= _PENDING_SCAN_INTERVAL:
                _last_pending_scan = now
                pending = list_sessions_without_watermark(cfg.state_dir, sessions, limit=5)
                for pending_sess in pending:
                    pend_sid = pending_sess.get("session_id")
                    pend_transcript = pending_sess.get("transcript_path")
                    if not pend_sid or not pend_transcript:
                        continue
                    # Skip if it's the same as current session (already processed above)
                    if pend_sid == sid:
                        continue
                    pend_result = run_capture_incremental(
                        Path(pend_transcript), pend_sid, debug=False
                    )
                    pend_titles = pend_result.get("saved_titles") or []
                    if pend_titles:
                        _log.info(
                            "idle daemon: captured %d insights from pending session %s",
                            len(pend_titles),
                            pend_sid[:8],
                        )

            # Log result — only write when something was actually captured.
            # Logging every 10s for 0-capture scans is the dominant source of
            # unbounded log growth; debug level is enough for the no-op case.
            ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            if titles:
                _maybe_rotate()
                with open(log_file, "a") as f:
                    f.write(
                        f'{{"ts": "{ts}", "stage": "captured", "sid": "{sid}", '
                        f'"status": "{result.get("status")}", "saved": {n}}}\n'
                    )
                _log.info("idle daemon: captured %d insights", n)
                (cfg.state_dir / "pending_idle_notification.txt").write_text(
                    "※ MEMO auto-saved\n", encoding="utf-8"
                )
            else:
                _log.debug("idle daemon: scanned (0 new insights)")

        except Exception as exc:
            _log.error("idle daemon: error: %s", exc)

        shutdown_event.wait(timeout=delay_secs)

    _log.info("idle daemon: stopping (SIGTERM)")
