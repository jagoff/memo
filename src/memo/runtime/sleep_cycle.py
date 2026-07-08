"""Autonomous background maintenance (Sleep Cycle) for `Memory`.

Monitors system idle time and runs heavy knowledge synthesis/consolidation
tasks when the system is not in use.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from memo.config import Config
from memo.flags import flag_bool, flag_int
from memo.memory import Memory

_log = logging.getLogger(__name__)


def run_sleep_cycle(debug: bool = False) -> None:
    """Run the sleep cycle loop.

    Fires periodically and checks if the system has been idle (no recall or
    writes) for at least MEMO_MAINT_IDLE_THRESHOLD_SECS.
    """
    cfg = Config.from_env()
    mem = Memory(cfg)

    interval_flag = flag_int("MEMO_MAINT_SLEEP_CYCLE_INTERVAL")
    threshold_flag = flag_int("MEMO_MAINT_IDLE_THRESHOLD_SECS")
    interval = 3600 if interval_flag is None else interval_flag
    idle_threshold = 300 if threshold_flag is None else threshold_flag

    if debug:
        print(
            f"# Sleep cycle: interval={interval}s, idle_threshold={idle_threshold}s",
            file=sys.stderr,
        )

    stop = threading.Event()

    def _sigterm(signum: int, frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    while not stop.is_set():
        last_activity = _get_last_activity(mem, cfg)
        idle_secs = time.time() - last_activity

        if idle_secs >= idle_threshold:
            if debug:
                print(
                    f"# Sleep cycle: system idle ({idle_secs:.0f}s), starting maintenance",
                    file=sys.stderr,
                )

            try:
                # 0. Eager Synthesis: Ingest active/recent memflow sessions
                if flag_bool("MEMO_SYNC_MEMFLOW_ENABLED"):
                    _ingest_memflow_sessions(mem, cfg, debug=debug)

                # 1. Synthesis: Generate emergent insights from clusters.
                # Use dry_run=False to actually save the synthesis notes.
                mem.synthesize_cross_cluster()

                # 2. Consolidation: Propose merges for near-duplicates.
                # Currently only returns proposals; in the future this could
                # auto-apply high-confidence merges.
                mem.consolidate()

                if debug:
                    print("# Sleep cycle: maintenance pass complete", file=sys.stderr)
            except Exception as exc:
                # Keep the daemon alive across a failed pass, but always surface
                # the failure (a debug-gated print silently lost it before).
                _log.warning("Sleep cycle: maintenance failed: %s", exc)
        else:
            if debug:
                print(
                    f"# Sleep cycle: system busy (idle {idle_secs:.0f}s < {idle_threshold}s), skipping",
                    file=sys.stderr,
                )

        # stop.wait() returns immediately on SIGTERM/SIGINT (PEP 475 makes a
        # bare time.sleep swallow the signal until the full interval elapses).
        stop.wait(timeout=interval)
    mem.close()


def _ingest_memflow_sessions(mem: Memory, cfg: Config, debug: bool = False) -> None:
    """Walk .memflow/sessions and run reflection on eligible transcripts."""
    from memo.flags import flag_str
    from memo.session import list_sessions

    memflow_dir = Path(flag_str("MEMO_MEMFLOW_DIR") or ".memflow").expanduser()
    if not memflow_dir.is_dir():
        if debug:
            print(f"# Sleep cycle: memflow dir not found: {memflow_dir}", file=sys.stderr)
        return

    # Use memo's session listing to find sessions that need reflection
    # We look for sessions that haven't been reflected yet.
    sessions = list_sessions(cfg.state_dir, limit=5)
    from memo.cli_transcripts import _reflect_session

    for snap in sessions:
        sid = snap.get("session_id")
        if not sid or snap.get("reflected_at"):
            continue

        if debug:
            print(f"# Sleep cycle: reflecting on session {sid[:8]}", file=sys.stderr)

        res = _reflect_session(sid, mem, cfg, debug=debug)
        if debug:
            status = res.get("status")
            saved = len(res.get("saved") or [])
            print(
                f"# Sleep cycle: session {sid[:8]} reflect status={status} saved={saved}",
                file=sys.stderr,
            )


def _get_last_activity(mem: Memory, cfg: Config) -> float:
    """Estimate last activity time (recall or write)."""
    last_activity = 0.0

    # 1. Check last recall log entry (user interaction)
    try:
        log_path = cfg.state_dir / "recall.log"
        if log_path.exists():
            last_activity = max(last_activity, log_path.stat().st_mtime)
    except Exception:  # noqa: S110
        pass

    # 2. Check last write in the store (durable updates)
    try:
        recent = mem.store.list_recent(limit=1)
        if recent:
            dt = datetime.fromisoformat(recent[0]["updated"].replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            last_activity = max(last_activity, dt.timestamp())
    except Exception:  # noqa: S110
        pass

    return last_activity
