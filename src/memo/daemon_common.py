"""Shared Unix-socket daemon plumbing for memo's recall/ingest/maint daemons.

PID-file + liveness helpers and the serve-until-shutdown loop were copy-pasted
verbatim across the three daemon modules. This consolidates them. Socket/PID
*paths* stay per-daemon (the filenames differ), bound by thin wrappers there.
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any


def is_pid_alive(pid: int) -> bool:
    """Return True if a process with this PID is running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def read_pid(pid_file: Path) -> int | None:
    """Read the PID from `pid_file`. Returns None if missing or invalid."""
    if not pid_file.is_file():
        return None
    try:
        return int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return None


def cleanup(*paths: Path) -> None:
    """Unlink each path, ignoring files that are already gone."""
    for p in paths:
        p.unlink(missing_ok=True)


def serve_until_shutdown(
    server: Any,
    shutdown_event: threading.Event,
    *,
    name: str = "daemon-serve",
    on_shutdown: Callable[[], None] | None = None,
    poll_interval: float = 1.0,
    join_timeout: float = 5.0,
) -> None:
    """Run ``server.serve_forever()`` on a worker thread and block until
    ``shutdown_event`` is set, then shut down in order.

    ``serve_forever()`` runs off the main thread so ``server.shutdown()``
    (called here) never join-deadlocks against it — the signal handler only sets
    the event, avoiding the ungraceful ``os._exit(0)`` the deadlock once forced.
    The ``on_shutdown`` callback (e.g. daemon-specific cleanup) runs last.
    """
    server_thread = threading.Thread(
        target=server.serve_forever, name=name, daemon=True,
    )
    server_thread.start()
    try:
        # Poll so a signal delivered to the main thread is observed promptly
        # even where Event.wait() is not interrupted by the handler.
        while not shutdown_event.wait(timeout=poll_interval):
            pass
    finally:
        with contextlib.suppress(Exception):
            server.shutdown()
        with contextlib.suppress(Exception):
            server.server_close()
        server_thread.join(timeout=join_timeout)
        if on_shutdown is not None:
            on_shutdown()
