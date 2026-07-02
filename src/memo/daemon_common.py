"""Shared Unix-socket daemon plumbing for memo's recall/ingest/maint daemons.

PID-file + liveness helpers and the serve-until-shutdown loop were copy-pasted
verbatim across the three daemon modules. This consolidates them. Socket/PID
*paths* stay per-daemon (the filenames differ), bound by thin wrappers there.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

_AF_UNIX_SAFE_PATH_LEN = 103


def daemon_paths(state_dir: Path, name: str) -> tuple[Path, Path]:
    """Return ``(socket_path, pid_file)`` for the daemon called ``name``.

    Single source of the filename convention shared by the recall/ingest/maint
    daemons: ``<name>.sock`` + ``<name>-daemon.pid`` under ``state_dir``. The
    per-daemon ``_socket_path`` / ``_pid_file`` wrappers delegate here.
    """
    return socket_path_for(state_dir, name), state_dir / f"{name}-daemon.pid"


def socket_path_for(state_dir: Path, name: str) -> Path:
    """Return a Unix socket path short enough for macOS AF_UNIX.

    macOS limits sockaddr_un paths to roughly 104 bytes. Pytest and some
    launchd/runtime setups can place ``state_dir`` deep under ``/private/var``;
    in that case keep PID/state files in ``state_dir`` but bind the socket in a
    stable per-user temp directory keyed by the absolute state path.
    """
    local = state_dir / f"{name}.sock"
    if len(str(local)) < _AF_UNIX_SAFE_PATH_LEN:
        return local

    digest = hashlib.sha256(str(state_dir.expanduser()).encode("utf-8")).hexdigest()[:16]
    uid = os.getuid() if hasattr(os, "getuid") else "nouid"
    root = Path(tempfile.gettempdir()) / f"memo-{uid}"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root / f"{name}-{digest}.sock"


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


def cleanup(sock_path: Path, pid_file: Path) -> None:
    """Unlink the daemon's socket + pid file, unless a live OTHER process owns them.

    After a lost startup race the pid file records the SURVIVING daemon, whose
    socket lives at the same path — an orphan's shutdown must not unlink the
    survivor's files. Unlink only when the pid file records OUR pid (normal
    daemon shutdown), a dead pid, or nothing readable at all (stale leftovers,
    safe to sweep — the CLI ``stop`` fallback relies on this).
    """
    owner = read_pid(pid_file)
    if owner is not None and owner != os.getpid() and is_pid_alive(owner):
        return
    sock_path.unlink(missing_ok=True)
    pid_file.unlink(missing_ok=True)


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
        target=server.serve_forever,
        name=name,
        daemon=True,
    )
    # ThreadingMixIn defaults (daemon_threads=False, block_on_close=True) make
    # server_close() join in-flight handler threads UNBOUNDED — one stuck
    # handler would hang SIGTERM shutdown forever and join_timeout would never
    # apply. Daemonic handlers keep shutdown bounded; requests are short-lived.
    server.daemon_threads = True
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
