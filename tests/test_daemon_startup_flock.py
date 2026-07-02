"""Startup-flock guards for the recall + idle daemons.

Two concurrent daemon starts can both pass the dead-PID check, then race
`sock_path.unlink` + bind — one unlinks the socket the other just bound,
orphaning a live daemon. maint/ingest guard their start with a non-blocking
exclusive flock; these tests pin the same guard on the recall daemon
(`recall_socket.run_server`) and the idle capture loop
(`server_idle_capture.run_idle_capture_loop`). Holding the lock in-test
works because flock conflicts across distinct fds even within one process.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

import pytest

from memo.daemon_common import read_pid


def _hold_lock(lock_path: Path) -> int:
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


# -- recall daemon ----------------------------------------------------------


def test_recall_run_server_exits_when_start_lock_held(tmp_path: Path, monkeypatch) -> None:
    from memo import recall_socket

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(state_dir))

    sock_path = recall_socket._socket_path(state_dir)
    pid_file = recall_socket._pid_file(state_dir)
    # A stale (dead) pid + a leftover socket: WITHOUT the flock, run_server
    # would pass the liveness check and unlink both. With the lock held by a
    # concurrent starter, it must exit(0) before touching either.
    sock_path.write_text("")
    pid_file.write_text("2000000000")

    # If run_server gets past the lock it would build Memory (multi-second MLX
    # init) and serve forever — fail fast instead.
    def _no_memory(cfg):  # pragma: no cover - only hit on regression
        raise SystemExit("run_server proceeded past the startup lock")

    monkeypatch.setattr("memo.memory.Memory", _no_memory)

    lock_fd = _hold_lock(pid_file.with_name(pid_file.name + ".lock"))
    try:
        with pytest.raises(SystemExit) as exc:
            recall_socket.run_server(state_dir)
        assert exc.value.code == 0
        # The loser must not have unlinked the (would-be survivor's) files.
        assert sock_path.exists()
        assert read_pid(pid_file) == 2000000000
    finally:
        os.close(lock_fd)
        sock_path.unlink(missing_ok=True)


# -- idle capture daemon ------------------------------------------------------


def test_idle_start_lock_is_exclusive_and_releasable(tmp_path: Path) -> None:
    from memo.server_idle_capture import _acquire_start_lock

    fd = _acquire_start_lock(tmp_path)
    assert fd is not None
    try:
        # Second concurrent start loses.
        assert _acquire_start_lock(tmp_path) is None
    finally:
        os.close(fd)

    # Released (daemon exited) → a fresh start can acquire again.
    fd2 = _acquire_start_lock(tmp_path)
    assert fd2 is not None
    os.close(fd2)


def test_idle_capture_loop_exits_when_lock_held(tmp_path: Path, monkeypatch) -> None:
    from memo import server_idle_capture

    state_dir = tmp_path / "state"
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(state_dir))

    # If the loop gets past the lock it would run forever — fail fast instead
    # (SystemExit is not caught by the loop's `except Exception`).
    def _boom(*args, **kwargs):  # pragma: no cover - only hit on regression
        raise SystemExit("run_idle_capture_loop ran past the startup lock")

    monkeypatch.setattr("memo.session.list_sessions", _boom)

    lock_fd = server_idle_capture._acquire_start_lock(state_dir)
    assert lock_fd is not None
    try:
        # Duplicate child: must return immediately, not raise, not loop.
        server_idle_capture.run_idle_capture_loop()
    finally:
        os.close(lock_fd)
