"""Tests for daemon_common (shared PID/liveness/cleanup helpers)."""

from __future__ import annotations

import os
from pathlib import Path

from memo.daemon_common import cleanup, daemon_paths, is_pid_alive, read_pid


def test_is_pid_alive_self() -> None:
    assert is_pid_alive(os.getpid()) is True


def test_is_pid_alive_dead() -> None:
    # PID 0 is never a normal user process; os.kill(0, 0) raises → not "alive".
    # Use a very high unlikely PID instead for a clean ProcessLookupError.
    assert is_pid_alive(2_000_000_000) is False


def test_read_pid_missing(tmp_path: Path) -> None:
    assert read_pid(tmp_path / "nope.pid") is None


def test_read_pid_valid(tmp_path: Path) -> None:
    pf = tmp_path / "d.pid"
    pf.write_text("4321\n")
    assert read_pid(pf) == 4321


def test_read_pid_garbage(tmp_path: Path) -> None:
    pf = tmp_path / "d.pid"
    pf.write_text("not-a-number")
    assert read_pid(pf) is None


def test_cleanup_unlinks_missing_ok(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.write_text("x")
    # b doesn't exist — cleanup must not raise.
    cleanup(a, b)
    assert not a.exists()
    assert not b.exists()


def test_daemon_paths_keep_pid_in_state_dir_for_short_paths(tmp_path: Path) -> None:
    state_dir = Path("/tmp") / f"memo-daemon-common-{os.getpid()}"
    sock, pid = daemon_paths(state_dir, "recall")
    assert sock == state_dir / "recall.sock"
    assert pid == state_dir / "recall-daemon.pid"


def test_daemon_paths_use_short_socket_for_deep_state_dir(tmp_path: Path) -> None:
    deep = tmp_path
    for n in range(12):
        deep = deep / f"very-long-directory-name-{n:02d}"

    sock, pid = daemon_paths(deep, "ingest")

    assert pid == deep / "ingest-daemon.pid"
    assert sock.name.endswith(".sock")
    assert sock != deep / "ingest.sock"
    assert len(str(sock)) < 104
