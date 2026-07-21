"""`memo recall-daemon restart` — launchd-aware stop/start.

restart sends SIGTERM (via stop) then must decide whether a launchd
KeepAlive already brought the daemon back: if a NEW live PID appears it
defers to launchd (no competing process); otherwise it starts a fresh
daemon. Both branches are exercised without a real subprocess or MLX by
stubbing the recall_server PID helpers and the start command's callback.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

import memo.cli_recall_daemon as crd


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def test_restart_defers_to_launchd_respawn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Old daemon at pid 1000 is alive; SIGTERM (stop) "kills" it and a launchd
    # KeepAlive respawns a new process at pid 2000.
    state = {"killed": False}

    def fake_read_pid(_state_dir: Path) -> int | None:
        return 2000 if state["killed"] else 1000

    def fake_kill(_pid: int, _sig: int) -> None:
        state["killed"] = True

    monkeypatch.setattr("memo.recall_server._read_pid", fake_read_pid)
    monkeypatch.setattr("memo.recall_server._is_pid_alive", lambda _pid: True)
    monkeypatch.setattr("memo.recall_server._cleanup", lambda _state_dir: None)
    monkeypatch.setattr("os.kill", fake_kill)

    started = {"called": False}
    monkeypatch.setattr(
        crd.recall_daemon_start, "callback", lambda: started.__setitem__("called", True)
    )

    result = CliRunner().invoke(crd.recall_daemon_group, ["restart"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "restarted" in result.output
    assert "2000" in result.output
    # launchd handled it — restart must NOT spawn a competing daemon.
    assert started["called"] is False


def test_restart_starts_fresh_when_no_respawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Daemon is not running and nothing respawns it: restart falls through to
    # start. Skip the real 5s respawn wait + the start subprocess.
    monkeypatch.setattr("memo.recall_server._read_pid", lambda _state_dir: None)
    monkeypatch.setattr("memo.recall_server._is_pid_alive", lambda _pid: False)
    monkeypatch.setattr("memo.recall_server._cleanup", lambda _state_dir: None)
    monkeypatch.setattr(crd.time, "sleep", lambda _s: None)

    started = {"called": False}
    monkeypatch.setattr(
        crd.recall_daemon_start, "callback", lambda: started.__setitem__("called", True)
    )

    result = CliRunner().invoke(crd.recall_daemon_group, ["restart"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert started["called"] is True


# -- `recall-daemon start` readiness probe ---------------------------------


def test_start_not_fooled_by_stale_socket_when_child_dies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a stale recall.sock left by a crashed daemon must not make
    `recall-daemon start` report success. The old bare exists() probe trusted
    the file; the connect-based probe + proc.poll fail-fast does not."""
    import subprocess

    from memo.cli import cli
    from memo.recall_socket import _socket_path

    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    stale = _socket_path(state)
    stale.touch()  # leftover from a crash — nothing is listening

    class _DeadChild:
        pid = 4242

        def poll(self) -> int:
            return 1  # child exited immediately (failed to boot)

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _DeadChild())
    monkeypatch.setattr(Path, "home", lambda: tmp_path)  # keep logs out of ~/Library
    monkeypatch.setattr("memo.recall_server._read_pid", lambda _s: None)
    monkeypatch.setattr("memo.recall_server._is_pid_alive", lambda _p: False)
    # A stale socket answers no ping.
    monkeypatch.setattr("memo.recall_server.connect_and_send", lambda *a, **k: None)

    result = CliRunner().invoke(cli, ["recall-daemon", "start"], env=_env(tmp_path))

    assert result.exit_code == 1
    assert "failed to start" in result.stderr


def test_start_succeeds_when_child_answers_ping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The connect-based probe reports success only once the (fake) child
    answers a ping on the freshly bound socket."""
    import subprocess

    from memo.cli import cli

    class _LiveChild:
        pid = 4242

        def poll(self) -> None:
            return None  # still running

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _LiveChild())
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(crd.time, "sleep", lambda _s: None)
    monkeypatch.setattr("memo.recall_server._read_pid", lambda _s: None)
    monkeypatch.setattr("memo.recall_server._is_pid_alive", lambda _p: False)
    monkeypatch.setattr("memo.recall_server.connect_and_send", lambda *a, **k: '{"ok": true}')

    result = CliRunner().invoke(cli, ["recall-daemon", "start"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "started" in result.stderr
