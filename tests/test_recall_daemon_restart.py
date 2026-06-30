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
