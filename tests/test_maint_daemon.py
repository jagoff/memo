"""Maintenance daemon + client + flag-gated fallback.

Runs a real `_MaintServer` on a tmp Unix socket with an INJECTED runner, so
the daemon path is exercised without loading MLXChat. Also asserts graceful
degradation: flag off, or daemon down -> `Memory.consolidate` runs in-process.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from memo import maint_client
from memo.maint_server import _MaintServer, _socket_path


@pytest.fixture
def maint_daemon(tmp_path: Path) -> Iterator[tuple[Path, list]]:
    calls: list[tuple[str, dict]] = []

    def fake_runner(op: str, params: dict) -> dict:
        calls.append((op, params))
        return {"proposals": [{"cluster": 1, "summary": "merged"}]}

    server = _MaintServer(str(_socket_path(tmp_path)), fake_runner)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        yield tmp_path, calls
    finally:
        server.shutdown()
        server.server_close()


def test_ping(maint_daemon: tuple[Path, list]) -> None:
    state_dir, _ = maint_daemon
    resp = maint_client.ping(state_dir=state_dir)
    assert resp is not None and resp["ok"] is True and resp["kind"] == "maint"


def test_consolidate_returns_proposals(maint_daemon: tuple[Path, list]) -> None:
    state_dir, calls = maint_daemon
    proposals = maint_client.consolidate(threshold=0.9, max_clusters=10, state_dir=state_dir)
    assert proposals == [{"cluster": 1, "summary": "merged"}]
    assert calls == [("consolidate", {"threshold": 0.9, "max_clusters": 10, "type_": None})]


def test_runner_error_returns_none(tmp_path: Path) -> None:
    def boom(op: str, params: dict) -> dict:
        raise RuntimeError("LLM OOM")

    server = _MaintServer(str(_socket_path(tmp_path)), boom)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        assert maint_client.consolidate(state_dir=tmp_path) is None  # error -> None -> fallback
    finally:
        server.shutdown()
        server.server_close()


def test_client_returns_none_when_daemon_absent(tmp_path: Path) -> None:
    assert maint_client.ping(state_dir=tmp_path) is None
    assert maint_client.consolidate(state_dir=tmp_path) is None


# -- `maint-daemon start` readiness probe ----------------------------------


def _start_env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path),
    }


def test_start_fails_on_stale_socket_when_child_dies(tmp_path: Path, monkeypatch) -> None:
    """Regression: a stale socket file left by a crashed daemon must not make
    `maint-daemon start` report success for a child that failed to boot."""
    import subprocess

    from click.testing import CliRunner

    from memo.cli import cli

    stale = _socket_path(tmp_path)
    stale.touch()  # leftover from a crash — nothing is listening

    class _DeadChild:
        pid = 4242

        def poll(self) -> int:
            return 1  # child exited immediately (failed to boot)

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _DeadChild())
    monkeypatch.setattr(Path, "home", lambda: tmp_path)  # keep logs out of ~/Library

    result = CliRunner().invoke(cli, ["maint-daemon", "start"], env=_start_env(tmp_path))

    assert result.exit_code == 1
    assert "failed to start" in result.stderr
    assert not stale.exists()  # the stale socket was removed before the probe


def test_start_succeeds_when_child_binds_fresh_socket(tmp_path: Path, monkeypatch) -> None:
    """The connect-based probe reports success only once the (fake) child
    actually answers a ping on a freshly bound socket — a stale leftover
    socket file must not break a genuine boot either."""
    import subprocess

    from click.testing import CliRunner

    from memo.cli import cli

    _socket_path(tmp_path).touch()  # stale leftover from a crash

    class _LiveChild:
        pid = 4242

        def poll(self) -> None:
            return None  # still running

    started: list[_MaintServer] = []

    def fake_popen(*a: object, **k: object) -> _LiveChild:
        # Simulate the child binding a fresh socket after the stale unlink.
        server = _MaintServer(str(_socket_path(tmp_path)), lambda op, params: {"ok": True})
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.append(server)
        return _LiveChild()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    try:
        result = CliRunner().invoke(cli, ["maint-daemon", "start"], env=_start_env(tmp_path))
        assert result.exit_code == 0, result.output
        assert "started" in result.stderr
    finally:
        for server in started:
            server.shutdown()
            server.server_close()


# -- flag-gated Memory.consolidate ----------------------------------------


def test_consolidate_in_process_when_flag_off(mock_memory, monkeypatch) -> None:
    monkeypatch.delenv("MEMO_MAINT_VIA_DAEMON", raising=False)
    monkeypatch.setattr(mock_memory, "_consolidate_in_process", lambda **k: [{"via": "in-process"}])
    monkeypatch.setattr(
        "memo.maint_client.consolidate", lambda **k: [{"via": "SHOULD_NOT_BE_USED"}]
    )
    assert mock_memory.consolidate() == [{"via": "in-process"}]


def test_consolidate_routes_to_daemon_when_flag_on(mock_memory, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_MAINT_VIA_DAEMON", "1")
    monkeypatch.setattr(mock_memory, "_consolidate_in_process", lambda **k: [{"via": "in-process"}])
    monkeypatch.setattr("memo.maint_client.consolidate", lambda **k: [{"via": "daemon"}])
    assert mock_memory.consolidate(threshold=0.9) == [{"via": "daemon"}]


def test_consolidate_falls_back_when_daemon_down(mock_memory, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_MAINT_VIA_DAEMON", "1")
    monkeypatch.setattr(mock_memory, "_consolidate_in_process", lambda **k: [{"via": "in-process"}])
    monkeypatch.setattr("memo.maint_client.consolidate", lambda **k: None)  # unreachable
    assert mock_memory.consolidate() == [{"via": "in-process"}]
