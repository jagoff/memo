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
