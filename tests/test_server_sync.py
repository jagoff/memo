"""Tests for server_sync MCP tool registration."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from memo.sync import SyncDiff


def _make_server_and_tools():
    """Return a (server_mock, tools_dict) pair.

    `server.tool()` is wired so each `@server.tool()` decorated function is
    captured in `tools` by its `__name__`, without going through FastMCP.
    """
    server = MagicMock()
    tools: dict = {}

    def tool_decorator():
        def wrapper(fn):
            tools[fn.__name__] = fn
            return fn
        return wrapper

    server.tool = tool_decorator
    return server, tools


def test_register_exposes_all_three_tools(tmp_cfg) -> None:
    """register() must expose exactly the three expected MCP tools."""
    from memo.memory import Memory
    from memo.server_sync import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    expected = {"memo_sync_diff", "memo_sync_push", "memo_sync_pull"}
    assert expected == set(tools), f"Tool mismatch: {set(tools)}"


def test_memo_sync_diff_returns_error_envelope(tmp_cfg) -> None:
    """memo_sync_diff must return an error dict (replay model has no diff)."""
    from memo.memory import Memory
    from memo.server_sync import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_sync_diff"]()
    assert isinstance(result, dict)
    assert "error" in result
    assert "diff" in result["error"].lower() or "replay" in result["error"].lower()


def test_memo_sync_diff_with_remote_arg_still_returns_error(tmp_cfg) -> None:
    """memo_sync_diff ignores remote arg and always returns an error."""
    from memo.memory import Memory
    from memo.server_sync import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_sync_diff"](remote="/some/path")
    assert isinstance(result, dict)
    assert "error" in result


def test_memo_sync_push_returns_error_envelope(tmp_cfg) -> None:
    """memo_sync_push must return an error dict (pull-only model)."""
    from memo.memory import Memory
    from memo.server_sync import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_sync_push"]()
    assert isinstance(result, dict)
    assert "error" in result
    assert "pull" in result["error"].lower() or "replay" in result["error"].lower()


def test_memo_sync_push_with_remote_arg_still_returns_error(tmp_cfg) -> None:
    """memo_sync_push ignores remote arg and always returns an error."""
    from memo.memory import Memory
    from memo.server_sync import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_sync_push"](remote="/some/path")
    assert isinstance(result, dict)
    assert "error" in result


def test_memo_sync_pull_requires_remote(tmp_cfg) -> None:
    """memo_sync_pull without remote must return an error dict."""
    from memo.memory import Memory
    from memo.server_sync import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_sync_pull"]()
    assert isinstance(result, dict)
    assert "error" in result
    assert "remote" in result["error"].lower()


def test_memo_sync_pull_calls_sync_from_remote(tmp_cfg, tmp_path: Path) -> None:
    """memo_sync_pull must call memory.sync.sync_from_remote with the resolved db path."""
    from memo.memory import Memory
    from memo.server_sync import register

    fake_diff = SyncDiff(applied=3, conflicts=1, errors=0)

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.sync = MagicMock()
    mem.sync.sync_from_remote.return_value = fake_diff

    server, tools = _make_server_and_tools()
    register(server, mem)

    remote_state_dir = str(tmp_path / "remote_state")
    result = tools["memo_sync_pull"](remote=remote_state_dir)

    assert mem.sync.sync_from_remote.called, "sync_from_remote must be called"
    called_path = mem.sync.sync_from_remote.call_args[0][0]
    assert isinstance(called_path, Path)
    assert called_path.name == "history.db"

    assert result == {"applied": 3, "conflicts": 1, "errors": 0}


def test_memo_sync_pull_accepts_direct_db_path(tmp_cfg, tmp_path: Path) -> None:
    """memo_sync_pull must use the path as-is when it ends in .db."""
    from memo.memory import Memory
    from memo.server_sync import register

    fake_diff = SyncDiff(applied=5, conflicts=0, errors=2)

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.sync = MagicMock()
    mem.sync.sync_from_remote.return_value = fake_diff

    server, tools = _make_server_and_tools()
    register(server, mem)

    db_path = str(tmp_path / "custom" / "history.db")
    result = tools["memo_sync_pull"](remote=db_path)

    called_path = mem.sync.sync_from_remote.call_args[0][0]
    assert called_path == Path(db_path)

    assert result == dataclasses.asdict(fake_diff)


def test_memo_sync_pull_result_has_required_keys(tmp_cfg, tmp_path: Path) -> None:
    """memo_sync_pull result must contain applied, conflicts and errors keys."""
    from memo.memory import Memory
    from memo.server_sync import register

    fake_diff = SyncDiff(applied=0, conflicts=0, errors=0)

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.sync = MagicMock()
    mem.sync.sync_from_remote.return_value = fake_diff

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_sync_pull"](remote=str(tmp_path))
    assert "applied" in result
    assert "conflicts" in result
    assert "errors" in result
