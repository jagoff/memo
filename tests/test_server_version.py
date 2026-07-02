"""Tests for server_version MCP tool registration."""

from __future__ import annotations

from unittest.mock import MagicMock

from memo.versioning import DiffResult, Version


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


def _fake_version(
    version_id: int = 1,
    memory_id: str = "abc123",
) -> Version:
    return Version(
        version_id=version_id,
        memory_id=memory_id,
        timestamp="2026-07-01T00:00:00+00:00",
        title="Test memory",
        type="fact",
        tags=["test"],
        body="Some body text.",
        reason=None,
    )


def _fake_diff_result(memory_id: str = "abc123") -> DiffResult:
    return DiffResult(
        memory_id=memory_id,
        version_a=2,
        version_b=1,
        unified_diff="--- v1\n+++ v2\n-old\n+new",
        changes=["-old", "+new"],
    )


def test_register_exposes_all_three_tools(tmp_cfg) -> None:
    """register() must expose exactly the three expected versioning MCP tools."""
    from memo.memory import Memory
    from memo.server_version import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    expected = {"memo_version_history", "memo_version_diff", "memo_version_rollback"}
    assert expected == set(tools), f"Tool mismatch: {set(tools)}"


def test_memo_version_history_calls_get_version_history(tmp_cfg) -> None:
    """memo_version_history must call memory.versioning.get_version_history and return a list of dicts."""
    from memo.memory import Memory
    from memo.server_version import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    v1 = _fake_version(version_id=2, memory_id="abc123")
    v2 = _fake_version(version_id=1, memory_id="abc123")
    mem.versioning.get_version_history.return_value = [v1, v2]

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_version_history"](memory_id="abc123", limit=5)

    mem.versioning.get_version_history.assert_called_once_with("abc123", limit=5)
    assert isinstance(result, list)
    assert len(result) == 2
    first = result[0]
    assert first["version_id"] == 2
    assert first["memory_id"] == "abc123"
    assert first["type"] == "fact"
    assert first["tags"] == ["test"]
    assert "body" in first


def test_memo_version_history_default_limit(tmp_cfg) -> None:
    """memo_version_history uses limit=10 by default."""
    from memo.memory import Memory
    from memo.server_version import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.versioning.get_version_history.return_value = []

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_version_history"](memory_id="xyz")

    mem.versioning.get_version_history.assert_called_once_with("xyz", limit=10)
    assert result == []


def test_memo_version_history_empty(tmp_cfg) -> None:
    """memo_version_history returns [] when no versions exist."""
    from memo.memory import Memory
    from memo.server_version import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.versioning.get_version_history.return_value = []

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_version_history"](memory_id="no-such-id")

    assert result == []


def test_memo_version_diff_returns_dict(tmp_cfg) -> None:
    """memo_version_diff must return a dict when a diff exists."""
    from memo.memory import Memory
    from memo.server_version import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.versioning.diff_versions.return_value = _fake_diff_result("abc123")

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_version_diff"](memory_id="abc123", version_a=2, version_b=1)

    mem.versioning.diff_versions.assert_called_once_with("abc123", 2, 1)
    assert isinstance(result, dict)
    assert result["memory_id"] == "abc123"
    assert result["version_a"] == 2
    assert result["version_b"] == 1
    assert "-old" in result["changes"]
    assert "+new" in result["changes"]
    assert "unified_diff" in result


def test_memo_version_diff_default_versions(tmp_cfg) -> None:
    """memo_version_diff passes None for both versions by default."""
    from memo.memory import Memory
    from memo.server_version import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.versioning.diff_versions.return_value = _fake_diff_result()

    server, tools = _make_server_and_tools()
    register(server, mem)

    tools["memo_version_diff"](memory_id="abc123")

    mem.versioning.diff_versions.assert_called_once_with("abc123", None, None)


def test_memo_version_diff_returns_none_when_no_diff(tmp_cfg) -> None:
    """memo_version_diff returns None when diff_versions returns None."""
    from memo.memory import Memory
    from memo.server_version import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.versioning.diff_versions.return_value = None

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_version_diff"](memory_id="no-history")

    assert result is None


def test_memo_version_rollback_success(tmp_cfg) -> None:
    """memo_version_rollback returns success=True when rollback_to_version succeeds."""
    from memo.memory import Memory
    from memo.server_version import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.versioning.rollback_to_version.return_value = True

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_version_rollback"](memory_id="abc123", version_id=3, reason="oops")

    mem.versioning.rollback_to_version.assert_called_once_with("abc123", 3, "oops")
    assert result == {"success": True, "memory_id": "abc123", "version_id": 3}


def test_memo_version_rollback_failure(tmp_cfg) -> None:
    """memo_version_rollback returns success=False when version is not found."""
    from memo.memory import Memory
    from memo.server_version import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.versioning.rollback_to_version.return_value = False

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_version_rollback"](memory_id="abc123", version_id=99)

    mem.versioning.rollback_to_version.assert_called_once_with("abc123", 99, None)
    assert result["success"] is False
    assert result["memory_id"] == "abc123"
    assert result["version_id"] == 99


def test_memo_version_rollback_no_reason(tmp_cfg) -> None:
    """memo_version_rollback passes reason=None by default."""
    from memo.memory import Memory
    from memo.server_version import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.versioning.rollback_to_version.return_value = True

    server, tools = _make_server_and_tools()
    register(server, mem)

    tools["memo_version_rollback"](memory_id="m1", version_id=1)

    mem.versioning.rollback_to_version.assert_called_once_with("m1", 1, None)
