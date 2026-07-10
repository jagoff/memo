"""Tests for server_feedback MCP tool registration."""

from __future__ import annotations

from unittest.mock import MagicMock

from memo.memory import Memory


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


def test_register_exposes_all_feedback_tools(tmp_cfg) -> None:
    """register() must expose exactly the expected feedback MCP tools."""
    from memo.server_feedback import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    expected = {
        "memo_feedback_record",
        "memo_feedback_flag",
        "memo_feedback_list",
        "memo_feedback_clear",
    }
    assert expected == set(tools), f"Tool mismatch: {set(tools)}"


def test_memo_feedback_flag_delegates_to_memory(tmp_cfg) -> None:
    """memo_feedback_flag must forward kind + superseded_by to feedback_flag."""
    from memo.server_feedback import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.feedback_flag.return_value = {"action": "superseded", "source_id": "abc123"}

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_feedback_flag"](source_id="abc123", kind="wrong", superseded_by="def456")

    mem.feedback_flag.assert_called_once_with("abc123", kind="wrong", superseded_by="def456")
    assert result["action"] == "superseded"


def test_memo_feedback_record_delegates_to_memory(tmp_cfg) -> None:
    """memo_feedback_record must call memory.feedback_record with correct args."""
    from memo.server_feedback import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.feedback_record.return_value = {
        "status": "ok",
        "source_id": "abc123",
        "rating": "thumbs_up",
    }

    server, tools = _make_server_and_tools()
    register(server, mem)

    assert "memo_feedback_record" in tools

    result = tools["memo_feedback_record"](
        source_id="abc123",
        query="what is MLX?",
        rating="thumbs_up",
    )

    mem.feedback_record.assert_called_once_with(
        "abc123", query_text="what is MLX?", rating="thumbs_up"
    )
    assert result == {"status": "ok", "source_id": "abc123", "rating": "thumbs_up"}


def test_memo_feedback_record_supports_all_rating_values(tmp_cfg) -> None:
    """memo_feedback_record must pass through all supported rating strings."""
    from memo.server_feedback import register

    for rating in ("thumbs_up", "up", "click", "thumbs_down", "down", "ignore"):
        mem = MagicMock(spec=Memory)
        mem.cfg = tmp_cfg
        mem.feedback_record.return_value = {"rating": rating}

        server, tools = _make_server_and_tools()
        register(server, mem)

        result = tools["memo_feedback_record"](source_id="id1", query="q", rating=rating)
        mem.feedback_record.assert_called_once_with("id1", query_text="q", rating=rating)
        assert result == {"rating": rating}


def test_memo_feedback_list_returns_envelope_with_rows_and_count(tmp_cfg) -> None:
    """memo_feedback_list must return a dict with 'rows' and 'count'."""
    from memo.server_feedback import register

    fake_rows = [
        {"source_id": "abc", "query": "foo", "rating": "thumbs_up"},
        {"source_id": "abc", "query": "bar", "rating": "click"},
    ]

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.feedback_list.return_value = fake_rows

    server, tools = _make_server_and_tools()
    register(server, mem)

    assert "memo_feedback_list" in tools

    result = tools["memo_feedback_list"](source_id="abc", limit=10)

    mem.feedback_list.assert_called_once_with(source_id="abc", limit=10)
    assert result["rows"] == fake_rows
    assert result["count"] == 2


def test_memo_feedback_list_default_args(tmp_cfg) -> None:
    """memo_feedback_list must default source_id=None and limit=50."""
    from memo.server_feedback import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.feedback_list.return_value = []

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_feedback_list"]()

    mem.feedback_list.assert_called_once_with(source_id=None, limit=50)
    assert result == {"rows": [], "count": 0}


def test_memo_feedback_list_without_source_filter(tmp_cfg) -> None:
    """memo_feedback_list returns all rows when source_id is None."""
    from memo.server_feedback import register

    fake_rows = [
        {"source_id": "x", "query": "q1", "rating": "up"},
        {"source_id": "y", "query": "q2", "rating": "down"},
        {"source_id": "z", "query": "q3", "rating": "ignore"},
    ]

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.feedback_list.return_value = fake_rows

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_feedback_list"](source_id=None, limit=50)

    assert result["count"] == 3
    assert isinstance(result["rows"], list)


def test_memo_feedback_clear_returns_envelope_with_deleted_count(tmp_cfg) -> None:
    """memo_feedback_clear must return dict with 'source_id' and 'deleted'."""
    from memo.server_feedback import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.feedback_clear.return_value = 3

    server, tools = _make_server_and_tools()
    register(server, mem)

    assert "memo_feedback_clear" in tools

    result = tools["memo_feedback_clear"](source_id="abc123")

    mem.feedback_clear.assert_called_once_with("abc123")
    assert result == {"source_id": "abc123", "deleted": 3}


def test_memo_feedback_clear_zero_deleted(tmp_cfg) -> None:
    """memo_feedback_clear must report deleted=0 when no rows existed."""
    from memo.server_feedback import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.feedback_clear.return_value = 0

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_feedback_clear"](source_id="nonexistent")

    assert result["deleted"] == 0
    assert result["source_id"] == "nonexistent"
