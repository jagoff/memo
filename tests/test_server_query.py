"""Tests for server_query MCP tool registration."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock


def _make_server_and_tools() -> tuple[MagicMock, dict]:
    """Return a (server_mock, tools_dict) pair.

    ``server.tool()`` is wired so each ``@server.tool()`` decorated function
    is captured in ``tools`` by its ``__name__``, without going through
    FastMCP.
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


def test_register_exposes_all_four_tools(tmp_cfg) -> None:
    """register() must expose exactly the four expected MCP tools."""
    from memo.memory import Memory
    from memo.server_query import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    expected = {"memo_query_save", "memo_query_list", "memo_query_run", "memo_query_delete"}
    assert expected == set(tools), f"Tool mismatch: {set(tools)}"


def test_memo_query_save_calls_store_and_returns_envelope(tmp_cfg) -> None:
    """memo_query_save must call save_query with all params and return the saved envelope."""
    from memo.memory import Memory
    from memo.server_query import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_query_save"](
        name="test-query",
        query_text="find all decisions",
        type_filter="decision",
        tags_filter=["arch"],
        date_from="2026-01-01",
        date_to="2026-12-31",
        search_mode="vec",
        limit=5,
        description="My test query",
    )

    mem.query_composer.query_store.save_query.assert_called_once_with(
        name="test-query",
        query_text="find all decisions",
        type_filter="decision",
        tags_filter=["arch"],
        date_from="2026-01-01",
        date_to="2026-12-31",
        search_mode="vec",
        limit=5,
        description="My test query",
    )
    assert result == {"status": "saved", "name": "test-query"}


def test_memo_query_save_minimal_args(tmp_cfg) -> None:
    """memo_query_save works with only required args; optional params default to None/'hybrid'/10."""
    from memo.memory import Memory
    from memo.server_query import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_query_save"](name="minimal", query_text="search text")

    assert result["status"] == "saved"
    assert result["name"] == "minimal"
    mem.query_composer.query_store.save_query.assert_called_once()


def test_memo_query_list_returns_query_dicts(tmp_cfg) -> None:
    """memo_query_list must convert each query object's __dict__ into the returned list."""
    from memo.memory import Memory
    from memo.server_query import register

    # Use SimpleNamespace so __dict__ is clean (only the attributes we set).
    q1 = SimpleNamespace(name="q1", query_text="find bugs", limit=10)
    q2 = SimpleNamespace(name="q2", query_text="find decisions", limit=5)

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.query_composer.query_store.list_queries.return_value = [q1, q2]

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_query_list"]()

    mem.query_composer.query_store.list_queries.assert_called_once()
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == {"name": "q1", "query_text": "find bugs", "limit": 10}
    assert result[1] == {"name": "q2", "query_text": "find decisions", "limit": 5}


def test_memo_query_list_empty(tmp_cfg) -> None:
    """memo_query_list returns an empty list when there are no saved queries."""
    from memo.memory import Memory
    from memo.server_query import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.query_composer.query_store.list_queries.return_value = []

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_query_list"]()

    assert result == []


def test_memo_query_run_returns_results_envelope(tmp_cfg) -> None:
    """memo_query_run must call get_query + execute_query and return the results envelope."""
    from memo.memory import Memory
    from memo.server_query import register

    mock_result_item = MagicMock()
    mock_result_item.to_dict.return_value = {"id": "abc", "content": "a decision"}

    mock_exec_result = MagicMock()
    mock_exec_result.query_name = "my-query"
    mock_exec_result.count = 1
    mock_exec_result.executed_at = "2026-07-01T03:00:00"
    mock_exec_result.results = [mock_result_item]

    mock_query = MagicMock()

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.query_composer.query_store.get_query.return_value = mock_query
    mem.query_composer.execute_query.return_value = mock_exec_result

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_query_run"](name="my-query")

    mem.query_composer.query_store.get_query.assert_called_once_with("my-query")
    mem.query_composer.execute_query.assert_called_once_with(mock_query)
    assert result["query_name"] == "my-query"
    assert result["count"] == 1
    assert result["executed_at"] == "2026-07-01T03:00:00"
    assert result["results"] == [{"id": "abc", "content": "a decision"}]


def test_memo_query_run_not_found_returns_error_envelope(tmp_cfg) -> None:
    """memo_query_run returns an error envelope when the named query doesn't exist."""
    from memo.memory import Memory
    from memo.server_query import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.query_composer.query_store.get_query.return_value = None

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_query_run"](name="missing")

    assert result["error"] == "Query not found"
    assert result["name"] == "missing"
    # execute_query must NOT be called when the query is absent
    mem.query_composer.execute_query.assert_not_called()


def test_memo_query_delete_success(tmp_cfg) -> None:
    """memo_query_delete returns {'success': True, 'name': ...} when the query is found and deleted."""
    from memo.memory import Memory
    from memo.server_query import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.query_composer.query_store.delete_query.return_value = True

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_query_delete"](name="old-query")

    mem.query_composer.query_store.delete_query.assert_called_once_with("old-query")
    assert result == {"success": True, "name": "old-query"}


def test_memo_query_delete_not_found(tmp_cfg) -> None:
    """memo_query_delete returns {'success': False, 'name': ...} when the query does not exist."""
    from memo.memory import Memory
    from memo.server_query import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.query_composer.query_store.delete_query.return_value = False

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_query_delete"](name="nonexistent")

    assert result["success"] is False
    assert result["name"] == "nonexistent"
