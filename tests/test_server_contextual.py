"""Tests for server_contextual MCP tool registration."""

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


def test_register_exposes_all_five_tools(tmp_cfg) -> None:
    """register() must expose exactly the five expected contextual MCP tools."""
    from memo.server_contextual import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    expected = {
        "memo_contextual_search",
        "memo_contextual_record_search",
        "memo_contextual_record_click",
        "memo_contextual_preferences",
        "memo_contextual_history",
    }
    assert expected == set(tools), f"Tool mismatch: {set(tools)}"


def test_memo_contextual_search_returns_list_of_dicts(tmp_cfg) -> None:
    """memo_contextual_search must convert ContextualSearchResult dataclasses to dicts."""
    from memo.contextual import ContextualSearchResult
    from memo.server_contextual import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    fake_result = ContextualSearchResult(
        memory_id="abc-001",
        title="Test memory",
        original_score=0.8,
        contextual_score=0.9,
        boost_factors={"entity_overlap": 0.1},
        snippet="A relevant snippet.",
    )
    mem.contextual.search_with_context.return_value = [fake_result]

    server, tools = _make_server_and_tools()
    register(server, mem)

    assert "memo_contextual_search" in tools

    result = tools["memo_contextual_search"](query="MLX embedder", limit=5, mode="vec")

    mem.contextual.search_with_context.assert_called_once_with(
        query="MLX embedder",
        limit=5,
        mode="vec",
    )
    assert isinstance(result, list)
    assert len(result) == 1
    item = result[0]
    assert isinstance(item, dict)
    assert item["memory_id"] == "abc-001"
    assert item["title"] == "Test memory"
    assert item["contextual_score"] == 0.9
    assert item["boost_factors"] == {"entity_overlap": 0.1}


def test_memo_contextual_search_empty_returns_empty_list(tmp_cfg) -> None:
    """memo_contextual_search must return [] when no results come back."""
    from memo.server_contextual import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.contextual.search_with_context.return_value = []

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_contextual_search"](query="nothing matches")
    assert result == []


def test_memo_contextual_record_search_returns_envelope(tmp_cfg) -> None:
    """memo_contextual_record_search must return {status, count} and call record_search."""
    from memo.server_contextual import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    assert "memo_contextual_record_search" in tools

    result = tools["memo_contextual_record_search"](
        query="recent search",
        memory_ids=["id-1", "id-2", "id-3"],
    )

    mem.contextual.record_search.assert_called_once_with("recent search", ["id-1", "id-2", "id-3"])
    assert result["status"] == "recorded"
    assert result["count"] == 3


def test_memo_contextual_record_search_empty_ids(tmp_cfg) -> None:
    """memo_contextual_record_search with empty id list returns count=0."""
    from memo.server_contextual import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_contextual_record_search"](query="miss", memory_ids=[])

    mem.contextual.record_search.assert_called_once_with("miss", [])
    assert result == {"status": "recorded", "count": 0}


def test_memo_contextual_record_click_returns_envelope(tmp_cfg) -> None:
    """memo_contextual_record_click must return {status, memory_id} and call record_click."""
    from memo.server_contextual import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    assert "memo_contextual_record_click" in tools

    result = tools["memo_contextual_record_click"](memory_id="xyz-999")

    mem.contextual.record_click.assert_called_once_with("xyz-999")
    assert result["status"] == "recorded"
    assert result["memory_id"] == "xyz-999"


def test_memo_contextual_preferences_returns_dict(tmp_cfg) -> None:
    """memo_contextual_preferences must serialise UserPreferences to a plain dict."""
    from memo.contextual import UserPreferences
    from memo.server_contextual import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    prefs = UserPreferences(
        preferred_types={"decision": 0.8, "fact": 0.6},
        preferred_entities={"mlx": 0.9},
        recency_weight=0.4,
        diversity_weight=0.2,
        last_updated="2026-07-01T00:00:00+00:00",
    )
    mem.contextual.context.get_preferences.return_value = prefs

    server, tools = _make_server_and_tools()
    register(server, mem)

    assert "memo_contextual_preferences" in tools

    result = tools["memo_contextual_preferences"]()

    mem.contextual.context.get_preferences.assert_called_once()
    assert isinstance(result, dict)
    assert result["preferred_types"] == {"decision": 0.8, "fact": 0.6}
    assert result["preferred_entities"] == {"mlx": 0.9}
    assert result["recency_weight"] == 0.4
    assert result["diversity_weight"] == 0.2


def test_memo_contextual_history_returns_list_of_dicts(tmp_cfg) -> None:
    """memo_contextual_history must serialise PromptContext dataclasses to dicts."""
    from memo.contextual import PromptContext
    from memo.server_contextual import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    fake_entry = PromptContext(
        timestamp="2026-07-01T03:00:00+00:00",
        prompt="How does MLX work?",
        recalled_memories=["mem-001", "mem-002"],
    )
    mem.contextual.context.get_recent_context.return_value = [fake_entry]

    server, tools = _make_server_and_tools()
    register(server, mem)

    assert "memo_contextual_history" in tools

    result = tools["memo_contextual_history"](limit=5)

    mem.contextual.context.get_recent_context.assert_called_once_with(n=5)
    assert isinstance(result, list)
    assert len(result) == 1
    item = result[0]
    assert isinstance(item, dict)
    assert item["prompt"] == "How does MLX work?"
    assert item["recalled_memories"] == ["mem-001", "mem-002"]
    assert item["timestamp"] == "2026-07-01T03:00:00+00:00"


def test_memo_contextual_history_empty(tmp_cfg) -> None:
    """memo_contextual_history must return [] when no history is present."""
    from memo.server_contextual import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.contextual.context.get_recent_context.return_value = []

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_contextual_history"]()
    assert result == []


def test_memo_contextual_history_default_limit(tmp_cfg) -> None:
    """memo_contextual_history must use limit=10 when not explicitly supplied."""
    from memo.server_contextual import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.contextual.context.get_recent_context.return_value = []

    server, tools = _make_server_and_tools()
    register(server, mem)

    tools["memo_contextual_history"]()  # no limit arg — falls back to default 10
    mem.contextual.context.get_recent_context.assert_called_once_with(n=10)
