"""Tests for server_entities MCP tool registration."""

from __future__ import annotations

from unittest.mock import MagicMock


def _make_server_and_tools() -> tuple[MagicMock, dict]:
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
    """register() must expose exactly the three expected entity MCP tools."""
    from memo.memory import Memory
    from memo.server_entities import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    expected = {"memo_extract_entities", "memo_entities", "memo_entity"}
    assert expected == set(tools), f"Tool mismatch: {set(tools)}"


def test_memo_extract_entities_default_args(tmp_cfg) -> None:
    """memo_extract_entities with defaults calls extract_entities(ids=None, all_=False, skip_already_indexed=True)."""
    from memo.memory import Memory
    from memo.server_entities import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.extract_entities.return_value = {
        "processed": 0,
        "entities_extracted": 0,
        "links_written": 0,
        "skipped": 3,
        "errors": 0,
    }

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_extract_entities"]()

    mem.extract_entities.assert_called_once_with(ids=None, all_=False, skip_already_indexed=True)
    assert result["processed"] == 0
    assert result["skipped"] == 3
    assert result["errors"] == 0


def test_memo_extract_entities_specific_ids(tmp_cfg) -> None:
    """memo_extract_entities with ids= forwards them and keeps skip_already_indexed=True."""
    from memo.memory import Memory
    from memo.server_entities import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.extract_entities.return_value = {
        "processed": 2,
        "entities_extracted": 5,
        "links_written": 5,
        "skipped": 0,
        "errors": 0,
    }

    server, tools = _make_server_and_tools()
    register(server, mem)

    ids = ["abc123def4567890abc123def4567890", "fed321cba9876543fed321cba9876543"]
    result = tools["memo_extract_entities"](ids=ids)

    mem.extract_entities.assert_called_once_with(ids=ids, all_=False, skip_already_indexed=True)
    assert result["processed"] == 2
    assert result["entities_extracted"] == 5


def test_memo_extract_entities_force_disables_skip(tmp_cfg) -> None:
    """force=True must translate to skip_already_indexed=False passed to extract_entities."""
    from memo.memory import Memory
    from memo.server_entities import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.extract_entities.return_value = {
        "processed": 10,
        "entities_extracted": 30,
        "links_written": 30,
        "skipped": 0,
        "errors": 0,
    }

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_extract_entities"](all_=True, force=True)

    mem.extract_entities.assert_called_once_with(ids=None, all_=True, skip_already_indexed=False)
    assert result["processed"] == 10
    assert result["links_written"] == 30


def test_memo_entities_default_args(tmp_cfg) -> None:
    """memo_entities with defaults calls graph.top_entities(limit=30, type_=None) and returns the list."""
    from memo.memory import Memory
    from memo.server_entities import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.graph = MagicMock()
    mem.graph.top_entities.return_value = [
        {"name": "mlx", "type": "technology", "count": 10},
        {"name": "memo", "type": "project", "count": 8},
    ]

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_entities"]()

    mem.graph.top_entities.assert_called_once_with(limit=30, type_=None)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["name"] == "mlx"
    assert result[1]["type"] == "project"


def test_memo_entities_with_type_filter(tmp_cfg) -> None:
    """memo_entities passes limit and type through to graph.top_entities."""
    from memo.memory import Memory
    from memo.server_entities import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.graph = MagicMock()
    mem.graph.top_entities.return_value = [
        {"name": "apple silicon", "type": "technology", "count": 3},
    ]

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_entities"](limit=5, type="technology")

    mem.graph.top_entities.assert_called_once_with(limit=5, type_="technology")
    assert len(result) == 1
    assert result[0]["name"] == "apple silicon"
    assert result[0]["type"] == "technology"


def test_memo_entity_returns_memory_ids(tmp_cfg) -> None:
    """memo_entity calls graph.entity_memories and returns the list of UUIDs."""
    from memo.memory import Memory
    from memo.server_entities import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.graph = MagicMock()
    mem.graph.entity_memories.return_value = [
        "aaaa1111bbbb2222cccc3333dddd4444",
        "1111aaaa2222bbbb3333cccc4444dddd",
    ]

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_entity"](name="mlx")

    mem.graph.entity_memories.assert_called_once_with("mlx", type_=None)
    assert isinstance(result, list)
    assert len(result) == 2
    assert "aaaa1111bbbb2222cccc3333dddd4444" in result


def test_memo_entity_with_type_filter(tmp_cfg) -> None:
    """memo_entity passes type= through as type_= to graph.entity_memories."""
    from memo.memory import Memory
    from memo.server_entities import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.graph = MagicMock()
    mem.graph.entity_memories.return_value = ["ffff0000aaaa1234bbbb5678cccc9012"]

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_entity"](name="Fernando", type="person")

    mem.graph.entity_memories.assert_called_once_with("Fernando", type_="person")
    assert result == ["ffff0000aaaa1234bbbb5678cccc9012"]


def test_memo_entity_empty_result(tmp_cfg) -> None:
    """memo_entity returns empty list when no memories mention the entity."""
    from memo.memory import Memory
    from memo.server_entities import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.graph = MagicMock()
    mem.graph.entity_memories.return_value = []

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_entity"](name="nonexistent_entity")

    mem.graph.entity_memories.assert_called_once_with("nonexistent_entity", type_=None)
    assert result == []
    assert isinstance(result, list)
