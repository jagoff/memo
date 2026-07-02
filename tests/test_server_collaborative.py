"""Tests for server_collaborative MCP tool registration."""
from __future__ import annotations

from unittest.mock import MagicMock

from memo.collaborative import CollectiveInsight, SharedConnection


def _make_server_and_tools():
    """Return a (server_mock, tools_dict) pair.

    ``server.tool()`` is wired so each ``@server.tool()`` decorated function
    is captured in ``tools`` by its ``__name__``, without going through FastMCP.
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


def _fake_connection(**overrides) -> SharedConnection:
    defaults: dict = dict(
        connection_id="conn-1",
        from_user="user-a",
        entity_a="Python",
        entity_b="MLX",
        relationship="used_with",
        confidence=0.9,
        discovered_at="2026-01-01T00:00:00+00:00",
        votes=3,
    )
    defaults.update(overrides)
    return SharedConnection(**defaults)


def _fake_insight(**overrides) -> CollectiveInsight:
    defaults: dict = dict(
        insight_id="insight-1",
        content="MLX runs natively on Apple Silicon.",
        contributors=["user-a"],
        upvotes=5,
        downvotes=1,
        created_at="2026-01-01T00:00:00+00:00",
    )
    defaults.update(overrides)
    return CollectiveInsight(**defaults)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_exposes_all_five_tools(tmp_cfg) -> None:
    """register() must expose exactly the five expected MCP tools."""
    from memo.memory import Memory
    from memo.server_collaborative import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    expected = {
        "memo_collaborative_share_connection",
        "memo_collaborative_connections",
        "memo_collaborative_recommend",
        "memo_collaborative_share_insight",
        "memo_collaborative_insights",
    }
    assert expected == set(tools), f"Tool mismatch: {set(tools)}"


# ---------------------------------------------------------------------------
# memo_collaborative_share_connection
# ---------------------------------------------------------------------------


def test_memo_collaborative_share_connection_delegates_and_returns_dict(tmp_cfg) -> None:
    """Tool must call share_connection with all args and return conn.__dict__."""
    from memo.memory import Memory
    from memo.server_collaborative import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    conn = _fake_connection()
    mem.collaborative.share_connection.return_value = conn

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_collaborative_share_connection"](
        user_id="user-a",
        entity_a="Python",
        entity_b="MLX",
        relationship="used_with",
        confidence=0.9,
    )

    mem.collaborative.share_connection.assert_called_once_with(
        user_id="user-a",
        entity_a="Python",
        entity_b="MLX",
        relationship="used_with",
        confidence=0.9,
    )
    assert isinstance(result, dict)
    assert result["connection_id"] == "conn-1"
    assert result["from_user"] == "user-a"
    assert result["entity_a"] == "Python"
    assert result["entity_b"] == "MLX"
    assert result["relationship"] == "used_with"
    assert result["confidence"] == 0.9
    assert result["votes"] == 3


def test_memo_collaborative_share_connection_default_confidence(tmp_cfg) -> None:
    """Tool must pass default confidence=0.7 when not supplied."""
    from memo.memory import Memory
    from memo.server_collaborative import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    conn = _fake_connection(confidence=0.7)
    mem.collaborative.share_connection.return_value = conn

    server, tools = _make_server_and_tools()
    register(server, mem)

    tools["memo_collaborative_share_connection"](
        user_id="user-b",
        entity_a="A",
        entity_b="B",
        relationship="related",
    )

    _, kwargs = mem.collaborative.share_connection.call_args
    assert kwargs["confidence"] == 0.7


# ---------------------------------------------------------------------------
# memo_collaborative_connections
# ---------------------------------------------------------------------------


def test_memo_collaborative_connections_returns_list_of_dicts(tmp_cfg) -> None:
    """Tool must delegate to get_shared_connections and return list of dicts."""
    from memo.memory import Memory
    from memo.server_collaborative import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    conns = [_fake_connection(connection_id=f"conn-{i}") for i in range(3)]
    mem.collaborative.get_shared_connections.return_value = conns

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_collaborative_connections"](entity="Python")

    mem.collaborative.get_shared_connections.assert_called_once_with("Python")
    assert isinstance(result, list)
    assert len(result) == 3
    for i, item in enumerate(result):
        assert isinstance(item, dict)
        assert item["connection_id"] == f"conn-{i}"


def test_memo_collaborative_connections_empty_result(tmp_cfg) -> None:
    """Tool must return an empty list when no connections exist."""
    from memo.memory import Memory
    from memo.server_collaborative import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    mem.collaborative.get_shared_connections.return_value = []

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_collaborative_connections"](entity="Unknown")

    assert result == []


# ---------------------------------------------------------------------------
# memo_collaborative_recommend
# ---------------------------------------------------------------------------


def test_memo_collaborative_recommend_passes_limit_and_returns_dicts(tmp_cfg) -> None:
    """Tool must pass limit kwarg and return list of connection dicts."""
    from memo.memory import Memory
    from memo.server_collaborative import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    rec = _fake_connection(connection_id="rec-1")
    mem.collaborative.get_recommended_connections.return_value = [rec]

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_collaborative_recommend"](entity="MLX", limit=5)

    mem.collaborative.get_recommended_connections.assert_called_once_with("MLX", limit=5)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["connection_id"] == "rec-1"


def test_memo_collaborative_recommend_default_limit(tmp_cfg) -> None:
    """Tool must default to limit=10."""
    from memo.memory import Memory
    from memo.server_collaborative import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    mem.collaborative.get_recommended_connections.return_value = []

    server, tools = _make_server_and_tools()
    register(server, mem)

    tools["memo_collaborative_recommend"](entity="Python")

    mem.collaborative.get_recommended_connections.assert_called_once_with("Python", limit=10)


# ---------------------------------------------------------------------------
# memo_collaborative_share_insight
# ---------------------------------------------------------------------------


def test_memo_collaborative_share_insight_delegates_and_returns_dict(tmp_cfg) -> None:
    """Tool must call share_insight with positional args and return insight.__dict__."""
    from memo.memory import Memory
    from memo.server_collaborative import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    insight = _fake_insight()
    mem.collaborative.share_insight.return_value = insight

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_collaborative_share_insight"](
        user_id="user-a",
        content="MLX runs natively on Apple Silicon.",
    )

    mem.collaborative.share_insight.assert_called_once_with(
        "user-a", "MLX runs natively on Apple Silicon."
    )
    assert isinstance(result, dict)
    assert result["insight_id"] == "insight-1"
    assert result["content"] == "MLX runs natively on Apple Silicon."
    assert result["contributors"] == ["user-a"]
    assert result["upvotes"] == 5
    assert result["downvotes"] == 1


# ---------------------------------------------------------------------------
# memo_collaborative_insights
# ---------------------------------------------------------------------------


def test_memo_collaborative_insights_returns_list_of_dicts(tmp_cfg) -> None:
    """Tool must delegate to get_top_insights and return list of dicts."""
    from memo.memory import Memory
    from memo.server_collaborative import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    top = [
        _fake_insight(insight_id="i-1", upvotes=10),
        _fake_insight(insight_id="i-2", upvotes=5),
    ]
    mem.collaborative.get_top_insights.return_value = top

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_collaborative_insights"](limit=10)

    mem.collaborative.get_top_insights.assert_called_once_with(limit=10)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["insight_id"] == "i-1"
    assert result[0]["upvotes"] == 10
    assert result[1]["insight_id"] == "i-2"


def test_memo_collaborative_insights_default_limit(tmp_cfg) -> None:
    """Tool must default to limit=10 when not supplied."""
    from memo.memory import Memory
    from memo.server_collaborative import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    mem.collaborative.get_top_insights.return_value = []

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_collaborative_insights"]()

    mem.collaborative.get_top_insights.assert_called_once_with(limit=10)
    assert result == []
