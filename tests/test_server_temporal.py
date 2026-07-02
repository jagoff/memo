"""Tests for server_temporal MCP tool registration."""

from __future__ import annotations

from unittest.mock import MagicMock


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


# ---------------------------------------------------------------------------
# Registration contract
# ---------------------------------------------------------------------------


def test_register_exposes_all_four_tools(tmp_cfg) -> None:
    """register() must expose exactly the four temporal MCP tools."""
    from memo.memory import Memory
    from memo.server_temporal import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    expected = {
        "memo_temporal_contradictions",
        "memo_temporal_timeline",
        "memo_temporal_stale",
        "memo_temporal_patterns",
    }
    assert expected == set(tools), f"Tool mismatch: {set(tools)}"


# ---------------------------------------------------------------------------
# memo_temporal_contradictions
# ---------------------------------------------------------------------------


def test_memo_temporal_contradictions_calls_detect_and_serializes(tmp_cfg) -> None:
    """memo_temporal_contradictions must call detect_entity_contradictions with
    the correct kwargs and return each Contradiction serialised via __dict__."""
    import pytest

    from memo.memory import Memory
    from memo.server_temporal import register
    from memo.temporal import Contradiction

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    fake = Contradiction(
        memory_id_a="aaa111",
        memory_id_b="bbb222",
        title_a="MLX is the embedder",
        title_b="Ollama is the embedder",
        date_a="2024-01-01T00:00:00+00:00",
        date_b="2024-06-01T00:00:00+00:00",
        relationship="contradiction",
        rationale="Memory A says MLX; Memory B says Ollama.",
        confidence=0.91,
    )
    mem.temporal.detect_entity_contradictions.return_value = [fake]

    result = tools["memo_temporal_contradictions"](
        entity="ollama",
        entity_type="technology",
        confidence_threshold=0.8,
        max_pairs=10,
    )

    mem.temporal.detect_entity_contradictions.assert_called_once_with(
        entity_name="ollama",
        entity_type="technology",
        confidence_threshold=0.8,
        max_pairs=10,
    )
    assert isinstance(result, list)
    assert len(result) == 1
    row = result[0]
    assert row["memory_id_a"] == "aaa111"
    assert row["memory_id_b"] == "bbb222"
    assert row["relationship"] == "contradiction"
    assert row["confidence"] == pytest.approx(0.91)


def test_memo_temporal_contradictions_empty_when_none_detected(tmp_cfg) -> None:
    """memo_temporal_contradictions must return [] when detect returns nothing."""
    from memo.memory import Memory
    from memo.server_temporal import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    mem.temporal.detect_entity_contradictions.return_value = []

    result = tools["memo_temporal_contradictions"](entity="ghost")

    assert result == []
    mem.temporal.detect_entity_contradictions.assert_called_once_with(
        entity_name="ghost",
        entity_type=None,
        confidence_threshold=0.7,
        max_pairs=20,
    )


# ---------------------------------------------------------------------------
# memo_temporal_timeline
# ---------------------------------------------------------------------------


def test_memo_temporal_timeline_returns_none_for_unknown_entity(tmp_cfg) -> None:
    """memo_temporal_timeline must return None when build_entity_timeline returns None."""
    from memo.memory import Memory
    from memo.server_temporal import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    mem.temporal.build_entity_timeline.return_value = None

    result = tools["memo_temporal_timeline"](entity="ghost")

    assert result is None
    mem.temporal.build_entity_timeline.assert_called_once_with(
        entity_name="ghost",
        entity_type=None,
    )


def test_memo_temporal_timeline_returns_envelope_for_known_entity(tmp_cfg) -> None:
    """memo_temporal_timeline must build the expected envelope dict from EntityTimeline."""
    from memo.memory import Memory
    from memo.server_temporal import register
    from memo.temporal import EntityTimeline, TimelineEvent

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    event = TimelineEvent(
        memory_id="mem001",
        title="Adopted MLX for inference",
        date="2024-03-15T10:00:00+00:00",
        type="decision",
        snippet="Switched from Ollama to MLX.",
    )
    timeline = EntityTimeline(
        entity_name="mlx",
        entity_type="technology",
        events=[event],
        first_seen="2024-03-15T10:00:00+00:00",
        last_seen="2024-03-15T10:00:00+00:00",
    )
    mem.temporal.build_entity_timeline.return_value = timeline

    result = tools["memo_temporal_timeline"](entity="mlx", entity_type="technology")

    assert result is not None
    assert result["entity_name"] == "mlx"
    assert result["entity_type"] == "technology"
    assert result["first_seen"] == "2024-03-15T10:00:00+00:00"
    assert result["last_seen"] == "2024-03-15T10:00:00+00:00"
    assert isinstance(result["events"], list)
    assert len(result["events"]) == 1
    ev = result["events"][0]
    assert ev["memory_id"] == "mem001"
    assert ev["title"] == "Adopted MLX for inference"
    assert ev["type"] == "decision"
    mem.temporal.build_entity_timeline.assert_called_once_with(
        entity_name="mlx",
        entity_type="technology",
    )


# ---------------------------------------------------------------------------
# memo_temporal_stale
# ---------------------------------------------------------------------------


def test_memo_temporal_stale_passes_through_list(tmp_cfg) -> None:
    """memo_temporal_stale must return the list from detect_stale_memories unchanged."""
    from memo.memory import Memory
    from memo.server_temporal import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    fake_stale = [
        {
            "id": "abc123",
            "title": "Old note about Ollama",
            "type": "fact",
            "updated": "2023-01-01T00:00:00+00:00",
            "days_since_update": 550,
            "access_count": 0,
        }
    ]
    mem.temporal.detect_stale_memories.return_value = fake_stale

    result = tools["memo_temporal_stale"](days_threshold=365, min_access_count=0)

    mem.temporal.detect_stale_memories.assert_called_once_with(
        days_threshold=365,
        min_access_count=0,
    )
    assert result == fake_stale
    assert result[0]["days_since_update"] == 550


def test_memo_temporal_stale_empty_corpus(tmp_cfg) -> None:
    """memo_temporal_stale must return [] when no stale memories exist."""
    from memo.memory import Memory
    from memo.server_temporal import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    mem.temporal.detect_stale_memories.return_value = []

    result = tools["memo_temporal_stale"]()

    assert result == []
    mem.temporal.detect_stale_memories.assert_called_once_with(
        days_threshold=180,
        min_access_count=0,
    )


# ---------------------------------------------------------------------------
# memo_temporal_patterns
# ---------------------------------------------------------------------------


def test_memo_temporal_patterns_passes_through_dict(tmp_cfg) -> None:
    """memo_temporal_patterns must return the dict from detect_temporal_patterns unchanged."""
    from memo.memory import Memory
    from memo.server_temporal import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    fake_patterns: dict = {
        "memories_per_month": {"2024-01": 5, "2024-02": 3},
        "type_distribution_over_time": {"2024-01": {"decision": 2, "fact": 3}},
        "most_active_entities": {"mlx": 12, "memo": 8},
    }
    mem.temporal.detect_temporal_patterns.return_value = fake_patterns

    result = tools["memo_temporal_patterns"]()

    mem.temporal.detect_temporal_patterns.assert_called_once_with()
    assert result == fake_patterns
    assert result["memories_per_month"]["2024-01"] == 5
    assert result["most_active_entities"]["mlx"] == 12
