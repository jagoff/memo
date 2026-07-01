"""Tests for server_contradict MCP tool registration."""
from __future__ import annotations

from unittest.mock import MagicMock

from memo.contradict import PairRecord, ScanResult


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


def _make_pair_record(**kwargs) -> PairRecord:
    defaults: dict = dict(
        pair_id=1,
        memory_id_a="aaaa1111",
        memory_id_b="bbbb2222",
        relationship="contradiction",
        confidence=0.85,
        rationale="They say opposite things.",
        status="open",
        detected_at="2026-01-01T00:00:00+00:00",
        resolved_at=None,
        resolution_note=None,
    )
    defaults.update(kwargs)
    return PairRecord(**defaults)


def test_register_exposes_all_four_tools(tmp_cfg) -> None:
    """register() must expose exactly the four expected MCP tools."""
    from memo.memory import Memory
    from memo.server_contradict import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    expected = {
        "memo_contradict_scan",
        "memo_contradict_list",
        "memo_contradict_resolve",
        "memo_contradict_stats",
    }
    assert expected == set(tools), f"Tool mismatch: {set(tools)}"


def test_memo_contradict_scan_returns_envelope(tmp_cfg) -> None:
    """memo_contradict_scan delegates to scan_corpus and returns a well-formed envelope."""
    from memo.memory import Memory
    from memo.server_contradict import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.contradict_scanner.scan_corpus.return_value = ScanResult(
        scanned_memories=100,
        pairs_examined=42,
        pairs_inserted=3,
        pairs_refreshed=1,
        pairs_skipped_resolved=5,
        contradictions_found=2,
        evolutions_found=1,
    )

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_contradict_scan"](
        top_k=5,
        sim_floor=0.55,
        confidence_threshold=0.7,
        min_days_apart=1,
        max_memories=100,
        max_pairs=50,
        since=None,
        type=None,
    )

    mem.contradict_scanner.scan_corpus.assert_called_once_with(
        top_k=5,
        sim_floor=0.55,
        confidence_threshold=0.7,
        min_days_apart=1,
        max_memories=100,
        max_pairs=50,
        since=None,
        type_=None,
    )
    assert result["scanned_memories"] == 100
    assert result["pairs_examined"] == 42
    assert result["pairs_inserted"] == 3
    assert result["pairs_refreshed"] == 1
    assert result["pairs_skipped_resolved"] == 5
    assert result["contradictions_found"] == 2
    assert result["evolutions_found"] == 1


def test_memo_contradict_list_open_routes_to_list_open(tmp_cfg) -> None:
    """memo_contradict_list with status='open' calls list_open, not list_all."""
    from memo.memory import Memory
    from memo.server_contradict import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    pair = _make_pair_record(pair_id=7, relationship="contradiction", confidence=0.9)
    mem.contradict_store.list_open.return_value = [pair]

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_contradict_list"](
        status="open",
        limit=10,
        min_confidence=0.5,
        relationship="contradiction",
    )

    mem.contradict_store.list_open.assert_called_once_with(
        limit=10,
        min_confidence=0.5,
        relationship="contradiction",
    )
    mem.contradict_store.list_all.assert_not_called()
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["pair_id"] == 7
    assert result[0]["relationship"] == "contradiction"


def test_memo_contradict_list_non_open_routes_to_list_all(tmp_cfg) -> None:
    """memo_contradict_list with a non-open status calls list_all and returns dicts."""
    from memo.memory import Memory
    from memo.server_contradict import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    pair = _make_pair_record(pair_id=3, status="dismissed", relationship="evolution", confidence=0.8)
    mem.contradict_store.list_all.return_value = [pair]

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_contradict_list"](
        status="dismissed",
        limit=5,
        min_confidence=0.0,
        relationship=None,
    )

    mem.contradict_store.list_all.assert_called_once_with(status="dismissed", limit=5)
    mem.contradict_store.list_open.assert_not_called()
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["pair_id"] == 3
    assert result[0]["status"] == "dismissed"
    assert result[0]["relationship"] == "evolution"


def test_memo_contradict_list_filters_by_relationship(tmp_cfg) -> None:
    """memo_contradict_list with non-open status filters results by relationship."""
    from memo.memory import Memory
    from memo.server_contradict import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    match = _make_pair_record(pair_id=1, status="evolved", relationship="evolution", confidence=0.88)
    no_match = _make_pair_record(
        pair_id=2, status="evolved", relationship="contradiction", confidence=0.75
    )
    mem.contradict_store.list_all.return_value = [match, no_match]

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_contradict_list"](
        status="evolved",
        limit=20,
        min_confidence=0.0,
        relationship="evolution",
    )

    assert len(result) == 1
    assert result[0]["pair_id"] == 1
    assert result[0]["relationship"] == "evolution"


def test_memo_contradict_list_filters_by_min_confidence(tmp_cfg) -> None:
    """memo_contradict_list with non-open status filters results by min_confidence."""
    from memo.memory import Memory
    from memo.server_contradict import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    high = _make_pair_record(pair_id=10, status="dismissed", confidence=0.9)
    low = _make_pair_record(pair_id=11, status="dismissed", confidence=0.5)
    mem.contradict_store.list_all.return_value = [high, low]

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_contradict_list"](
        status="dismissed",
        limit=20,
        min_confidence=0.8,
        relationship=None,
    )

    assert len(result) == 1
    assert result[0]["pair_id"] == 10
    assert result[0]["confidence"] >= 0.8


def test_memo_contradict_resolve_returns_envelope(tmp_cfg) -> None:
    """memo_contradict_resolve calls store.resolve and returns the result envelope."""
    from memo.memory import Memory
    from memo.server_contradict import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.contradict_store.resolve.return_value = True

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_contradict_resolve"](
        pair_id=42, status="dismissed", note="false positive"
    )

    mem.contradict_store.resolve.assert_called_once_with(42, "dismissed", note="false positive")
    assert result["updated"] is True
    assert result["pair_id"] == 42
    assert result["status"] == "dismissed"


def test_memo_contradict_resolve_not_found_returns_false(tmp_cfg) -> None:
    """memo_contradict_resolve reflects updated=False when store.resolve returns False."""
    from memo.memory import Memory
    from memo.server_contradict import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.contradict_store.resolve.return_value = False

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_contradict_resolve"](pair_id=999, status="fused", note=None)

    assert result["updated"] is False
    assert result["pair_id"] == 999
    assert result["status"] == "fused"


def test_memo_contradict_stats_delegates(tmp_cfg) -> None:
    """memo_contradict_stats returns the dict from contradict_store.stats() verbatim."""
    from memo.memory import Memory
    from memo.server_contradict import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.contradict_store.stats.return_value = {"open": 4, "dismissed": 2, "evolved": 1}

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_contradict_stats"]()

    mem.contradict_store.stats.assert_called_once_with()
    assert result == {"open": 4, "dismissed": 2, "evolved": 1}
