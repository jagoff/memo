"""Tests for server_asof MCP tool registration (time-machine domain)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch


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
    """register() must expose all three expected MCP tools."""
    from memo.memory import Memory
    from memo.server_asof import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    expected = {"memo_search_as_of", "memo_ask_as_of", "memo_diff"}
    assert expected == set(tools), f"Tool mismatch: {set(tools)}"


def test_memo_search_as_of_returns_envelope(tmp_cfg) -> None:
    """memo_search_as_of must call reconstruct() and return as_of/snapshot_size/results."""
    from memo.memory import Memory
    from memo.server_asof import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    as_of_dt = datetime(2025, 1, 15, tzinfo=UTC)

    mock_hit = MagicMock()
    mock_hit.type = "fact"
    mock_hit.to_dict.return_value = {"id": "abc123", "title": "Test fact", "type": "fact"}

    mock_snap = MagicMock()
    mock_snap.as_of = as_of_dt
    mock_snap.__len__ = MagicMock(return_value=42)
    mock_snap.search.return_value = [mock_hit]

    with patch("memo.time_machine.reconstruct", return_value=mock_snap) as mock_reconstruct:
        result = tools["memo_search_as_of"](query="test query", as_of="2025-01-15")

    mock_reconstruct.assert_called_once_with(mem, as_of="2025-01-15")
    mock_snap.search.assert_called_once_with("test query", limit=10, mode="hybrid")
    assert result["as_of"] == as_of_dt.isoformat()
    assert result["snapshot_size"] == 42
    assert isinstance(result["results"], list)
    assert len(result["results"]) == 1
    assert result["results"][0]["id"] == "abc123"


def test_memo_search_as_of_type_filter(tmp_cfg) -> None:
    """memo_search_as_of must post-filter hits by type when type= is given."""
    from memo.memory import Memory
    from memo.server_asof import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    as_of_dt = datetime(2025, 6, 1, tzinfo=UTC)

    fact_hit = MagicMock()
    fact_hit.type = "fact"
    fact_hit.to_dict.return_value = {"id": "fact1", "title": "A fact", "type": "fact"}

    note_hit = MagicMock()
    note_hit.type = "note"
    note_hit.to_dict.return_value = {"id": "note1", "title": "A note", "type": "note"}

    mock_snap = MagicMock()
    mock_snap.as_of = as_of_dt
    mock_snap.__len__ = MagicMock(return_value=10)
    mock_snap.search.return_value = [fact_hit, note_hit]

    with patch("memo.time_machine.reconstruct", return_value=mock_snap):
        result = tools["memo_search_as_of"](query="anything", as_of="2025-06-01", type="fact")

    # Only the fact_hit should survive the post-filter
    assert len(result["results"]) == 1
    assert result["results"][0]["type"] == "fact"


def test_memo_search_as_of_type_filter_applied_before_limit(tmp_cfg) -> None:
    """A type-filtered as-of search must over-fetch, filter, THEN trim — so it
    returns up to `limit` type-matching hits even when the top-`limit` slots
    were spent on other-typed rows. Filtering after the limit (the old bug)
    would starve the result to whatever matched inside the first `limit`."""
    from memo.memory import Memory
    from memo.server_asof import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    def _search(query, *, limit, mode):
        # Build `limit` ranked hits where only every 5th is a decision. At
        # limit=10 only 2 decisions exist; the over-fetch (limit*4=40) exposes 8.
        hits = []
        for i in range(limit):
            h = MagicMock()
            h.type = "decision" if i % 5 == 0 else "note"
            h.to_dict.return_value = {"id": f"h{i}", "type": h.type}
            hits.append(h)
        return hits

    mock_snap = MagicMock()
    mock_snap.as_of = datetime(2025, 6, 1, tzinfo=UTC)
    mock_snap.__len__ = MagicMock(return_value=100)
    mock_snap.search.side_effect = _search

    with patch("memo.time_machine.reconstruct", return_value=mock_snap):
        result = tools["memo_search_as_of"](
            query="q", as_of="2025-06-01", limit=10, type="decision"
        )

    # Over-fetch happened: snap.search called with a widened limit.
    assert mock_snap.search.call_args.kwargs["limit"] > 10
    # And it returns the full limit of decisions (8 available > 2 within top-10).
    assert len(result["results"]) == 8
    assert all(r["type"] == "decision" for r in result["results"])


def test_memo_search_as_of_no_type_filter_returns_all(tmp_cfg) -> None:
    """memo_search_as_of must return all hits when no type filter is given."""
    from memo.memory import Memory
    from memo.server_asof import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    as_of_dt = datetime(2025, 6, 1, tzinfo=UTC)

    hit_a = MagicMock()
    hit_a.type = "fact"
    hit_a.to_dict.return_value = {"id": "a", "title": "A", "type": "fact"}

    hit_b = MagicMock()
    hit_b.type = "decision"
    hit_b.to_dict.return_value = {"id": "b", "title": "B", "type": "decision"}

    mock_snap = MagicMock()
    mock_snap.as_of = as_of_dt
    mock_snap.__len__ = MagicMock(return_value=2)
    mock_snap.search.return_value = [hit_a, hit_b]

    with patch("memo.time_machine.reconstruct", return_value=mock_snap):
        result = tools["memo_search_as_of"](query="q", as_of="2025-06-01")

    assert len(result["results"]) == 2


def test_memo_ask_as_of_returns_envelope(tmp_cfg) -> None:
    """memo_ask_as_of must call reconstruct() then snap.ask() and pass through the result."""
    from memo.memory import Memory
    from memo.server_asof import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    expected_answer: dict = {
        "question": "What did I decide?",
        "answer": "You decided to refactor.",
        "sources": [{"id": "abc", "title": "Decision", "type": "decision"}],
        "as_of": "2025-03-10T00:00:00+00:00",
        "snapshot_size": 5,
    }

    mock_snap = MagicMock()
    mock_snap.ask.return_value = expected_answer

    with patch("memo.time_machine.reconstruct", return_value=mock_snap) as mock_reconstruct:
        result = tools["memo_ask_as_of"](question="What did I decide?", as_of="2025-03-10")

    mock_reconstruct.assert_called_once_with(mem, as_of="2025-03-10")
    mock_snap.ask.assert_called_once_with("What did I decide?", k=5)
    assert result == expected_answer
    assert result["answer"] == "You decided to refactor."


def test_memo_ask_as_of_custom_k(tmp_cfg) -> None:
    """memo_ask_as_of must forward the k parameter to snap.ask()."""
    from memo.memory import Memory
    from memo.server_asof import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    mock_snap = MagicMock()
    mock_snap.ask.return_value = {
        "question": "q",
        "answer": "a",
        "sources": [],
        "as_of": "x",
        "snapshot_size": 0,
    }

    with patch("memo.time_machine.reconstruct", return_value=mock_snap):
        tools["memo_ask_as_of"](question="q", as_of="2025-03-10", k=3)

    mock_snap.ask.assert_called_once_with("q", k=3)


def test_memo_diff_returns_envelope(tmp_cfg) -> None:
    """memo_diff must call diff() and return from_ts/to_ts/summary/added/removed/updated."""
    from memo.memory import Memory
    from memo.server_asof import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    from_dt = datetime(2025, 1, 1, tzinfo=UTC)
    to_dt = datetime(2025, 6, 1, tzinfo=UTC)

    added_rec = MagicMock()
    added_rec.id = "new1"
    added_rec.title = "New record"
    added_rec.type = "fact"

    removed_rec = MagicMock()
    removed_rec.id = "old1"
    removed_rec.title = "Old record"
    removed_rec.type = "decision"

    mock_corpus_diff = MagicMock()
    mock_corpus_diff.from_ts = from_dt
    mock_corpus_diff.to_ts = to_dt
    mock_corpus_diff.added = [added_rec]
    mock_corpus_diff.removed = [removed_rec]
    mock_corpus_diff.updated = [{"id": "mid1", "title": "Changed", "changed_fields": ["title"]}]
    mock_corpus_diff.summary.return_value = "1 added · 1 removed · 1 updated"

    with patch("memo.time_machine.diff", return_value=mock_corpus_diff) as mock_diff:
        result = tools["memo_diff"](from_ts="2025-01-01", to_ts="2025-06-01")

    mock_diff.assert_called_once_with(mem, from_ts="2025-01-01", to_ts="2025-06-01")
    assert result["from_ts"] == from_dt.isoformat()
    assert result["to_ts"] == to_dt.isoformat()
    assert result["summary"] == "1 added · 1 removed · 1 updated"
    assert len(result["added"]) == 1
    assert result["added"][0] == {"id": "new1", "title": "New record", "type": "fact"}
    assert len(result["removed"]) == 1
    assert result["removed"][0] == {"id": "old1", "title": "Old record", "type": "decision"}
    assert result["updated"] == [{"id": "mid1", "title": "Changed", "changed_fields": ["title"]}]


def test_memo_diff_defaults_to_ts_to_now(tmp_cfg) -> None:
    """memo_diff must generate a 'now' timestamp for to_ts when omitted."""
    from memo.memory import Memory
    from memo.server_asof import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    from_dt = datetime(2025, 1, 1, tzinfo=UTC)
    to_dt = datetime(2025, 7, 1, tzinfo=UTC)

    mock_corpus_diff = MagicMock()
    mock_corpus_diff.from_ts = from_dt
    mock_corpus_diff.to_ts = to_dt
    mock_corpus_diff.added = []
    mock_corpus_diff.removed = []
    mock_corpus_diff.updated = []
    mock_corpus_diff.summary.return_value = "0 added · 0 removed · 0 updated"

    with patch("memo.time_machine.diff", return_value=mock_corpus_diff) as mock_diff:
        result = tools["memo_diff"](from_ts="2025-01-01")

    assert mock_diff.called
    call_args, call_kwargs = mock_diff.call_args
    # First positional arg is memory; from_ts forwarded; to_ts auto-generated
    assert call_args[0] is mem
    assert call_kwargs["from_ts"] == "2025-01-01"
    assert "to_ts" in call_kwargs, "to_ts must be forwarded even when generated from now()"
    assert result["from_ts"] == from_dt.isoformat()
    assert result["to_ts"] == to_dt.isoformat()
    assert result["summary"] == "0 added · 0 removed · 0 updated"


def test_memo_diff_empty_snapshot(tmp_cfg) -> None:
    """memo_diff must return empty lists when nothing changed between snapshots."""
    from memo.memory import Memory
    from memo.server_asof import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    from_dt = datetime(2025, 3, 1, tzinfo=UTC)
    to_dt = datetime(2025, 3, 2, tzinfo=UTC)

    mock_corpus_diff = MagicMock()
    mock_corpus_diff.from_ts = from_dt
    mock_corpus_diff.to_ts = to_dt
    mock_corpus_diff.added = []
    mock_corpus_diff.removed = []
    mock_corpus_diff.updated = []
    mock_corpus_diff.summary.return_value = "0 added · 0 removed · 0 updated"

    with patch("memo.time_machine.diff", return_value=mock_corpus_diff):
        result = tools["memo_diff"](from_ts="2025-03-01", to_ts="2025-03-02")

    assert result["added"] == []
    assert result["removed"] == []
    assert result["updated"] == []
    assert "0 added" in result["summary"]


def test_no_module_level_mlx_imports() -> None:
    """server_asof must not have module-level MLX imports (deferred-import invariant)."""
    import ast
    import pathlib

    src = pathlib.Path(__file__).parent.parent / "src" / "memo" / "server_asof.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))

    violations = []
    for node in tree.body:  # top-level only
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("mlx"):
                    violations.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("mlx"):
            violations.append(f"line {node.lineno}: from {node.module} import ...")

    assert not violations, f"Module-level MLX imports found: {violations}"
