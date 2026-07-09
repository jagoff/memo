"""By-file search lane over capture-stamped files_read/files_modified arrays."""

from __future__ import annotations

from unittest.mock import MagicMock


def test_search_by_file_filters_on_captured_file_arrays(mock_memory):
    hit = mock_memory.save(
        content="Fixed the recall daemon timeout by raising the socket read budget.",
        title="Recall daemon timeout fix",
        type_="bug",
        extra={"files_modified": ["src/memo/recall_socket.py"], "session_id": "s1"},
    )
    mock_memory.save(
        content="Unrelated note about vault ingest ordering during nightly runs.",
        title="Vault ingest ordering",
        type_="note",
        extra={"files_modified": ["src/memo/cli_ingest.py"]},
    )
    out = mock_memory.search_by_file("daemon timeout", file="recall_socket.py", limit=5)
    assert [r.id for r in out] == [hit.id]


def test_search_by_file_no_match_returns_empty(mock_memory):
    mock_memory.save(
        content="A note that never touched the file in question at all.",
        title="Unrelated",
        type_="note",
        extra={"files_read": ["src/memo/graph.py"]},
    )
    assert mock_memory.search_by_file("anything", file="does_not_exist.py", limit=5) == []


def test_search_by_file_empty_fragment_falls_back_to_plain_search(mock_memory):
    mock_memory.save(content="Plain searchable note about daemons.", title="Plain", type_="note")
    out = mock_memory.search_by_file("daemons", file="", limit=5)
    assert out  # behaves as normal search when no fragment given


def _make_server_and_tools():
    server = MagicMock()
    tools: dict = {}

    def tool_decorator():
        def wrapper(fn):
            tools[fn.__name__] = fn
            return fn

        return wrapper

    server.tool = tool_decorator
    return server, tools


def test_memo_search_routes_file_param_to_search_by_file(tmp_cfg):
    from memo.memory import Memory
    from memo.server_core_search import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.search_by_file.return_value = []
    server, tools = _make_server_and_tools()
    register(server, mem)

    tools["memo_search"](query="daemon timeout", file="recall_socket.py", limit=5)
    mem.search_by_file.assert_called_once_with(
        "daemon timeout",
        file="recall_socket.py",
        limit=5,
        mode="hybrid",
        type_=None,
        quality_rerank=True,
    )
    mem.search.assert_not_called()


def test_memo_search_does_not_run_capture_inline(tmp_cfg, monkeypatch):
    """Read-only MCP search must not block behind transcript capture."""
    import memo.server_core_search as search_server
    from memo.memory import Memory
    from memo.server_core_search import register

    def _capture_should_not_run(*args, **kwargs):
        raise AssertionError("memo_search must not run capture inline")

    monkeypatch.setattr(search_server, "_read_notification", lambda memory: "")
    monkeypatch.setattr("memo.capture.run_capture_incremental", _capture_should_not_run)

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.search.return_value = []
    server, tools = _make_server_and_tools()
    register(server, mem)

    out = tools["memo_search"](query="trust adoption doctor", mode="hybrid", limit=1)

    assert out == {"hits": [], "notification": ""}
    mem.search.assert_called_once_with(
        "trust adoption doctor",
        limit=1,
        type_=None,
        mode="hybrid",
        date_from=None,
        date_to=None,
        quality_rerank=True,
    )


def test_memo_search_trace_opts_into_quality_rerank(tmp_cfg):
    from memo.memory import Memory
    from memo.server_core_search import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.search_with_trace.return_value = {"hits": [], "trace": []}
    server, tools = _make_server_and_tools()
    register(server, mem)

    out = tools["memo_search_trace"](query="alpha", limit=3)

    assert out == {"hits": [], "trace": []}
    mem.search_with_trace.assert_called_once_with(
        "alpha",
        limit=3,
        type_=None,
        mode="hybrid",
        quality_rerank=True,
    )


def test_search_by_file_forwards_quality_rerank(mock_memory, monkeypatch):
    hit = mock_memory.save(
        content="Recall socket change.",
        title="Recall socket",
        extra={"files_modified": ["src/memo/recall_socket.py"]},
    )
    seen: dict[str, object] = {}

    def _fake_search(query: str, **kwargs):
        seen["query"] = query
        seen.update(kwargs)
        return [hit]

    monkeypatch.setattr(mock_memory, "search", _fake_search)

    out = mock_memory.search_by_file(
        "socket",
        file="recall_socket.py",
        limit=1,
        quality_rerank=True,
    )

    assert [r.id for r in out] == [hit.id]
    assert seen["query"] == "socket"
    assert seen["quality_rerank"] is True


def test_memo_ask_does_not_run_capture_inline(tmp_cfg, monkeypatch):
    """Read-only MCP ask must return after answer synthesis and consult logging."""
    import memo.server_core_search as search_server
    from memo.memory import Memory
    from memo.server_core_search import register

    def _capture_should_not_run(*args, **kwargs):
        raise AssertionError("memo_ask must not run capture inline")

    monkeypatch.setattr(search_server, "_read_notification", lambda memory: "")
    monkeypatch.setattr("memo.capture.run_capture_incremental", _capture_should_not_run)

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem.ask.return_value = {"answer": "ok", "citations": []}
    server, tools = _make_server_and_tools()
    register(server, mem)

    out = tools["memo_ask"](question="what changed?", k=1)

    assert out == {"answer": "ok", "citations": [], "notification": ""}
    mem.ask.assert_called_once_with(
        "what changed?",
        k=1,
        type_=None,
        snippet_chars=800,
        include_repos=True,
        session_id=None,
    )
