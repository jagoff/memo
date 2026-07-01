"""Tests for server_core_history MCP tool registration."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from memo.memory import AmbiguousIdError, Memory


def _make_server_and_tools() -> tuple[Any, dict]:
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


def _make_mem(tmp_cfg) -> MagicMock:
    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    # history and store are set as instance attributes in Memory.__init__,
    # so spec=Memory does not auto-expose them. Wire them manually.
    mem.history = MagicMock()
    mem.store = MagicMock()
    return mem


# ---------------------------------------------------------------------------
# Presence
# ---------------------------------------------------------------------------


def test_register_exposes_all_six_tools(tmp_cfg) -> None:
    """register() must expose exactly the six expected MCP tools."""
    from memo.server_core_history import register

    mem = _make_mem(tmp_cfg)
    server, tools = _make_server_and_tools()
    register(server, mem)

    expected = {
        "memo_provenance",
        "memo_record_diff",
        "memo_history",
        "memo_session_list",
        "memo_session_get",
        "memo_stats",
    }
    assert expected == set(tools), f"Tool mismatch: {set(tools)}"


# ---------------------------------------------------------------------------
# memo_provenance
# ---------------------------------------------------------------------------


def test_memo_provenance_delegates_to_memory(tmp_cfg) -> None:
    """memo_provenance must delegate to memory.provenance and return its result."""
    from memo.server_core_history import register

    mem = _make_mem(tmp_cfg)
    mem.provenance.return_value = {"id": "abc123", "source": "cli"}

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_provenance"](id="abc123")

    mem.provenance.assert_called_once_with("abc123")
    assert result == {"id": "abc123", "source": "cli"}


def test_memo_provenance_none_when_not_found(tmp_cfg) -> None:
    """memo_provenance must return None when memory.provenance returns None."""
    from memo.server_core_history import register

    mem = _make_mem(tmp_cfg)
    mem.provenance.return_value = None

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_provenance"](id="nonexistent")
    assert result is None


# ---------------------------------------------------------------------------
# memo_record_diff
# ---------------------------------------------------------------------------

_FULL_ID = "a" * 32  # 32 chars → short-id branch is skipped


def _fake_record(title: str = "Test Memory", mem_type: str = "fact") -> MagicMock:
    r = MagicMock()
    r.title = title
    r.type = mem_type
    return r


def test_memo_record_diff_full_id_returns_envelope(tmp_cfg) -> None:
    """memo_record_diff with a 32-char id must not call resolve_id and return envelope."""
    from memo.server_core_history import register

    mem = _make_mem(tmp_cfg)
    mem.get.return_value = _fake_record()
    mem.history.list_recent.return_value = [{"op": "save", "id": _FULL_ID}]

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_record_diff"](id=_FULL_ID, limit=10)

    mem.resolve_id.assert_not_called()
    mem.get.assert_called_once_with(_FULL_ID)
    mem.history.list_recent.assert_called_once_with(limit=10, record_id=_FULL_ID)

    assert result["id"] == _FULL_ID
    assert result["title"] == "Test Memory"
    assert result["type"] == "fact"
    assert isinstance(result["events"], list)
    assert result["returned_events"] == 1
    assert result["has_more"] is False


def test_memo_record_diff_short_id_resolves(tmp_cfg) -> None:
    """memo_record_diff with a short id must resolve it via memory.resolve_id."""
    from memo.server_core_history import register

    short_id = "abc123"
    mem = _make_mem(tmp_cfg)
    mem.resolve_id.return_value = _FULL_ID
    mem.get.return_value = _fake_record(title="Resolved", mem_type="decision")
    mem.history.list_recent.return_value = []

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_record_diff"](id=short_id)

    mem.resolve_id.assert_called_once_with(short_id)
    mem.get.assert_called_once_with(_FULL_ID)
    assert result["id"] == _FULL_ID
    assert result["title"] == "Resolved"
    assert result["returned_events"] == 0
    assert result["has_more"] is False


def test_memo_record_diff_ambiguous_id_returns_error(tmp_cfg) -> None:
    """memo_record_diff must return an error envelope when resolve_id is ambiguous."""
    from memo.server_core_history import register

    short_id = "dup"
    mem = _make_mem(tmp_cfg)
    mem.resolve_id.side_effect = AmbiguousIdError(short_id, ["dup0000abc", "dup1111xyz"])

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_record_diff"](id=short_id)

    assert result["error"] == "ambiguous"
    assert result["prefix"] == short_id
    assert "dup0000abc" in result["matches"]
    assert "dup1111xyz" in result["matches"]


def test_memo_record_diff_has_more_when_events_fill_limit(tmp_cfg) -> None:
    """has_more is True when returned events count equals the limit."""
    from memo.server_core_history import register

    mem = _make_mem(tmp_cfg)
    mem.get.return_value = _fake_record()
    mem.history.list_recent.return_value = [{"op": "update"}] * 5  # exactly limit

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_record_diff"](id=_FULL_ID, limit=5)

    assert result["has_more"] is True
    assert result["returned_events"] == 5


# ---------------------------------------------------------------------------
# memo_history
# ---------------------------------------------------------------------------


def test_memo_history_no_id_returns_recent(tmp_cfg) -> None:
    """memo_history with no id must call list_recent without record_id."""
    from memo.server_core_history import register

    mem = _make_mem(tmp_cfg)
    mem.history.list_recent.return_value = [{"op": "save"}, {"op": "update"}]

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_history"](limit=20)

    mem.history.list_recent.assert_called_once_with(limit=20, op=None, record_id=None)
    assert isinstance(result, list)
    assert len(result) == 2


def test_memo_history_with_full_id_passes_to_list_recent(tmp_cfg) -> None:
    """memo_history with a 32-char id must pass it straight to list_recent."""
    from memo.server_core_history import register

    mem = _make_mem(tmp_cfg)
    mem.history.list_recent.return_value = []

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_history"](limit=5, id=_FULL_ID)

    mem.resolve_id.assert_not_called()
    mem.history.list_recent.assert_called_once_with(limit=5, op=None, record_id=_FULL_ID)
    assert result == []


def test_memo_history_short_id_resolves_before_list(tmp_cfg) -> None:
    """memo_history with a short id must resolve it before calling list_recent."""
    from memo.server_core_history import register

    short_id = "short"
    mem = _make_mem(tmp_cfg)
    mem.resolve_id.return_value = _FULL_ID
    mem.history.list_recent.return_value = [{"op": "delete"}]

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_history"](limit=10, id=short_id)

    mem.resolve_id.assert_called_once_with(short_id)
    mem.history.list_recent.assert_called_once_with(limit=10, op=None, record_id=_FULL_ID)
    assert result[0]["op"] == "delete"


def test_memo_history_short_id_not_found_returns_empty(tmp_cfg) -> None:
    """memo_history must return [] when resolve_id returns None (id not found)."""
    from memo.server_core_history import register

    mem = _make_mem(tmp_cfg)
    mem.resolve_id.return_value = None

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_history"](limit=10, id="gone")

    assert result == []
    mem.history.list_recent.assert_not_called()


def test_memo_history_ambiguous_id_returns_error_list(tmp_cfg) -> None:
    """memo_history must return a single-element error list when id is ambiguous."""
    from memo.server_core_history import register

    short_id = "dup"
    mem = _make_mem(tmp_cfg)
    mem.resolve_id.side_effect = AmbiguousIdError(short_id, ["dup00000001", "dup00000002"])

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_history"](id=short_id)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["error"] == "ambiguous"
    assert result[0]["prefix"] == short_id


def test_memo_history_filters_by_op(tmp_cfg) -> None:
    """memo_history must pass the op filter down to list_recent."""
    from memo.server_core_history import register

    mem = _make_mem(tmp_cfg)
    mem.history.list_recent.return_value = [{"op": "save"}]

    server, tools = _make_server_and_tools()
    register(server, mem)

    tools["memo_history"](limit=5, op="save")

    mem.history.list_recent.assert_called_once_with(limit=5, op="save", record_id=None)


# ---------------------------------------------------------------------------
# memo_session_list
# ---------------------------------------------------------------------------


def test_memo_session_list_calls_list_sessions(tmp_cfg) -> None:
    """memo_session_list must delegate to list_sessions with correct args."""
    from memo.server_core_history import register

    mem = _make_mem(tmp_cfg)
    server, tools = _make_server_and_tools()
    register(server, mem)

    fake_sessions = [{"session_id": "s1"}, {"session_id": "s2"}]
    with patch("memo.session.list_sessions", return_value=fake_sessions) as mock_ls:
        result = tools["memo_session_list"](limit=5, project="myproject")

    mock_ls.assert_called_once_with(tmp_cfg.state_dir, limit=5, project="myproject")
    assert result == fake_sessions


def test_memo_session_list_defaults(tmp_cfg) -> None:
    """memo_session_list default args (limit=10, project=None) must be forwarded."""
    from memo.server_core_history import register

    mem = _make_mem(tmp_cfg)
    server, tools = _make_server_and_tools()
    register(server, mem)

    with patch("memo.session.list_sessions", return_value=[]) as mock_ls:
        result = tools["memo_session_list"]()

    mock_ls.assert_called_once_with(tmp_cfg.state_dir, limit=10, project=None)
    assert result == []


# ---------------------------------------------------------------------------
# memo_session_get
# ---------------------------------------------------------------------------


def test_memo_session_get_returns_session(tmp_cfg) -> None:
    """memo_session_get must delegate to get_session and return its result."""
    from memo.server_core_history import register

    mem = _make_mem(tmp_cfg)
    server, tools = _make_server_and_tools()
    register(server, mem)

    fake_session = {"session_id": "abc-123", "project": "demo"}
    with patch("memo.session.get_session", return_value=fake_session) as mock_gs:
        result = tools["memo_session_get"](session_id="abc-123")

    mock_gs.assert_called_once_with(tmp_cfg.state_dir, "abc-123")
    assert result == fake_session


def test_memo_session_get_none_when_missing(tmp_cfg) -> None:
    """memo_session_get must return None when get_session returns None."""
    from memo.server_core_history import register

    mem = _make_mem(tmp_cfg)
    server, tools = _make_server_and_tools()
    register(server, mem)

    with patch("memo.session.get_session", return_value=None):
        result = tools["memo_session_get"](session_id="no-such-session")

    assert result is None


# ---------------------------------------------------------------------------
# memo_stats
# ---------------------------------------------------------------------------


def test_memo_stats_returns_core_fields(tmp_cfg) -> None:
    """memo_stats must include total, data_dir, db_path, and embedder_model."""
    from memo.server_core_history import register

    mem = _make_mem(tmp_cfg)
    mem.store.count.return_value = 42
    # history.error_count: accessed via getattr with default 0
    mem.history.error_count = 0

    server, tools = _make_server_and_tools()
    register(server, mem)

    # Suppress the optional recall_health import gracefully
    with patch("memo.dashboard.recall_health", return_value={"status": "ok"}, create=True):
        result = tools["memo_stats"]()

    assert result["total"] == 42
    assert result["data_dir"] == str(tmp_cfg.data_dir)
    assert result["db_path"] == str(tmp_cfg.db_path)
    assert result["embedder_model"] == tmp_cfg.embedder_model
    assert "history_errors" in result


def test_memo_stats_count_called(tmp_cfg) -> None:
    """memo_stats must call memory.store.count() exactly once."""
    from memo.server_core_history import register

    mem = _make_mem(tmp_cfg)
    mem.store.count.return_value = 7

    server, tools = _make_server_and_tools()
    register(server, mem)

    tools["memo_stats"]()

    mem.store.count.assert_called_once()


def test_memo_stats_vault_path_none_when_not_set(tmp_cfg) -> None:
    """memo_stats vault_path field must be None when cfg.vault_path is None."""
    from memo.server_core_history import register

    mem = _make_mem(tmp_cfg)
    # Override vault_path to None on the cfg copy
    mem.cfg = MagicMock()
    mem.cfg.data_dir = tmp_cfg.data_dir
    mem.cfg.state_dir = tmp_cfg.state_dir
    mem.cfg.db_path = tmp_cfg.db_path
    mem.cfg.embedder_model = tmp_cfg.embedder_model
    mem.cfg.vault_path = None
    mem.store.count.return_value = 0
    mem.history.error_count = 0

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_stats"]()

    assert result["vault_path"] is None
