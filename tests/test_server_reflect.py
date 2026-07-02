"""Tests for server_reflect MCP tool registration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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


def test_register_exposes_only_memo_reflect(tmp_cfg) -> None:
    """register() must expose exactly one tool: memo_reflect."""
    from memo.server_reflect import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    assert {"memo_reflect"} == set(tools), f"Unexpected tools: {set(tools)}"


def test_memo_reflect_no_sessions(tmp_cfg) -> None:
    """memo_reflect returns status=no_sessions when no sessions exist."""
    from memo.server_reflect import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    with patch("memo.session.list_sessions", return_value=[]) as mock_ls:
        result = tools["memo_reflect"]()

    mock_ls.assert_called_once()
    assert result == {"status": "no_sessions"}


def test_memo_reflect_uses_most_recent_session(tmp_cfg) -> None:
    """Without session_id, memo_reflect picks the first session from list_sessions."""
    from memo.server_reflect import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    fake_sessions = [{"session_id": "abc123def", "updated": "2026-07-01T00:00:00"}]
    reflect_result = {"status": "ok", "saved": 3, "session_id": "abc123def"}

    with (
        patch("memo.session.list_sessions", return_value=fake_sessions),
        patch("memo.cli_transcripts._reflect_session", return_value=reflect_result) as mock_rs,
    ):
        result = tools["memo_reflect"]()

    mock_rs.assert_called_once()
    call_args = mock_rs.call_args
    assert call_args.args[0] == "abc123def", "Must pass the resolved session_id"
    assert result == reflect_result


def test_memo_reflect_with_explicit_session_id(tmp_cfg) -> None:
    """When session_id is provided, list_sessions is skipped entirely."""
    from memo.server_reflect import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    reflect_result = {"status": "ok", "saved": 1, "session_id": "myses01"}

    with (
        patch("memo.session.list_sessions") as mock_ls,
        patch("memo.cli_transcripts._reflect_session", return_value=reflect_result) as mock_rs,
    ):
        result = tools["memo_reflect"](session_id="myses01")

    assert not mock_ls.called, "list_sessions must NOT be called when session_id is given"
    mock_rs.assert_called_once()
    assert mock_rs.call_args.args[0] == "myses01"
    assert result == reflect_result


def test_memo_reflect_calls_ensure_chat_before_reflect(tmp_cfg) -> None:
    """memo_reflect must call memory._ensure_chat() before _reflect_session."""
    from memo.server_reflect import register

    call_order: list[str] = []

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg
    mem._ensure_chat.side_effect = lambda: call_order.append("_ensure_chat")

    server, tools = _make_server_and_tools()
    register(server, mem)

    fake_sessions = [{"session_id": "sid0001a", "updated": "2026-07-01T00:00:00"}]
    reflect_result = {"status": "ok", "saved": 0, "session_id": "sid0001a"}

    def fake_reflect(sid, *a, **kw):
        call_order.append("_reflect_session")
        return reflect_result

    with (
        patch("memo.session.list_sessions", return_value=fake_sessions),
        patch("memo.cli_transcripts._reflect_session", side_effect=fake_reflect),
    ):
        tools["memo_reflect"]()

    assert "_ensure_chat" in call_order, "_ensure_chat must be called"
    assert "_reflect_session" in call_order, "_reflect_session must be called"
    assert call_order.index("_ensure_chat") < call_order.index("_reflect_session"), (
        "_ensure_chat must be called before _reflect_session"
    )


def test_memo_reflect_if_due_already_reflected(tmp_cfg) -> None:
    """memo_reflect with if_due=True returns already_reflected when session has reflected_at."""
    from memo.server_reflect import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    fake_sessions = [{"session_id": "done0001", "updated": "2026-07-01T00:00:00"}]
    snap = {"session_id": "done0001", "reflected_at": "2026-07-01T03:00:00"}

    with (
        patch("memo.session.list_sessions", return_value=fake_sessions),
        patch("memo.session.get_session", return_value=snap) as mock_gs,
        patch("memo.cli_transcripts._reflect_session") as mock_rs,
    ):
        result = tools["memo_reflect"](if_due=True)

    mock_gs.assert_called_once()
    assert not mock_rs.called, "_reflect_session must NOT be called for already-reflected session"
    assert result["status"] == "already_reflected"
    assert result["session_id"] == "done0001"
    assert result["reflected_at"] == "2026-07-01T03:00:00"


def test_memo_reflect_if_due_not_yet_reflected(tmp_cfg) -> None:
    """memo_reflect with if_due=True proceeds when session has no reflected_at."""
    from memo.server_reflect import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    fake_sessions = [{"session_id": "new0001a", "updated": "2026-07-01T00:00:00"}]
    snap = {"session_id": "new0001a"}  # no reflected_at
    reflect_result = {"status": "ok", "saved": 2, "session_id": "new0001a"}

    with (
        patch("memo.session.list_sessions", return_value=fake_sessions),
        patch("memo.session.get_session", return_value=snap),
        patch("memo.cli_transcripts._reflect_session", return_value=reflect_result) as mock_rs,
    ):
        result = tools["memo_reflect"](if_due=True)

    mock_rs.assert_called_once()
    assert result == reflect_result


def test_memo_reflect_dry_run_forwarded(tmp_cfg) -> None:
    """memo_reflect passes dry_run=True to _reflect_session."""
    from memo.server_reflect import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    fake_sessions = [{"session_id": "dry0001a", "updated": "2026-07-01T00:00:00"}]
    reflect_result = {"status": "dry_run", "would_save": 4, "session_id": "dry0001a"}

    with (
        patch("memo.session.list_sessions", return_value=fake_sessions),
        patch("memo.cli_transcripts._reflect_session", return_value=reflect_result) as mock_rs,
    ):
        result = tools["memo_reflect"](dry_run=True)

    mock_rs.assert_called_once()
    _kw = mock_rs.call_args.kwargs
    assert _kw.get("dry_run") is True, "dry_run=True must be forwarded to _reflect_session"
    assert result == reflect_result


def test_no_module_level_mlx_imports() -> None:
    """server_reflect must not have module-level MLX imports (deferred-import invariant)."""
    import ast
    import pathlib

    src = pathlib.Path(__file__).parent.parent / "src" / "memo" / "server_reflect.py"
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
