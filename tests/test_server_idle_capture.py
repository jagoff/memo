"""Tests for server_idle_capture MCP tool registration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


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


def test_memo_start_session_does_not_spawn_subprocesses(tmp_path: Path, tmp_cfg) -> None:
    """memo_start_session must only call checkpoint(), never spawn background processes."""
    from memo.memory import Memory
    from memo.server_idle_capture import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    assert "memo_start_session" in tools, "memo_start_session not registered"

    with (
        patch("subprocess.Popen") as mock_popen,
        patch(
            "memo.session.checkpoint", return_value={"project": "test", "head_commit": "abc123def"}
        ) as mock_ckpt,
    ):
        result = tools["memo_start_session"](session_id="test-123", cwd=str(tmp_path))

    assert not mock_popen.called, "memo_start_session must NOT spawn subprocesses"
    assert mock_ckpt.called, "memo_start_session must call checkpoint()"
    assert result["status"] == "started"
    assert result["session_id"] == "test-123"


def test_register_exposes_all_four_tools(tmp_cfg) -> None:
    """register() must expose all four expected MCP tools."""
    from memo.memory import Memory
    from memo.server_idle_capture import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    expected = {
        "memo_idle_capture",
        "memo_pop_notification",
        "memo_start_session",
        "memo_save_text",
    }
    assert expected == set(tools), f"Tool mismatch: {set(tools)}"


def test_no_module_level_mlx_imports() -> None:
    """server_idle_capture must not have module-level MLX imports (deferred-import invariant)."""
    import sys

    # Ensure it's not cached
    mod_name = "memo.server_idle_capture"
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    # If mlx is importable, guard it anyway — we want the source check
    import ast
    import pathlib

    src = pathlib.Path(__file__).parent.parent / "src" / "memo" / "server_idle_capture.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))

    violations = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Only flag top-level imports (not inside functions/classes)
            # Top-level nodes are direct children of the module body
            pass  # handled below via module body check

    for node in tree.body:  # top-level only
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("mlx"):
                    violations.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("mlx"):
            violations.append(f"line {node.lineno}: from {node.module} import ...")

    assert not violations, f"Module-level MLX imports found: {violations}"


def test_memo_pop_notification_returns_empty_when_no_file(tmp_cfg) -> None:
    """memo_pop_notification must return '' when no pending notification exists."""
    from memo.memory import Memory
    from memo.server_idle_capture import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    result = tools["memo_pop_notification"]()
    assert result == ""


def test_memo_pop_notification_reads_and_deletes(tmp_cfg) -> None:
    """memo_pop_notification must read and delete the notification file."""
    from memo.memory import Memory
    from memo.server_idle_capture import register

    mem = MagicMock(spec=Memory)
    mem.cfg = tmp_cfg

    server, tools = _make_server_and_tools()
    register(server, mem)

    notif_path = tmp_cfg.state_dir / "pending_idle_notification.txt"
    notif_path.write_text("※ MEMO auto-saved\n", encoding="utf-8")

    result = tools["memo_pop_notification"]()
    assert result == "※ MEMO auto-saved"
    assert not notif_path.exists(), "Notification file should be deleted after reading"

    # Second call should return empty
    result2 = tools["memo_pop_notification"]()
    assert result2 == ""
