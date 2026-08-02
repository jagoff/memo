"""MCP diagnostics for disabled legacy registered-terminal coordination."""

from __future__ import annotations

import asyncio

import pytest

from memo.server import build_server


@pytest.mark.parametrize("profile", ["agent", "core", "slim", "full", "default"])
def test_every_mcp_profile_exposes_only_read_only_terminal_list(
    mem_with_stub,
    monkeypatch,
    profile: str,
) -> None:
    monkeypatch.setenv("MEMO_MCP_PROFILE", profile)
    server = build_server(memory=mem_with_stub)

    names = {tool.name for tool in asyncio.run(server.list_tools())}

    assert "memo_terminal_list" in names
    assert "memo_terminal_send" not in names
    assert "memo_terminal_enter" not in names

    tool = asyncio.run(server.get_tool("memo_terminal_list"))
    assert tool is not None
    assert tool.fn() == {"terminals": [], "count": 0}
