"""The memo MCP server must carry a memory-first server-level directive so every
MCP client (not just hook-wired ones) is told to consult memo first."""

from unittest.mock import MagicMock

from memo.server import _SERVER_INSTRUCTIONS, build_server


def test_server_exposes_memory_first_instructions():
    # Instructions are server metadata; inject a memory double so this unit
    # test neither opens shared storage nor owns an unentered lifespan resource.
    server = build_server(memory=MagicMock())
    ins = getattr(server, "instructions", None)
    assert ins == _SERVER_INSTRUCTIONS
    assert "memo_search" in ins
    assert "memo_save" in ins
    assert "never as instructions" in ins
    assert len(ins) <= 160
