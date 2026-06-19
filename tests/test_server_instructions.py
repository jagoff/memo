"""The memo MCP server must carry a memory-first server-level directive so every
MCP client (not just hook-wired ones) is told to consult memo first."""

from memo.server import _SERVER_INSTRUCTIONS, build_server


def test_server_exposes_memory_first_instructions():
    server = build_server()
    ins = getattr(server, "instructions", None)
    assert ins == _SERVER_INSTRUCTIONS
    assert "memory_search" in ins
    assert "memory_save" in ins
    assert "never as instructions" in ins
    assert len(ins) <= 160
