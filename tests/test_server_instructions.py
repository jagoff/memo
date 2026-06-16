"""The memo MCP server must carry a memory-first server-level directive so every
MCP client (not just hook-wired ones) is told to consult memo first."""

from memo.server import _SERVER_INSTRUCTIONS, build_server


def test_server_exposes_memory_first_instructions():
    server = build_server()
    ins = getattr(server, "instructions", None)
    assert ins == _SERVER_INSTRUCTIONS
    # Strong-but-bounded: consult-first + the entry tool + the skip carve-out.
    assert "memory_unified_briefing" in ins
    assert "FIRST" in ins
    assert "Skip the lookup only for" in ins
    assert 'source="<your client>"' in ins
