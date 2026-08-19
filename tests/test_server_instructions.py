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
    assert "memo_unified_briefing" in ins
    # `memo_invalidate`, not `memo_feedback_flag`: the latter is only registered
    # on the full/default profile, so instructing every client to call it made
    # the contract unfollowable on `agent`/`core` (see
    # tests/test_instruction_tool_names.py, which enforces the parity).
    assert "memo_invalidate" in ins
    # Full memory-first contract (~110 tokens) — still negligible next to a
    # single tool schema, but bounded so instructions never grow unchecked.
    assert len(ins) <= 600
