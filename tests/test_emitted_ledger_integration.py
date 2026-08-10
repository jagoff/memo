"""Integration test: `memo_search` wired to the emission ledger.

Exercises the real MCP tool (via the `call_tool` fixture, which resolves the
registered FastMCP tool's `.fn` -- see `tests/test_server.py`'s `_tool()`
helper for the established pattern) against a real `Memory`, proving the
ledger round-trip end to end rather than unit-testing `apply_ledger` in
isolation (that lives in `tests/test_emitted_ledger_apply.py`).
"""

from __future__ import annotations

import pytest

from memo import emitted_ledger as el


@pytest.fixture
def ledger_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "1")
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-int")
    return tmp_path


def test_repeated_search_digests_the_second_time(memory_with_memories, call_tool, ledger_env):
    first = call_tool("memo_search", query="chat", limit=5)
    assert first["hits"], "fixture must return at least one hit"
    assert "already_in_context" not in first

    second = call_tool("memo_search", query="chat", limit=5)
    assert second["hits"] == []
    assert {e["id"] for e in second["already_in_context"]} == {h["id"] for h in first["hits"]}
    # F6.2 (apply_ledger, task-3 review): a call that digests everything and
    # sends nothing full mints no fresh batch, so top-level `cache_ref` is
    # deliberately absent -- see test_emitted_ledger_apply.py's
    # test_second_identical_call_digests. Each digested entry still carries
    # the ORIGINAL batch's ref from the first call.
    assert "cache_ref" not in second
    assert all(e["ref"].startswith("memo-r/") for e in second["already_in_context"])


def test_update_between_searches_reemits_that_memory(
    memory_with_memories, call_tool, ledger_env
):
    first = call_tool("memo_search", query="chat", limit=5)
    target = first["hits"][0]["id"]
    # memo_update's full-body-replace parameter is `content`, not `body`.
    call_tool("memo_update", id=target, content="rewritten body for the ledger test")

    third = call_tool("memo_search", query="chat", limit=5)
    assert target in {h["id"] for h in third["hits"]}
    assert target not in {e["id"] for e in third.get("already_in_context", [])}


def test_flag_off_leaves_the_payload_untouched(
    memory_with_memories, call_tool, monkeypatch, tmp_path
):
    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "0")
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-off-int")
    first = call_tool("memo_search", query="chat", limit=5)
    second = call_tool("memo_search", query="chat", limit=5)
    assert first["hits"] == second["hits"]
    assert "already_in_context" not in second
    assert el.read(tmp_path, "sess-off-int") == {}
