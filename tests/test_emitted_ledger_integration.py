"""Integration test: `memo_search` wired to the emission ledger.

Exercises the real MCP tool two ways:

- via the `call_tool` fixture (resolves the registered FastMCP tool's `.fn`
  directly -- see `tests/test_server.py`'s `_tool()` helper), which bypasses
  FastMCP's dispatch and its middleware chain entirely and exercises
  `apply_ledger`'s immediate-write fallback path;
- via a real `fastmcp.Client` against `build_server(...)`, which runs the
  ACTUAL middleware chain (`mcp_budget.make_response_budget_middleware`).
  Required for the F1 tests below: `call_tool`'s `.fn()` shortcut cannot
  exercise the response-budget middleware at all, and F1 is specifically
  about what that middleware does with a payload after the tool body runs.
"""

from __future__ import annotations

import pytest

from memo import emitted_ledger as el
from memo.flags import REGISTRY

_LEDGER_TOOLS_DEFAULT = str(REGISTRY["MEMO_EMITTED_LEDGER_TOOLS"].default)
_LEDGER_MAX_DEFAULT = str(REGISTRY["MEMO_EMITTED_LEDGER_MAX"].default)


@pytest.fixture
def ledger_env(monkeypatch):
    """Pin every MEMO_EMITTED_LEDGER* flag explicitly, not just the on/off
    one -- a developer's ambient shell exports for the tools allowlist or
    entry cap would otherwise leak into what these tests exercise (task-4
    review F5.3)."""
    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "1")
    monkeypatch.setenv("MEMO_EMITTED_LEDGER_TOOLS", _LEDGER_TOOLS_DEFAULT)
    monkeypatch.setenv("MEMO_EMITTED_LEDGER_MAX", _LEDGER_MAX_DEFAULT)
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-int")


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


def test_update_between_searches_reemits_that_memory(memory_with_memories, call_tool, ledger_env):
    first = call_tool("memo_search", query="chat", limit=5)
    target = first["hits"][0]["id"]
    # memo_update's full-body-replace parameter is `content`, not `body`.
    # It signals failure by RETURNING an error dict rather than raising, so
    # a failed update would otherwise present as a bogus ledger failure
    # below instead of the update problem it actually is (task-4 review
    # F5.2 -- observed failing once in ~70 runs before this assertion).
    updated = call_tool("memo_update", id=target, content="rewritten body for the ledger test")
    assert updated is not None and "error" not in updated, f"memo_update failed: {updated}"

    third = call_tool("memo_search", query="chat", limit=5)
    assert target in {h["id"] for h in third["hits"]}
    assert target not in {e["id"] for e in third.get("already_in_context", [])}


def test_body_chars_change_reemits_full_not_a_stale_digest(
    memory_with_memories, call_tool, ledger_env
):
    """F3 (task-4 review): the ledger must hash the text ACTUALLY EMITTED --
    after memo_search's own `body_chars` truncation -- not the stored body.
    memo_search is the first tool where those two can diverge, and the
    fixture memories are short enough that only `body_chars=20` actually
    truncates them.

    Verified by mutation: passing the pre-truncation body as `text_of` (the
    exact forbidden pattern `apply_ledger`'s docstring warns against) left
    every other ledger test green. Only this test catches it: widening
    `body_chars` emits far more text than the narrow call did, so a hit must
    never be digested against it.
    """
    narrow = call_tool("memo_search", query="chat", limit=5, body_chars=20)
    assert narrow["hits"], "fixture must return at least one hit"
    assert all(h["body_truncated"] for h in narrow["hits"]), "fixture bodies must exceed 20 chars"
    assert "already_in_context" not in narrow

    wide = call_tool("memo_search", query="chat", limit=5, body_chars=2000)
    assert {h["id"] for h in wide["hits"]} == {h["id"] for h in narrow["hits"]}
    assert "already_in_context" not in wide


def test_flag_off_leaves_the_payload_untouched(memory_with_memories, call_tool, monkeypatch):
    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "0")
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-off-int")
    first = call_tool("memo_search", query="chat", limit=5)
    second = call_tool("memo_search", query="chat", limit=5)
    assert first["hits"] == second["hits"]
    assert "already_in_context" not in second
    # F2 (task-4 review): read the state dir the tool ACTUALLY writes under.
    # `tmp_cfg` roots the ledger at `tmp_path/"state"`, so reading
    # `tmp_path` itself always returns {} regardless of whether the
    # flag-off guard works -- not a real assertion.
    assert el.read(memory_with_memories.cfg.state_dir, "sess-off-int") == {}


# -- F1 (task-4 review, CRITICAL) --------------------------------------------
#
# apply_ledger runs inside a tool body, which FastMCP executes in a worker
# thread; the response-budget middleware runs outside that, after the tool
# body (and its ledger write) has already completed. Before this fix, a
# search that tripped the budget cap had its whole payload replaced with an
# error -- the model received zero bodies -- while the ledger had already
# recorded every one of them as emitted, so the NEXT search wrongly digested
# content the model never saw. These three go through a REAL fastmcp.Client
# against build_server(...) so the REAL middleware runs; `call_tool`'s
# `.fn()` shortcut above bypasses that chain and cannot exercise this at all.


@pytest.mark.asyncio
async def test_budget_exceeded_call_writes_no_ledger_entries(memory_with_memories, monkeypatch):
    from fastmcp import Client

    from memo.server import build_server

    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "1")
    monkeypatch.setenv("MEMO_EMITTED_LEDGER_TOOLS", _LEDGER_TOOLS_DEFAULT)
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-budget")
    # Tiny enough that this fixture's ordinary 2-hit response trips it --
    # no need to reproduce the reviewer's 40-memory corpus to hit F1.
    monkeypatch.setenv("MEMO_MCP_RESPONSE_BUDGET_TOKENS", "50")

    server = build_server(memory=memory_with_memories)
    async with Client(server) as client:
        result = await client.call_tool("memo_search", {"query": "chat", "limit": 5})

    assert result.structured_content is not None
    assert result.structured_content.get("error") == "response_budget_exceeded"
    assert el.read(memory_with_memories.cfg.state_dir, "sess-budget") == {}


@pytest.mark.asyncio
async def test_in_budget_call_through_the_real_client_still_commits_and_digests(
    memory_with_memories, monkeypatch
):
    """Proves the F1 fix DEFERS the write rather than disabling it: a
    normal in-budget call through the actual middleware chain still records
    its emissions and still digests them on a repeat call."""
    from fastmcp import Client

    from memo.server import build_server

    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "1")
    monkeypatch.setenv("MEMO_EMITTED_LEDGER_TOOLS", _LEDGER_TOOLS_DEFAULT)
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-inbudget")

    server = build_server(memory=memory_with_memories)
    async with Client(server) as client:
        first = await client.call_tool("memo_search", {"query": "chat", "limit": 5})
        second = await client.call_tool("memo_search", {"query": "chat", "limit": 5})

    assert first.structured_content["hits"]
    assert second.structured_content["hits"] == []
    assert second.structured_content["already_in_context"]
    assert el.read(memory_with_memories.cfg.state_dir, "sess-inbudget")


@pytest.mark.asyncio
async def test_tool_body_exception_leaves_the_ledger_empty(memory_with_memories, monkeypatch):
    """apply_ledger stages its writes before the rest of memo_search's body
    runs (the presence bump, the notification read). If something AFTER it
    raises, the caller never receives the payload apply_ledger staged
    entries for -- the ledger must end up empty, not holding a phantom
    record of bodies the model was never shown."""
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    from memo.server import build_server

    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "1")
    monkeypatch.setenv("MEMO_EMITTED_LEDGER_TOOLS", _LEDGER_TOOLS_DEFAULT)
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-raise")

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("memo.presence.bump", _boom)

    server = build_server(memory=memory_with_memories)
    async with Client(server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("memo_search", {"query": "chat", "limit": 5})

    assert el.read(memory_with_memories.cfg.state_dir, "sess-raise") == {}
