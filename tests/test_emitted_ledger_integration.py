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

import json

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
async def test_budget_exceeded_digest_call_writes_no_counters(memory_with_memories, monkeypatch):
    """Task 8: a digested batch the middleware then DISCARDS must not be
    counted as a saving -- the caller never actually received the digest
    pointer, so counting tokens_suppressed for it would lie to the promotion
    gate. `apply_ledger`'s counter bump goes through the same `_LEDGER_STAGE`
    the F1 ledger-entry fix uses, so this must discard exactly like
    `test_budget_exceeded_call_writes_no_ledger_entries` above."""
    from fastmcp import Client

    from memo.server import build_server

    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "1")
    monkeypatch.setenv("MEMO_EMITTED_LEDGER_TOOLS", _LEDGER_TOOLS_DEFAULT)
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-budget-digest")

    server = build_server(memory=memory_with_memories)
    async with Client(server) as client:
        first = await client.call_tool("memo_search", {"query": "chat", "limit": 5})
        assert first.structured_content["hits"]

        # The second call would digest everything -- shrink the budget only
        # now so the FIRST call (which populates the ledger) still fits.
        monkeypatch.setenv("MEMO_MCP_RESPONSE_BUDGET_TOKENS", "10")
        second = await client.call_tool("memo_search", {"query": "chat", "limit": 5})

    assert second.structured_content.get("error") == "response_budget_exceeded"
    stats = el.stats(memory_with_memories.cfg.state_dir, "sess-budget-digest")
    assert stats["digests_served"] == 0
    assert stats["tokens_suppressed"] == 0
    assert stats["tokens_digest"] == 0


@pytest.mark.asyncio
async def test_in_budget_call_through_the_real_client_still_commits_and_digests(
    memory_with_memories, monkeypatch
):
    """Proves the F1 fix DEFERS the write rather than disabling it: a
    normal in-budget call through the actual middleware chain still records
    its emissions and still digests them on a repeat call.

    Also pins the task-8 review's F1 fix (whole-row basis, not body-only):
    `expected_suppressed` is computed independently here from the FIRST
    call's actual hits (the full `MemoryRecord.to_dict()` shape memo_search
    returns -- id/path/title/type/tags/created/updated/body/extra/score/...),
    not by re-invoking apply_ledger's own internals, so a regression back to
    charging only `hit["body"]` fails this test rather than just the `> 0`
    check that let the original bug through review.
    """
    from fastmcp import Client

    from memo.mcp_budget import est_tokens
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

    first_hits = first.structured_content["hits"]
    stats = el.stats(memory_with_memories.cfg.state_dir, "sess-inbudget")
    assert stats["digests_served"] == len(second.structured_content["already_in_context"])

    expected_suppressed = sum(
        est_tokens(json.dumps(h, separators=(",", ":"), default=str)) for h in first_hits
    )
    assert stats["tokens_suppressed"] == expected_suppressed
    # Proves the fix, not a coincidence: a body-only basis would have
    # measured far less for these multi-field records.
    body_only = sum(est_tokens(str(h.get("body") or "")) for h in first_hits)
    assert expected_suppressed > body_only * 3

    assert stats["tokens_digest"] > 0
    assert stats["net_saved_est"] == stats["tokens_suppressed"] - stats["tokens_digest"]
    # A real payload's whole-row saving clears the digest stub's fixed
    # overhead -- see task-8-report.md for the exact measured numbers.
    assert stats["net_saved_est"] > 0


@pytest.fixture
def stub_llm(monkeypatch):
    """Stub the MLX chat backend so `memo_ask` exercises its real
    sources-building pipeline without requiring MLX weights or Apple Silicon.
    Mirrors `test_server_sampling.py`'s
    `test_memo_ask_falls_back_to_mlx_without_handler` -- `memory.ask()`
    reaches `Memory._build_mlx_chat()`, which raises without a real MLX
    runtime unless `mlx_available` is patched, then constructs a real
    `MLXChat` unless its `__init__`/`chat` are patched too.
    """
    monkeypatch.setattr("memo.platform_detect.mlx_available", lambda: True)
    monkeypatch.setattr("memo.llm.MLXChat.__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(
        "memo.llm.MLXChat.chat",
        lambda self, model, messages, options=None: {"message": {"content": "stub answer"}},
    )


# -- Task 5: memo_ask / memo_evidence_pack wired; memo_context / -------------
# memo_unified_briefing deliberately left out of the default allowlist. -----
#
# memo_ask returns hits under `sources` (id/title/type/score/snippet --
# `snippet` is the truncated field, see ask_ops.py `_build_ask_context`).
# memo_evidence_pack returns hits under `items` (EvidenceItem.to_dict(), also
# an `id`/`snippet` shape). Both differ from memo_search's `body` default, so
# both need a custom `text_of`.
#
# memo_context's structured `hits` key (`_consult_hits_with_sections`) never
# carries body text at all (id/title/score/section only -- see
# context_surface.py), and the body text that DOES exist lives inside the
# packed `prompt` string (context_pack.py's `_format_section` interpolates
# `row['snippet']` directly into it) -- exactly the disqualifying case the
# task brief calls out: suppressing the bodyless structured list while the
# prompt string still carries every body in full would be a no-op dressed up
# as a feature. memo_unified_briefing returns one `compact_text`-squashed
# markdown string (`compose_unified_briefing`) with no per-hit list to
# partition at all. Both are therefore absent from `MEMO_EMITTED_LEDGER_TOOLS`'s
# default (`flags_misc.py`) and untouched here.


@pytest.mark.parametrize(
    ("tool", "kwargs", "hits_key"),
    [
        ("memo_ask", {"question": "chat"}, "sources"),
        ("memo_evidence_pack", {"question": "chat"}, "items"),
    ],
)
def test_cross_tool_suppression_after_search(
    memory_with_memories, call_tool, ledger_env, stub_llm, tool, kwargs, hits_key
):
    """A memo_search at turn N suppresses the same bodies from a different tool
    at turn N+1 -- the overlap this feature exists to remove."""
    first = call_tool("memo_search", query="chat", limit=5)
    seen = {h["id"] for h in first["hits"]}
    assert seen

    second = call_tool(tool, **kwargs)
    digested = {e["id"] for e in second.get("already_in_context", [])}
    assert digested & seen, f"{tool} did not digest anything memo_search already emitted"
    remaining = {h["id"] for h in second.get(hits_key, []) or []}
    assert not (remaining & digested), "a hit was both emitted and digested"


@pytest.mark.parametrize(
    ("tool", "kwargs", "hits_key"),
    [
        ("memo_ask", {"question": "chat"}, "sources"),
        ("memo_evidence_pack", {"question": "chat"}, "items"),
    ],
)
def test_repeated_call_digests_the_second_time(
    memory_with_memories, call_tool, ledger_env, stub_llm, tool, kwargs, hits_key
):
    first = call_tool(tool, **kwargs)
    first_ids = {h["id"] for h in first.get(hits_key, []) or []}
    assert first_ids, f"{tool} fixture must return at least one hit"
    assert "already_in_context" not in first

    second = call_tool(tool, **kwargs)
    second_ids = {h["id"] for h in second.get(hits_key, []) or []}
    digested_ids = {e["id"] for e in second.get("already_in_context", [])}
    assert not second_ids, f"{tool} should have digested every previously-seen id"
    assert digested_ids == first_ids


def test_ask_snippet_chars_change_reemits_full_not_a_stale_digest(
    memory_with_memories, call_tool, ledger_env, stub_llm
):
    """F3 analog (task-4 review) for memo_ask: proves a widened `snippet_chars`
    is never mistaken for text already digested at a narrower one. Unlike
    memo_search, `sources` rows carry no separate untruncated-body field a
    buggy `text_of` could read instead of `snippet` -- that class of bug is
    instead caught by `test_cross_tool_suppression_after_search` /
    `test_repeated_call_digests_the_second_time` failing outright (checked:
    swapping in the module's default `text_of`, which reads a `body` key
    these rows don't have, makes `apply_ledger` treat every hit as
    id/text-less and silently never digest -- see task-5-report.md)."""
    narrow = call_tool("memo_ask", question="chat", snippet_chars=10)
    assert narrow["sources"], "fixture must return at least one source"
    # 10 chars plus the ellipsis memo_ask appends when it truncates -- proves
    # the snippet was actually cut, without pinning the exact ellipsis math.
    assert all(len(s["snippet"]) <= 11 for s in narrow["sources"])
    assert "already_in_context" not in narrow

    wide = call_tool("memo_ask", question="chat", snippet_chars=2000)
    assert {s["id"] for s in wide["sources"]} == {s["id"] for s in narrow["sources"]}
    assert "already_in_context" not in wide


def test_ask_never_digests_a_repo_sourced_row(
    memory_with_memories, call_tool, ledger_env, stub_llm, monkeypatch
):
    """F1 (task-5 review): `ask_ops._build_ask_context` appends repo-corpus
    rows (`source == "repo"`) into the same `sources` list as memory rows
    whenever a repo is indexed -- `include_repos=True` is memo_ask's default,
    gated only by `self.store.list_repo_sources(limit=1)`. A repo row's id
    is NOT a memory id: `memo_get(id)` -- the digest's own escape hatch --
    cannot resolve it. Only `source == "memory"` rows may participate;
    everything else must always be sent in full and never recorded, the
    same as an id-less/bodyless hit.
    """
    import types

    repo_hit = types.SimpleNamespace(
        id="deadbeefdeadbeefdeadbeefdeadbeef",
        path="src/memo/chat/pipeline.py",
        locator="pipeline.py:10-20",
        text="repo chunk text about the chat pipeline internals " * 5,
        repo_name="memo",
        score=0.9,
        line_start=10,
        line_end=20,
        match_type="hybrid",
    )
    monkeypatch.setattr(
        memory_with_memories.store, "list_repo_sources", lambda *a, **k: [{"repo_name": "memo"}]
    )
    monkeypatch.setattr(memory_with_memories, "repo_search", lambda *a, **k: [repo_hit])

    first = call_tool("memo_ask", question="chat")
    by_id = {s["id"]: s for s in first["sources"]}
    assert by_id.get(repo_hit.id, {}).get("source") == "repo", (
        "fixture must actually exercise a repo-sourced row"
    )
    assert any(s.get("source") == "memory" for s in first["sources"]), (
        "fixture must also return at least one memory row"
    )

    second = call_tool("memo_ask", question="chat")
    digested_ids = {e["id"] for e in second.get("already_in_context", [])}
    assert repo_hit.id not in digested_ids, "a repo row must never be recorded/digested"
    remaining_ids = {s["id"] for s in second["sources"]}
    assert repo_hit.id in remaining_ids, "a repo row must be returned in full every time"
    assert digested_ids, "memory rows must still digest normally alongside an ignored repo row"


def test_evidence_pack_max_chars_change_reemits_full_not_a_stale_digest(
    memory_with_memories, call_tool, ledger_env
):
    """F3 analog for memo_evidence_pack: unlike memo_search's `body_chars` (a
    per-hit cap) or memo_ask's `snippet_chars` (also per-hit), evidence_pack's
    `max_chars` is a single RUNNING budget shared across all items
    (`_build_items`'s `remaining`), so a narrow budget can drop an item
    entirely rather than merely shortening it. The invariant under test is
    narrower as a result: a widened budget must never be treated as
    already-seen, not that the two calls return identical id sets.
    """
    long_body = "chat pipeline notes. " * 60
    memory_with_memories.save(content=long_body, title="Long chat memory")

    narrow = call_tool("memo_evidence_pack", question="chat", max_chars=256)
    assert narrow["items"], "fixture must return at least one evidence item"
    assert "already_in_context" not in narrow

    wide = call_tool("memo_evidence_pack", question="chat", max_chars=12_000)
    assert "already_in_context" not in wide


def test_context_and_briefing_are_not_in_the_default_allowlist():
    """Pins the Task 5 scope decision at the config-surface level: a tool
    that can never digest must not be lying in the allowlist."""
    allow = {t.strip() for t in _LEDGER_TOOLS_DEFAULT.split(",") if t.strip()}
    assert "memo_context" not in allow
    assert "memo_unified_briefing" not in allow
    assert {"memo_search", "memo_ask", "memo_evidence_pack"} <= allow


def test_context_never_suppresses_even_after_search(memory_with_memories, call_tool, ledger_env):
    first = call_tool("memo_search", query="chat", limit=5)
    assert first["hits"]

    second = call_tool("memo_context", question="chat")
    assert "already_in_context" not in second
    assert second.get("hits"), "memo_context must keep returning its structured hit rows"


def test_unified_briefing_never_suppresses_even_after_search(
    memory_with_memories, call_tool, ledger_env
):
    first = call_tool("memo_search", query="chat", limit=5)
    assert first["hits"]

    second = call_tool("memo_unified_briefing")
    assert "already_in_context" not in second
    assert second["markdown"]


@pytest.mark.parametrize(
    ("tool", "kwargs", "hits_key"),
    [
        ("memo_ask", {"question": "chat"}, "sources"),
        ("memo_context", {"question": "chat"}, "hits"),
        ("memo_evidence_pack", {"question": "chat"}, "items"),
    ],
)
def test_flag_off_leaves_each_tool_untouched(
    memory_with_memories, call_tool, monkeypatch, stub_llm, tool, kwargs, hits_key
):
    """With MEMO_EMITTED_LEDGER off, memo_ask/memo_evidence_pack (wired) and
    memo_context (deliberately not) must all behave exactly as before this
    feature existed: repeating the same call never suppresses a hit."""
    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "0")
    monkeypatch.setenv("MEMO_SESSION_ID", f"sess-off-{tool}")

    first = call_tool(tool, **kwargs)
    second = call_tool(tool, **kwargs)

    first_ids = {h["id"] for h in first.get(hits_key, []) or [] if h.get("id")}
    second_ids = {h["id"] for h in second.get(hits_key, []) or [] if h.get("id")}
    assert first_ids, f"{tool} fixture must return at least one id-bearing hit"
    assert first_ids == second_ids
    assert "already_in_context" not in second
    assert el.read(memory_with_memories.cfg.state_dir, f"sess-off-{tool}") == {}


def test_flag_off_leaves_unified_briefing_untouched(memory_with_memories, call_tool, monkeypatch):
    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "0")
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-off-briefing")

    first = call_tool("memo_unified_briefing")
    second = call_tool("memo_unified_briefing")

    assert first["markdown"] == second["markdown"]
    assert "already_in_context" not in second
    assert el.read(memory_with_memories.cfg.state_dir, "sess-off-briefing") == {}


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


def test_hook_emission_suppresses_a_later_search(memory_with_memories, call_tool, ledger_env):
    """The largest real overlap: the recall hook injected it at turn 2, so
    memo_search must not re-send the same body at turn 3."""
    from memo import emitted_ledger as el

    state_dir = memory_with_memories.cfg.state_dir
    first = call_tool("memo_search", query="chat", limit=5)
    target = first["hits"][0]
    el.reset(state_dir, "sess-int")

    # Simulate the hook having injected exactly this rendering.
    body = target["body"]
    el.append(
        state_dir,
        "sess-int",
        [
            el.Entry(
                id=target["id"],
                h=el.emitted_hash(body),
                n=len(body),
                ref="memo-h/abc123",
                t=1,
                src="hook",
            )
        ],
    )

    second = call_tool("memo_search", query="chat", limit=5)
    digested = {e["id"]: e for e in second.get("already_in_context", [])}
    assert target["id"] in digested
    assert digested[target["id"]]["ref"] == "memo-h/abc123"

    assert el.read(memory_with_memories.cfg.state_dir, "sess-raise") == {}


# -- Task 8: memo_get's recovery counter -------------------------------------
#
# `memo_get` is deliberately absent from MEMO_EMITTED_LEDGER_TOOLS (it is the
# digest's own escape hatch), so it never consults `apply_ledger`. What it
# does instead: if the id it was asked for already has an entry in this
# session's ledger -- meaning some earlier call already put a rendering of it
# into the window -- a memo_get for it counts as a RECOVERY against the
# feature's own saving, per the spec's conservative-attribution rule.


def test_memo_get_after_digest_bumps_the_recovery_counter(
    memory_with_memories, call_tool, ledger_env
):
    from memo.mcp_budget import est_tokens

    first = call_tool("memo_search", query="chat", limit=5)
    target_id = first["hits"][0]["id"]  # memo_search's own call already ledgers this id

    fetched = call_tool("memo_get", id=target_id)
    assert fetched is not None

    stats = el.stats(memory_with_memories.cfg.state_dir, "sess-int")
    assert stats["memo_get_after_digest"] == 1
    # F1 (task-8 review): tokens_recovered must charge the whole record
    # memo_get actually returns (the full `rec.to_dict()`), not just its
    # `body` field -- that full dict is what the caller actually paid for.
    expected_tokens_recovered = est_tokens(json.dumps(fetched, separators=(",", ":"), default=str))
    body_only = est_tokens(str(fetched.get("body") or ""))
    assert expected_tokens_recovered > body_only * 3  # proves the fix, not a coincidence
    # Nothing was suppressed/digested by memo_search itself in this test (it
    # was the first call), so the only term left is the recovery cost --
    # net_saved_est must go negative by exactly that amount.
    assert stats["net_saved_est"] == -expected_tokens_recovered


def test_memo_get_recovery_survives_a_broken_flag_read(
    memory_with_memories, call_tool, ledger_env, monkeypatch
):
    """F3 (task-8 review): the fail-open envelope around
    `_record_ledger_recovery` was untested -- deleting its try/except left
    every other test in this file green, and the failure it guards is real:
    `flag_bool` resolves through `config_md.load_values`, whose
    `path.read_text(encoding="utf-8")` is guarded only by `except OSError`,
    so a non-UTF-8 byte in a user's markdown config raises
    `UnicodeDecodeError` straight out of `flag_bool`. With the envelope
    removed that makes `memo_get` raise on EVERY call for that user.
    Mirrors `test_counter_bump_failure_does_not_break_the_actual_
    suppression` (test_emitted_ledger_apply.py) on the apply_ledger side.
    """
    import memo.flags as flags_mod

    first = call_tool("memo_search", query="chat", limit=5)
    target_id = first["hits"][0]["id"]

    calls: list[object] = []

    def boom(*args: object, **kwargs: object) -> bool:
        calls.append(args)
        raise RuntimeError("boom")

    monkeypatch.setattr(flags_mod, "flag_bool", boom)

    fetched = call_tool("memo_get", id=target_id)
    assert calls, "expected flag_bool() to be called (and raise) before the assertion"
    assert fetched is not None
    assert fetched["id"] == target_id


def test_memo_get_with_a_prefix_still_counts_as_a_recovery(
    memory_with_memories, call_tool, ledger_env
):
    """F4 (task-8 review): `memo_get` accepts a short id prefix -- its own
    docstring says so, and it's the normal way an agent calls it, not the
    exception. `_record_ledger_recovery` must check the ledger against the
    RESOLVED full id (`rec.id`), not the caller's raw `id` argument, or a
    prefix lookup on a previously-emitted memory silently fails the
    `memory_id not in el.read(...)` check and is never counted. Every other
    recovery test in this file passes a full id (taken straight from
    memo_search's hits), so none of them exercise this path -- confirmed by
    mutation: swapping `rec.id` for the raw `id` parameter in
    `_record_ledger_recovery` keeps every OTHER test in this file green."""
    first = call_tool("memo_search", query="chat", limit=5)
    target_id = first["hits"][0]["id"]
    prefix = target_id[:8]

    fetched = call_tool("memo_get", id=prefix)
    assert fetched is not None
    assert fetched["id"] == target_id

    stats = el.stats(memory_with_memories.cfg.state_dir, "sess-int")
    assert stats["memo_get_after_digest"] == 1


def test_memo_get_on_a_never_ledgered_id_does_not_bump_the_recovery_counter(
    memory_with_memories, call_tool, ledger_env
):
    """No prior emission for this id in the session ledger -- e.g. the very
    first tool call of a session -- means there is nothing to recover FROM."""
    first = call_tool("memo_search", query="chat", limit=5)
    target_id = first["hits"][0]["id"]
    el.reset(memory_with_memories.cfg.state_dir, "sess-int")  # wipe what memo_search just wrote

    fetched = call_tool("memo_get", id=target_id)
    assert fetched is not None

    stats = el.stats(memory_with_memories.cfg.state_dir, "sess-int")
    assert stats["memo_get_after_digest"] == 0
    assert stats["net_saved_est"] == 0


def test_memo_get_recovery_counter_stays_off_with_the_flag_off(
    memory_with_memories, call_tool, ledger_env, monkeypatch
):
    first = call_tool("memo_search", query="chat", limit=5)
    target_id = first["hits"][0]["id"]

    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "0")
    fetched = call_tool("memo_get", id=target_id)
    assert fetched is not None

    stats = el.stats(memory_with_memories.cfg.state_dir, "sess-int")
    assert stats["memo_get_after_digest"] == 0


def test_discarding_the_ledger_stage_drops_staged_counter_bumps_too(tmp_path):
    """Symmetric with test_budget_exceeded_digest_call_writes_no_counters, at
    the mechanism level: memo_get's recovery bump (`server_common.
    stage_counters`) shares the exact same `_LEDGER_STAGE` that
    `apply_ledger`'s entries use, so `discard_ledger_stage` must drop a
    staged counter bump exactly like it drops a staged ledger entry.

    (Not exercised end-to-end through a real fastmcp.Client for memo_get: its
    `dict[str, Any] | None` return type makes FastMCP wrap normal responses
    as `{"result": ...}`, but the budget-exceeded substitution in
    `mcp_budget.make_response_budget_middleware` returns the raw error dict
    unwrapped -- a pre-existing schema-validation mismatch on the client
    side, unrelated to this feature and out of this task's scope. See
    task-8-report.md.)
    """
    from memo import server_common as sc

    token = sc.open_ledger_stage()
    try:
        sc.stage_counters(tmp_path, "sess-stage", get_after_digest=1, tokens_recovered=50)
    finally:
        # try/finally, not a bare sequential call: `_LEDGER_STAGE` is a plain
        # ContextVar with no scoping construct of its own -- an exception
        # between open and discard/commit would leave it bound to a stale
        # list for the rest of this (synchronous, single-threaded) test
        # process, silently turning every later test's apply_ledger call
        # into a stage nothing ever commits. Observed firsthand while this
        # test was still red (a typo before the real implementation existed
        # cascaded into unrelated tests elsewhere in this file).
        sc.discard_ledger_stage(token)

    stats = el.stats(tmp_path, "sess-stage")
    assert stats["memo_get_after_digest"] == 0
    assert stats["net_saved_est"] == 0


def test_committing_the_ledger_stage_writes_staged_counter_bumps(tmp_path):
    from memo import server_common as sc

    token = sc.open_ledger_stage()
    try:
        sc.stage_counters(tmp_path, "sess-stage", get_after_digest=1, tokens_recovered=50)
    finally:
        sc.commit_ledger_stage(token)

    stats = el.stats(tmp_path, "sess-stage")
    assert stats["memo_get_after_digest"] == 1
    assert stats["net_saved_est"] == -50
