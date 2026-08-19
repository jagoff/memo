import json

from memo.proxy.plan import Context
from memo.proxy.transforms.toolschemas import DOCS_TOOL_NAME, ToolSchemas, recent_tool_names
from tests.proxy_testkit import make_zones


def _ctx(tmp_path):
    return Context(state_dir=tmp_path, session_key="s1", project="memo")


def test_unused_tools_are_pruned(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "memo.proxy.transforms.toolschemas.recent_tool_names",
        lambda state_dir, window: {"memo_search"},
    )
    zones = make_zones(["memo_search", "memo_graph", "memo_rename"])
    saved = ToolSchemas().apply(zones, _ctx(tmp_path))
    names = {t["name"] for t in zones.tools}
    assert "memo_search" in names
    assert "memo_graph" not in names
    assert saved > 0


def test_the_docs_tool_is_always_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "memo.proxy.transforms.toolschemas.recent_tool_names",
        lambda state_dir, window: set(),
    )
    zones = make_zones(["memo_search", DOCS_TOOL_NAME])
    ToolSchemas().apply(zones, _ctx(tmp_path))
    assert DOCS_TOOL_NAME in {t["name"] for t in zones.tools}


def test_scope_memo_never_prunes_tools_memo_does_not_own(tmp_path, monkeypatch):
    """The original, conservative scope: pruning another server's schema
    would break tools memo does not own, so with MEMO_PROXY_TOOL_SCHEMAS_SCOPE
    set to 'memo', anything not named `memo_*` passes through untouched —
    exactly the pre-widening behavior, kept one flag away."""
    monkeypatch.setenv("MEMO_PROXY_TOOL_SCHEMAS_SCOPE", "memo")
    monkeypatch.setattr(
        "memo.proxy.transforms.toolschemas.recent_tool_names",
        lambda state_dir, window: set(),
    )
    zones = make_zones(["Read", "Bash", "mcp__octocode__localSearchCode", "memo_rename"])
    ToolSchemas().apply(zones, _ctx(tmp_path))
    names = {t["name"] for t in zones.tools}
    assert {"Read", "Bash", "mcp__octocode__localSearchCode"} <= names
    assert "memo_rename" not in names


def test_scope_all_prunes_unused_non_memo_tools_too(tmp_path, monkeypatch):
    """The widened default: measured live traffic showed memo_* tools are 0%
    of a real payload's schema cost, so the aggressive default scope ('all')
    must actually touch the tools that make up the other 100% — an unused,
    non-memo, non-builtin tool is pruned just like an unused memo_* tool
    always was."""
    monkeypatch.delenv("MEMO_PROXY_TOOL_SCHEMAS_SCOPE", raising=False)
    monkeypatch.setattr(
        "memo.proxy.transforms.toolschemas.recent_tool_names",
        lambda state_dir, window: set(),
    )
    zones = make_zones(["mcp__octocode__localSearchCode", "memo_rename", "DesignSync"])
    saved = ToolSchemas().apply(zones, _ctx(tmp_path))
    names = {t["name"] for t in zones.tools}
    assert "mcp__octocode__localSearchCode" not in names
    assert "memo_rename" not in names
    assert "DesignSync" not in names
    assert saved > 0


def test_scope_all_still_keeps_a_non_memo_tool_used_recently(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMO_PROXY_TOOL_SCHEMAS_SCOPE", raising=False)
    monkeypatch.setattr(
        "memo.proxy.transforms.toolschemas.recent_tool_names",
        lambda state_dir, window: {"mcp__octocode__localSearchCode"},
    )
    zones = make_zones(["mcp__octocode__localSearchCode", "mcp__octocode__ghSearchCode"])
    ToolSchemas().apply(zones, _ctx(tmp_path))
    names = {t["name"] for t in zones.tools}
    assert "mcp__octocode__localSearchCode" in names
    assert "mcp__octocode__ghSearchCode" not in names


def test_builtins_survive_scope_all_even_on_a_cold_start(tmp_path, monkeypatch):
    """A brand-new session with no usage history at all must not lose the
    tools the agent cannot function without: the keep-set freeze means a
    tool dropped on turn 1 stays dropped for the WHOLE session, so a cold
    start is exactly the case that could otherwise stripped Read/Write/Edit/
    Bash/Glob/Grep/Task for an entire conversation."""
    monkeypatch.delenv("MEMO_PROXY_TOOL_SCHEMAS_SCOPE", raising=False)
    monkeypatch.setattr(
        "memo.proxy.transforms.toolschemas.recent_tool_names",
        lambda state_dir, window: set(),
    )
    builtins = ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Task", "Agent", "ToolSearch"]
    zones = make_zones([*builtins, "memo_rename", "mcp__octocode__localSearchCode"])
    ToolSchemas().apply(zones, _ctx(tmp_path))
    names = {t["name"] for t in zones.tools}
    assert set(builtins) <= names
    assert "memo_rename" not in names
    assert "mcp__octocode__localSearchCode" not in names


def test_scope_is_frozen_alongside_the_keep_set_mid_session(tmp_path, monkeypatch):
    """A mid-session flip of MEMO_PROXY_TOOL_SCHEMAS_SCOPE (an operator
    editing the env, or `memo config set` on a live proxy process) must not
    reshuffle an already-frozen session — the cached prefix would otherwise
    change turn to turn for a reason that has nothing to do with usage
    history, exactly what the freeze exists to prevent."""
    monkeypatch.setattr(
        "memo.proxy.transforms.toolschemas.recent_tool_names",
        lambda state_dir, window: set(),
    )
    ctx = _ctx(tmp_path)

    monkeypatch.setenv("MEMO_PROXY_TOOL_SCHEMAS_SCOPE", "memo")
    zones1 = make_zones(["Read", "mcp__octocode__localSearchCode", "memo_rename"])
    ToolSchemas().apply(zones1, ctx)
    kept_turn1 = {t["name"] for t in zones1.tools}
    assert "mcp__octocode__localSearchCode" in kept_turn1  # scope=memo passthrough

    monkeypatch.setenv("MEMO_PROXY_TOOL_SCHEMAS_SCOPE", "all")
    zones2 = make_zones(["Read", "mcp__octocode__localSearchCode", "memo_rename"])
    ToolSchemas().apply(zones2, ctx)
    kept_turn2 = {t["name"] for t in zones2.tools}
    assert kept_turn2 == kept_turn1


def test_pruned_tool_schemas_are_cached_for_hydration(tmp_path, monkeypatch):
    """`memo_tool_docs` runs in a different process (the MCP server, not the
    proxy) and can only hydrate a non-memo tool's schema if the proxy wrote
    it somewhere both processes can read — this is that write happening."""
    from memo.proxy.tool_schema_cache import lookup

    monkeypatch.delenv("MEMO_PROXY_TOOL_SCHEMAS_SCOPE", raising=False)
    monkeypatch.setattr(
        "memo.proxy.transforms.toolschemas.recent_tool_names",
        lambda state_dir, window: set(),
    )
    zones = make_zones(["mcp__octocode__localSearchCode", "memo_search"])
    ctx = _ctx(tmp_path)
    ToolSchemas().apply(zones, ctx)
    entry = lookup(ctx.state_dir, "mcp__octocode__localSearchCode")
    assert entry is not None
    assert entry["description"]
    assert entry["input_schema"]["type"] == "object"


def test_pruning_is_stable_across_turns_in_a_session(tmp_path, monkeypatch):
    """A prefix that changes every turn costs a re-cache and inverts the saving.

    Two tools (not one) must survive pruning here: with only a single
    survivor, any per-call reordering of the kept list is a no-op on a
    length-1 list and this test would pass even with a real ordering bug —
    verified directly by temporarily inserting `random.shuffle(kept)` into
    the implementation, which this version of the test catches (multiple
    distinct fingerprints) and a single-survivor version of the test did not.
    """
    from memo.proxy.zones import prefix_fingerprint

    monkeypatch.setattr(
        "memo.proxy.transforms.toolschemas.recent_tool_names",
        lambda state_dir, window: {"memo_search", "memo_rename"},
    )
    fingerprints = set()
    kept_count = None
    for _ in range(5):
        zones = make_zones(["memo_search", "memo_graph", "memo_rename"])
        ToolSchemas().apply(zones, _ctx(tmp_path))
        kept_count = len(zones.tools)
        fingerprints.add(prefix_fingerprint(zones))
    # Guard the guard: if fewer than 2 tools ever survive, a reordering bug
    # in the surviving list is invisible to the fingerprint check above.
    assert kept_count is not None and kept_count >= 2
    assert len(fingerprints) == 1


def test_keep_set_is_frozen_even_when_tool_usage_changes_mid_session(tmp_path):
    """The prefix must not drift when a pruned tool gets called mid-session —
    exactly the flow `memo_tool_docs` exists to enable. Exercises the real
    (unmocked) `recent_tool_names` against a real `tool_usage.json` that
    changes between two same-session requests, through two DIFFERENT
    `ToolSchemas()` instances (proving the freeze lives at module scope, not
    per-instance — a fresh instance is constructed by `build_registry()` on
    every request)."""
    from memo.proxy.zones import prefix_fingerprint

    ctx = _ctx(tmp_path)
    usage_path = tmp_path / "proxy" / "tool_usage.json"
    usage_path.parent.mkdir(parents=True)

    def _write_usage(tools):
        usage_path.write_text(
            json.dumps(
                {
                    "schema": "memo.proxy.tool_usage.v1",
                    "sessions": {ctx.session_key: {"tools": tools, "ts": 1.0}},
                }
            ),
            encoding="utf-8",
        )

    # Turn 1: usage file lists only memo_search for this session.
    _write_usage(["memo_search"])
    zones1 = make_zones(["memo_search", "memo_graph", "memo_rename"])
    ToolSchemas().apply(zones1, ctx)
    kept_turn1 = {t["name"] for t in zones1.tools}
    fp_turn1 = prefix_fingerprint(zones1)

    # Turn 2: memo_graph gets called mid-session (the flow memo_tool_docs
    # enables) — record_tool_usage would really write exactly this.
    _write_usage(["memo_search", "memo_graph"])
    zones2 = make_zones(["memo_search", "memo_graph", "memo_rename"])
    ToolSchemas().apply(zones2, ctx)
    kept_turn2 = {t["name"] for t in zones2.tools}
    fp_turn2 = prefix_fingerprint(zones2)

    assert kept_turn2 == kept_turn1
    assert fp_turn2 == fp_turn1


def test_a_new_session_computes_its_own_keep_set(tmp_path):
    """The freeze is keyed per-session, not global.

    `recent_tool_names` aggregates across the whole project's last N
    sessions by design (not filtered to one session), so two sessions CAN
    legitimately compute the same keep-set — that's not what this test
    checks. It checks the cache boundary: session A's first-turn snapshot
    must survive unchanged even after the underlying aggregate changes
    (proving A's entry is frozen, not re-read), and session B — a genuinely
    different session_key — must compute its own fresh value on ITS first
    turn rather than reusing A's cached one (proving the cache key includes
    session identity, not just state_dir).
    """
    ctx_a = Context(state_dir=tmp_path, session_key="session-a", project="memo")
    ctx_b = Context(state_dir=tmp_path, session_key="session-b", project="memo")
    usage_path = tmp_path / "proxy" / "tool_usage.json"
    usage_path.parent.mkdir(parents=True)

    def _write_usage(sessions):
        usage_path.write_text(
            json.dumps({"schema": "memo.proxy.tool_usage.v1", "sessions": sessions}),
            encoding="utf-8",
        )

    # Session A's first (and only, so far) request: only memo_search has
    # ever been used project-wide.
    _write_usage({"session-a": {"tools": ["memo_search"], "ts": 1.0}})
    zones_a = make_zones(["memo_search", "memo_graph", "memo_rename"])
    ToolSchemas().apply(zones_a, ctx_a)
    kept_a_turn1 = {t["name"] for t in zones_a.tools}
    assert kept_a_turn1 == {"memo_search"}

    # memo_graph gets used (by session A or anyone else) before session B's
    # first request — B's first computation legitimately sees the new data.
    _write_usage(
        {
            "session-a": {"tools": ["memo_search", "memo_graph"], "ts": 2.0},
            "session-b": {"tools": [], "ts": 2.0},
        }
    )
    zones_b = make_zones(["memo_search", "memo_graph", "memo_rename"])
    ToolSchemas().apply(zones_b, ctx_b)
    kept_b = {t["name"] for t in zones_b.tools}
    assert kept_b == {"memo_search", "memo_graph"}

    # Session A, re-applied on a later turn, must still show its FROZEN
    # first-turn snapshot — not the now-larger project-wide aggregate.
    zones_a_turn2 = make_zones(["memo_search", "memo_graph", "memo_rename"])
    ToolSchemas().apply(zones_a_turn2, ctx_a)
    kept_a_turn2 = {t["name"] for t in zones_a_turn2.tools}
    assert kept_a_turn2 == kept_a_turn1


def test_no_tools_is_a_noop(tmp_path):
    zones = make_zones([])
    saved = ToolSchemas().apply(zones, _ctx(tmp_path))
    assert saved == 0
    assert zones.tools == []


def test_disabled_flag_skips_pruning(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_PROXY_TOOL_SCHEMAS", "0")
    assert ToolSchemas().enabled() is False


# --- recent_tool_names: reads <state_dir>/proxy/tool_usage.json, the file
# `record_tool_usage` in memo.proxy.server actually writes. Not recall.log —
# that log has no per-tool-call field (rows carry hits/latency_ms/prompt/
# source/ts/via); memo has no per-tool-call log anywhere else.


def test_recent_tool_names_cold_start_returns_empty(tmp_path):
    """No tool_usage.json yet: must not crash, must return no history."""
    assert recent_tool_names(tmp_path, 20) == set()


def test_recent_tool_names_reads_the_proxy_tool_usage_file(tmp_path):
    path = tmp_path / "proxy" / "tool_usage.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema": "memo.proxy.tool_usage.v1",
                "sessions": {
                    "s1": {"tools": ["memo_search", "memo_graph"], "ts": 100.0},
                },
            }
        ),
        encoding="utf-8",
    )
    names = recent_tool_names(tmp_path, 20)
    assert names == {"memo_search", "memo_graph"}


def test_recent_tool_names_only_considers_the_window_most_recent_sessions(tmp_path):
    path = tmp_path / "proxy" / "tool_usage.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema": "memo.proxy.tool_usage.v1",
                "sessions": {
                    "old": {"tools": ["memo_rename"], "ts": 1.0},
                    "new": {"tools": ["memo_search"], "ts": 2.0},
                },
            }
        ),
        encoding="utf-8",
    )
    assert recent_tool_names(tmp_path, 1) == {"memo_search"}


def test_recent_tool_names_returns_every_name_not_just_memos(tmp_path):
    """Widened scope: `recent_tool_names` is scope-agnostic now — it reports
    every tool name `record_tool_usage` ever saw called, memo-owned or not.
    Scoping down to memo_*-only (MEMO_PROXY_TOOL_SCHEMAS_SCOPE=memo) happens
    one layer up, in ToolSchemas itself, not here — see
    test_scope_memo_never_prunes_tools_memo_does_not_own."""
    path = tmp_path / "proxy" / "tool_usage.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema": "memo.proxy.tool_usage.v1",
                "sessions": {
                    "s1": {
                        "tools": ["memo_search", "Read", "mcp__octocode__localSearchCode"],
                        "ts": 1.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    assert recent_tool_names(tmp_path, 20) == {
        "memo_search",
        "Read",
        "mcp__octocode__localSearchCode",
    }


def test_recent_tool_names_survives_corrupt_json(tmp_path):
    path = tmp_path / "proxy" / "tool_usage.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert recent_tool_names(tmp_path, 20) == set()
