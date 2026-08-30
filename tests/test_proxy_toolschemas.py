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


def test_keep_set_survives_a_proxy_restart(tmp_path, monkeypatch):
    """The freeze must outlive the process, not just the request.

    `com.memo.proxy` runs under launchd KeepAlive. Before the keep-set was
    persisted, a restart mid-session re-derived it from a `tool_usage.json`
    that had grown in the meantime, so the `tools` array changed shape while
    the session was still going. `tools` sits in front of `system` in the
    cached prefix, so that one change re-caches the WHOLE conversation at the
    1.25x creation premium — measured at ~700k tokens on a real session,
    dwarfing everything this transform saves.
    """
    from memo.proxy.transforms import toolschemas

    usage = {"memo_search"}
    monkeypatch.setattr(
        "memo.proxy.transforms.toolschemas.recent_tool_names",
        lambda state_dir, window: set(usage),
    )

    zones = make_zones(["memo_search", "memo_graph", "memo_rename"])
    ToolSchemas().apply(zones, _ctx(tmp_path))
    before = sorted(t["name"] for t in zones.tools)

    # The restart: in-process freeze gone, and the session has since called a
    # tool it had not called when the keep-set was first frozen.
    toolschemas._session_keep_cache.clear()
    usage.add("memo_graph")

    zones = make_zones(["memo_search", "memo_graph", "memo_rename"])
    ToolSchemas().apply(zones, _ctx(tmp_path))
    after = sorted(t["name"] for t in zones.tools)

    assert after == before, "keep-set moved across a restart; prefix re-cache"
    assert "memo_graph" not in after


def test_a_different_session_is_frozen_independently(tmp_path, monkeypatch):
    """Persistence is per session, not global: a NEW session must still get
    an up-to-date keep-set, otherwise the first session on a machine would
    pin every later one to its own cold-start tool history forever."""
    from memo.proxy.transforms import toolschemas

    usage = {"memo_search"}
    monkeypatch.setattr(
        "memo.proxy.transforms.toolschemas.recent_tool_names",
        lambda state_dir, window: set(usage),
    )

    zones = make_zones(["memo_search", "memo_graph"])
    ToolSchemas().apply(zones, Context(state_dir=tmp_path, session_key="s1", project="memo"))
    assert "memo_graph" not in {t["name"] for t in zones.tools}

    toolschemas._session_keep_cache.clear()
    usage.add("memo_graph")

    zones = make_zones(["memo_search", "memo_graph"])
    ToolSchemas().apply(zones, Context(state_dir=tmp_path, session_key="s2", project="memo"))
    assert "memo_graph" in {t["name"] for t in zones.tools}


def test_persisting_the_keep_set_never_raises(tmp_path, monkeypatch):
    """Fail-open contract: an unwritable state dir degrades to the old
    in-process-only freeze, it does not fail the user's request."""
    from memo.proxy.transforms import toolschemas

    monkeypatch.setattr(
        "memo.proxy.transforms.toolschemas.recent_tool_names",
        lambda state_dir, window: {"memo_search"},
    )
    monkeypatch.setattr(
        "memo.proxy.transforms.toolschemas.keep_sets_path",
        lambda state_dir: tmp_path / "no" / "such" / "dir" / "x.json",
    )
    monkeypatch.setattr(
        toolschemas.Path, "mkdir", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only"))
    )
    zones = make_zones(["memo_search", "memo_graph"])
    assert ToolSchemas().apply(zones, _ctx(tmp_path)) > 0


def test_the_keep_set_never_grows_mid_session(tmp_path, monkeypatch):
    """Adding a tool mid-session is NOT prefix-safe, whatever it feels like.

    A "soft refresh" was added that, every 10 turns, folded newly-used tools
    into the frozen keep-set, commented "Never removes -- only additions
    preserve prefix stability". That is false. `tools` is serialized at the
    very FRONT of the cached prefix, so appending one element changes those
    bytes and the whole conversation re-caches at the 1.25x creation premium
    -- the exact failure the keep-set freeze exists to prevent.

    It never fired only because `Context.turn_count` was declared and never
    assigned, so the `turn_count > 0` guard was always false. Dead code with a
    live-looking call site is worse than either: one line elsewhere turns it
    into a full re-cache every ten turns.
    """
    import memo.proxy.transforms.toolschemas as ts

    assert not hasattr(ts, "_soft_refresh_keep_set")
    assert not hasattr(ts, "_SOFT_REFRESH_INTERVAL")

    used = {"memo_search"}
    monkeypatch.setattr(
        "memo.proxy.transforms.toolschemas.recent_tool_names",
        lambda state_dir, window: set(used),
    )
    ctx = _ctx(tmp_path)
    zones = make_zones(["memo_search", "memo_graph"])
    ToolSchemas().apply(zones, ctx)
    first = {t["name"] for t in zones.tools}

    # A later turn starts using memo_graph. The frozen set must not absorb it.
    used.add("memo_graph")
    zones2 = make_zones(["memo_search", "memo_graph"])
    ToolSchemas().apply(zones2, ctx)
    assert {t["name"] for t in zones2.tools} == first


def test_the_builtin_set_is_a_constant_not_a_subprocess() -> None:
    """The never-prune set must be stable across turns, so it is a constant.

    Whatever it holds lands in the `tools` array at the front of the cached
    prefix, so a set that answered on one turn and failed on the next would
    reshape the prefix and re-cache the whole conversation. This used to be
    computed by shelling out to `claude --list-tools` — an option that does
    not exist, so the call could only ever fail and fall back. The previous
    test for it mocked `subprocess.run` into succeeding, and so reported a
    discovery path that never ran in production.
    """
    import memo.proxy.transforms.toolschemas as ts

    assert isinstance(ts._BUILTIN_NEVER_PRUNE, frozenset)
    assert {"Read", "Write", "Edit", "Bash", "Glob", "Grep"} <= ts._BUILTIN_NEVER_PRUNE
    assert ts._BUILTIN_NEVER_PRUNE == ts._BUILTIN_NEVER_PRUNE, "must not be recomputed"


def test_structuredoutput_is_never_pruned(tmp_path, monkeypatch):
    """Pruning StructuredOutput makes the model call it blind, and it must.

    The harness executes StructuredOutput client-side and *forces* schema
    agents to call it, so stripping its schema from the request does not
    remove the capability — it removes the model's knowledge of the required
    fields. Measured across 237 subagent transcripts: 121 first calls (51%)
    came back `must have required property ...`, discarding 511k output
    tokens to retries. A tool the model is compelled to call is the one tool
    whose schema is never safe to drop.
    """
    monkeypatch.delenv("MEMO_PROXY_TOOL_SCHEMAS_SCOPE", raising=False)
    monkeypatch.setattr(
        "memo.proxy.transforms.toolschemas.recent_tool_names",
        lambda state_dir, window: set(),
    )
    zones = make_zones(["StructuredOutput", "memo_rename"])

    ToolSchemas().apply(zones, _ctx(tmp_path))

    names = {t["name"] for t in zones.tools}
    assert "StructuredOutput" in names
    assert "memo_rename" not in names, "unrelated pruning must still happen"


def test_builtin_discovery_does_not_shell_out() -> None:
    """The built-in set is a constant, not the result of a subprocess.

    `_discover_builtins` ran `claude --list-tools` at import time. That option
    does not exist — it exits non-zero with `unknown option '--list-tools'` —
    so the call could only ever fail, then fall back. It paid a subprocess on
    every import to learn nothing.
    """
    import memo.proxy.transforms.toolschemas as ts

    assert not hasattr(ts, "_discover_builtins")
    assert "StructuredOutput" in ts._BUILTIN_NEVER_PRUNE


# ── Wire names are MCP-prefixed; bare names never reach the proxy ────────────
# Every test above names its tools `memo_search` / `memo_rename`. A real
# payload never does: Claude Code puts memo's tools on the wire as
# `mcp__memo__memo_search`. So the fixtures could not produce the production
# condition, and the two guards that key off a bare name — `_ALWAYS_KEEP` and
# the `scope="memo"` ownership test — passed here while matching nothing live.


def test_always_keep_survives_under_the_real_mcp_wire_name(tmp_path, monkeypatch):
    """`memo_save` is in `_ALWAYS_KEEP` ("kept regardless of usage or scope"),
    but the wire spells it `mcp__memo__memo_save`. With no recorded usage the
    exact-match lookup misses and the write path is pruned off the model's
    surface — the same defect class as pruning a forced-call schema."""
    monkeypatch.delenv("MEMO_PROXY_TOOL_SCHEMAS_SCOPE", raising=False)
    monkeypatch.setattr(
        "memo.proxy.transforms.toolschemas.recent_tool_names",
        lambda state_dir, window: set(),
    )
    zones = make_zones(["mcp__memo__memo_save", "mcp__memo__memo_search", "mcp__memo__memo_rename"])
    ToolSchemas().apply(zones, _ctx(tmp_path))
    names = {t["name"] for t in zones.tools}
    assert "mcp__memo__memo_save" in names
    assert "mcp__memo__memo_search" in names
    # Not in _ALWAYS_KEEP and unused: pruning it is the whole point.
    assert "mcp__memo__memo_rename" not in names


def test_scope_memo_prunes_an_unused_memo_tool_under_its_wire_name(tmp_path, monkeypatch):
    """The conservative rollback scope must still prune memo's OWN unused
    tools. Keyed on `startswith("memo_")`, no wire name is ever recognized as
    owned, so every tool takes the passthrough branch and the transform saves
    nothing at all — a rollback that silently disables itself."""
    monkeypatch.setenv("MEMO_PROXY_TOOL_SCHEMAS_SCOPE", "memo")
    monkeypatch.setattr(
        "memo.proxy.transforms.toolschemas.recent_tool_names",
        lambda state_dir, window: set(),
    )
    zones = make_zones(["Read", "mcp__octocode__localSearchCode", "mcp__memo__memo_rename"])
    saved = ToolSchemas().apply(zones, _ctx(tmp_path))
    names = {t["name"] for t in zones.tools}
    assert {"Read", "mcp__octocode__localSearchCode"} <= names
    assert "mcp__memo__memo_rename" not in names
    assert saved > 0
