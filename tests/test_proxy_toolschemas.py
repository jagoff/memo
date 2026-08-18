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


def test_non_memo_tools_are_never_pruned(tmp_path, monkeypatch):
    """Pruning another server's schema would break tools memo does not own."""
    monkeypatch.setattr(
        "memo.proxy.transforms.toolschemas.recent_tool_names",
        lambda state_dir, window: set(),
    )
    zones = make_zones(["Read", "Bash", "memo_rename"])
    ToolSchemas().apply(zones, _ctx(tmp_path))
    names = {t["name"] for t in zones.tools}
    assert {"Read", "Bash"} <= names
    assert "memo_rename" not in names


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


def test_recent_tool_names_ignores_non_memo_names(tmp_path):
    path = tmp_path / "proxy" / "tool_usage.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema": "memo.proxy.tool_usage.v1",
                "sessions": {"s1": {"tools": ["memo_search", "Read", "Bash"], "ts": 1.0}},
            }
        ),
        encoding="utf-8",
    )
    assert recent_tool_names(tmp_path, 20) == {"memo_search"}


def test_recent_tool_names_survives_corrupt_json(tmp_path):
    path = tmp_path / "proxy" / "tool_usage.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert recent_tool_names(tmp_path, 20) == set()
