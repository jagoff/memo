"""dream_profile — pure core: paths, source selection, deterministic render (B1)."""

from __future__ import annotations

from memo import dream_profile as dp


class _Cfg:
    def __init__(self, tmp_path):
        self.memory_dir = tmp_path / "memories"
        self.state_dir = tmp_path / "state"


def _row(type_, tags=None, id_="a" * 32):
    return {"id": id_, "type": type_, "tags": tags or [], "title": "t"}


def test_profile_path_global_and_project(tmp_path):
    cfg = _Cfg(tmp_path)
    assert dp.profile_path(cfg) == tmp_path / "memories" / "_profile" / "profile.md"
    assert (
        dp.profile_path(cfg, "Memo Repo")
        == tmp_path / "memories" / "_profile" / "project-memo-repo.md"
    )


def test_project_file_never_collides_with_global(tmp_path):
    # a project literally named "profile" must not overwrite the global doc
    assert dp.profile_path(_Cfg(tmp_path), "profile").name == "project-profile.md"


def test_select_sources_filters_types_and_scope():
    rows = [
        _row("preference", id_="1" * 32),
        _row("decision", tags=["project:memo"], id_="2" * 32),
        _row("note", id_="3" * 32),  # wrong type
        _row("reference", id_="4" * 32),  # wrong type
        _row("synthesis", tags=["project:memo"], id_="5" * 32),
    ]
    assert [r["id"] for r in dp.select_sources(rows, project=None)] == ["1" * 32]
    assert [r["id"] for r in dp.select_sources(rows, project="memo")] == [
        "2" * 32,
        "5" * 32,
    ]


def test_select_sources_caps_at_limit():
    rows = [_row("preference", id_=f"{i:032x}") for i in range(10)]
    assert len(dp.select_sources(rows, limit=3)) == 3


def test_project_buckets_distinct_ordered():
    rows = [
        _row("decision", tags=["project:memo"]),
        _row("note", tags=["project:ignored"]),  # wrong type — excluded
        _row("preference", tags=["project:synapse"]),
        _row("decision", tags=["project:memo"]),  # duplicate bucket
        _row("feedback"),  # global — not a project bucket
    ]
    assert dp.project_buckets(rows) == ["memo", "synapse"]


def test_render_profile_has_frontmatter_but_no_id_key():
    doc = dp.render_profile(
        scope="global",
        narrative="- prefers Spanish replies",
        rules=[],
        source_ids=["a1b2c3d4e5f6a7b8"],
        updated="2026-07-03T03:00:00+00:00",
        char_budget=4000,
    )
    assert doc.startswith("---\n")
    assert "\nid:" not in doc  # reindex must skip this file (maintain_ops.py:252-255)
    assert '"a1b2c3d4"' in doc  # memory-id provenance, 8-char short ids
    assert "- prefers Spanish replies" in doc


def test_render_profile_budget_trims_narrative_keeps_rules():
    doc = dp.render_profile(
        scope="global",
        narrative="x" * 500,
        rules=[("f" * 32, "always run pytest before commit")],
        source_ids=[],
        updated="2026-07-03T03:00:00+00:00",
        char_budget=120,
    )
    assert "## Standing rules" in doc
    assert "always run pytest before commit" in doc  # rules survive the cut
    assert "x" * 500 not in doc  # narrative trimmed to fit


# --- directive graduation (B3) -------------------------------------------------


def test_standing_rule_ids_requires_k_distinct_sessions():
    rows = [
        {"recall_id": "aaaaaaaa", "session_id": "s1", "used_score": 0.9},
        {"recall_id": "aaaaaaaa", "session_id": "s2", "used_score": 0.8},
        {"recall_id": "aaaaaaaa", "session_id": "s2", "used_score": 0.8},  # same session
        {"recall_id": "bbbbbbbb", "session_id": "s1", "used_score": 0.9},  # 1 session
    ]
    assert dp.standing_rule_ids(rows, k=2, min_used=0.5) == ["aaaaaaaa"]


def test_standing_rule_ids_ignores_low_used_score():
    rows = [
        {"recall_id": "aaaaaaaa", "session_id": "s1", "used_score": 0.2},
        {"recall_id": "aaaaaaaa", "session_id": "s2", "used_score": 0.9},
    ]
    assert dp.standing_rule_ids(rows, k=2, min_used=0.5) == []


def test_standing_rule_ids_orders_by_session_count_then_prefix():
    rows = [
        {"recall_id": "cccccccc", "session_id": s, "used_score": 1.0}
        for s in ("s1", "s2", "s3")
    ] + [
        {"recall_id": "aaaaaaaa", "session_id": s, "used_score": 1.0}
        for s in ("s1", "s2")
    ]
    assert dp.standing_rule_ids(rows, k=2) == ["cccccccc", "aaaaaaaa"]


def test_standing_rule_ids_tolerates_malformed_rows():
    rows = [
        {"recall_id": "", "session_id": "s1", "used_score": 1.0},
        {"recall_id": "dddddddd", "session_id": "", "used_score": 1.0},
        {"recall_id": "dddddddd", "session_id": "s1", "used_score": "not-a-number"},
    ]
    assert dp.standing_rule_ids(rows, k=1) == []


def test_losing_ids_retires_older_side_of_resolved_pairs():
    pairs = [
        {"status": "kept_newer", "memory_id_a": "old", "memory_id_b": "new"},
        {"status": "open", "memory_id_a": "x", "memory_id_b": "y"},  # unresolved
        {"status": "dismissed", "memory_id_a": "p", "memory_id_b": "q"},  # false pos
    ]
    updated = {"old": "2026-01-01", "new": "2026-06-01", "x": "1", "y": "2", "p": "1", "q": "2"}
    assert dp.losing_ids(pairs, updated.get) == {"old"}


def test_losing_ids_kept_older_retires_newer_side():
    # kept_older = older side won (explicit user choice) → the NEWER side loses
    pairs = [{"status": "kept_older", "memory_id_a": "old", "memory_id_b": "new"}]
    updated = {"old": "2026-01-01", "new": "2026-06-01"}
    assert dp.losing_ids(pairs, updated.get) == {"new"}


def test_losing_ids_fused_retires_both_sides():
    # fused = both merged into a NEW memory → both original sides retire,
    # even when one of them is still live (not yet archived)
    pairs = [{"status": "fused", "memory_id_a": "left", "memory_id_b": "right"}]
    updated = {"left": "2026-01-01", "right": "2026-06-01"}
    assert dp.losing_ids(pairs, updated.get) == {"left", "right"}


def test_losing_ids_retires_missing_record_outright():
    # archived (superseded) records resolve to None — retired without comparison
    pairs = [{"status": "evolved", "memory_id_a": "gone", "memory_id_b": "live"}]
    assert dp.losing_ids(pairs, {"live": "2026-06-01"}.get) == {"gone"}
