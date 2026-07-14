"""dream_profile — pure core: paths, source selection, deterministic render (B1)."""

from __future__ import annotations

import json as _json

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
        {"recall_id": "cccccccc", "session_id": s, "used_score": 1.0} for s in ("s1", "s2", "s3")
    ] + [{"recall_id": "aaaaaaaa", "session_id": s, "used_score": 1.0} for s in ("s1", "s2")]
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


# --- orchestrator (B1: rewrite-in-place; B3: rules block) ----------------------


class _Rec:
    def __init__(
        self, id_, type_="preference", title="Prefers pytest", body="body", updated="2026-06-01"
    ):
        self.id, self.type, self.title, self.body, self.updated = (
            id_,
            type_,
            title,
            body,
            updated,
        )


class _PairStore:
    def __init__(self, pairs=None):
        self._pairs = pairs or []

    def list_all(self, status=None, limit=200):
        return self._pairs


class _FakePair:
    def __init__(self, status, a, b):
        self.status, self.memory_id_a, self.memory_id_b = status, a, b


class _Mem:
    """Duck-typed Memory facade: store.list_recent + prefix-resolving get."""

    def __init__(self, rows, recs, pairs=None):
        self._rows, self._recs = rows, recs
        self.contradict_store = _PairStore(pairs)
        self.store = self

    def list_recent(self, limit=500, exclude_types=None):
        return self._rows

    def get(self, id_):
        for full, rec in self._recs.items():
            if full.startswith(id_):
                return rec
        return None


def _mk_cfg(tmp_path):
    cfg = _Cfg(tmp_path)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _write_grounding(state_dir, rid8, sessions, used=0.9):
    lines = [
        _json.dumps(
            {"session_id": s, "turn": 1, "recall_id": rid8, "used_score": used, "method": "cited"}
        )
        for s in sessions
    ]
    (state_dir / "grounding.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_run_profile_pass_writes_global_profile_with_provenance(tmp_path, monkeypatch):
    rid = "a" * 32
    mem = _Mem([{"id": rid, "type": "preference", "tags": []}], {rid: _Rec(rid)})
    cfg = _mk_cfg(tmp_path)
    monkeypatch.setattr(dp, "_llm_distill", lambda *a, **k: "- prefers pytest")
    res = dp.run_profile_pass(cfg, mem, dry_run=False)
    assert res["status"] == "done"
    doc = dp.profile_path(cfg).read_text(encoding="utf-8")
    assert "- prefers pytest" in doc
    assert f'"{rid[:8]}"' in doc  # memory-id provenance


def test_run_profile_pass_feeds_prior_text_for_in_place_rewrite(tmp_path, monkeypatch):
    rid = "a" * 32
    mem = _Mem([{"id": rid, "type": "decision", "tags": []}], {rid: _Rec(rid)})
    cfg = _mk_cfg(tmp_path)
    priors: list[str] = []

    def _fake(mem_, docs, *, prior, scope, budget):
        priors.append(prior)
        return "- v2 narrative"

    monkeypatch.setattr(dp, "_llm_distill", _fake)
    dp.run_profile_pass(cfg, mem)
    dp.run_profile_pass(cfg, mem)
    assert priors[0] == ""  # first night: no prior
    assert "- v2 narrative" in priors[1]  # second night rewrites IN PLACE


def test_run_profile_pass_writes_per_project_profiles(tmp_path, monkeypatch):
    rg, rp = "b" * 32, "c" * 32
    rows = [
        {"id": rg, "type": "preference", "tags": []},
        {"id": rp, "type": "decision", "tags": ["project:memo"]},
    ]
    mem = _Mem(rows, {rg: _Rec(rg), rp: _Rec(rp, type_="decision")})
    cfg = _mk_cfg(tmp_path)
    monkeypatch.setattr(dp, "_llm_distill", lambda *a, **k: "- distilled")
    res = dp.run_profile_pass(cfg, mem)
    scopes = {w["scope"] for w in res["written"]}
    assert scopes == {"global", "memo"}
    assert dp.profile_path(cfg, "memo").is_file()


def test_run_profile_pass_graduates_and_retires_standing_rules(tmp_path, monkeypatch):
    winner, loser = "d" * 32, "e" * 32
    recs = {
        winner: _Rec(winner, title="always pin transformers<5.13"),
        loser: _Rec(loser, title="old superseded rule", updated="2025-01-01"),
    }
    # both cited in 3 distinct sessions; loser is the older side of a resolved pair
    mem = _Mem(
        [{"id": winner, "type": "preference", "tags": []}],
        recs,
        pairs=[_FakePair("kept_newer", loser, winner)],
    )
    cfg = _mk_cfg(tmp_path)
    _write_grounding(cfg.state_dir, winner[:8], ["s1", "s2", "s3"])
    with (cfg.state_dir / "grounding.log").open("a", encoding="utf-8") as fh:
        for s in ("s1", "s2", "s3"):
            fh.write(
                _json.dumps(
                    {
                        "session_id": s,
                        "turn": 2,
                        "recall_id": loser[:8],
                        "used_score": 0.9,
                        "method": "cited",
                    }
                )
                + "\n"
            )
    monkeypatch.setattr(dp, "_llm_distill", lambda *a, **k: "- narrative")
    res = dp.run_profile_pass(cfg, mem, directive_k=3)
    doc = dp.profile_path(cfg).read_text(encoding="utf-8")
    assert "## Standing rules" in doc
    assert "always pin transformers<5.13" in doc
    assert "old superseded rule" not in doc  # retired via contradict pair
    assert res["standing_rules"] == 1


def test_run_profile_pass_dry_run_writes_nothing(tmp_path, monkeypatch):
    rid = "a" * 32
    mem = _Mem([{"id": rid, "type": "preference", "tags": []}], {rid: _Rec(rid)})
    cfg = _mk_cfg(tmp_path)
    monkeypatch.setattr(dp, "_llm_distill", lambda *a, **k: "- x")
    res = dp.run_profile_pass(cfg, mem, dry_run=True)
    assert res["status"] == "done"
    assert res["written"][0]["status"] == "would_write"
    assert not dp.profile_path(cfg).exists()


def test_run_profile_pass_never_raises(tmp_path):
    class _Boom:
        contradict_store = _PairStore()

        @property
        def store(self):
            raise RuntimeError("store exploded")

    res = dp.run_profile_pass(_mk_cfg(tmp_path), _Boom())
    assert res["status"] == "error"
    assert "store exploded" in res["error"]
