"""Tests for graph-community synthesis (spec 3)."""

from __future__ import annotations

from memo.dream_communities import community_clusters, decide_syntheses, provenance_hash


def test_provenance_hash_is_order_independent():
    assert provenance_hash(["a", "b"]) == provenance_hash(["b", "a"])
    assert provenance_hash(["a"]) != provenance_hash(["b"])


def test_community_clusters_maps_curated_packet_and_forces_entity_only_graph():
    captured: dict = {}

    class _Mem:
        def graph_discover(self, **kwargs):
            captured.update(kwargs)
            return {
                "available": True,
                "projection_version": "v1",
                "communities": [
                    {
                        "nodes": [{"label": "x"}, {"label": "y"}],
                        "representative": {"label": "x"},
                        "memory_ids": ["m1", "m2", "m3"],
                        "edge_evidence": [{"evidence_ids": ["memory://m1"]}],
                    }
                ],
            }

    cl = community_clusters(_Mem(), min_size=2, max_communities=5)
    assert captured["include_code"] is False
    assert len(cl) == 1
    assert cl[0]["entities"] == ["x", "y"]
    assert set(cl[0]["memory_ids"]) == {"m1", "m2", "m3"}
    assert cl[0]["representative"] == "x"
    assert cl[0]["projection_version"] == "v1"
    assert cl[0]["edge_evidence"]


def test_community_clusters_skips_hub_blob():
    class _Mem:
        def graph_discover(self, **kwargs):
            assert kwargs["max_region_size"] == 40
            return {
                "available": True,
                "projection_version": "v1",
                "communities": [
                    {
                        "nodes": [{"label": name} for name in ("a", "b", "c", "d")],
                        "representative": {"label": "a"},
                        "memory_ids": ["m1"],
                        "edge_evidence": [],
                    }
                ],
            }

    cl = community_clusters(_Mem(), min_size=4, max_communities=5, max_size=40)
    assert [c["representative"] for c in cl] == ["a"]  # 200-entity blob dropped


def test_community_clusters_does_not_fallback_when_projection_unavailable():
    class _Mem:
        def graph_discover(self, **kwargs):
            return {"available": False, "reason": "projection_stale", "communities": []}

    cl = community_clusters(_Mem(), min_size=4, max_communities=5, max_size=40)
    assert cl == []


def test_decide_syntheses_dedup_dryrun_save_and_fail():
    clusters = [{"entities": ["a", "b"], "representative": "a", "memory_ids": ["m1"]}]

    skip = decide_syntheses(
        clusters, synthesize_fn=lambda c: {"title": "T", "body": "B"}, exists_fn=lambda h: True
    )
    assert skip[0]["status"] == "skip_exists"

    dry = decide_syntheses(
        clusters,
        synthesize_fn=lambda c: {"title": "T", "body": "B"},
        exists_fn=lambda h: False,
        dry_run=True,
    )
    assert dry[0]["status"] == "would_save"
    assert dry[0]["entities"] == ["a", "b"]

    save = decide_syntheses(
        clusters, synthesize_fn=lambda c: {"title": "T", "body": "B"}, exists_fn=lambda h: False
    )
    assert save[0]["status"] == "save"
    assert save[0]["title"] == "T"
    assert save[0]["provenance"] == ["a", "b"]

    fail = decide_syntheses(clusters, synthesize_fn=lambda c: None, exists_fn=lambda h: False)
    assert fail[0]["status"] == "synth_failed"


def test_run_synthesize_communities_disabled_by_default(monkeypatch):
    from memo.dream_communities import run_synthesize_communities

    monkeypatch.delenv("MEMO_DREAM_COMMUNITIES_ENABLED", raising=False)
    res = run_synthesize_communities(None, None)  # mem untouched when the flag is off
    assert res["status"] == "disabled"


def test_community_key_stable_against_tail_growth():
    from memo.dream_communities import _community_key

    core = ["a", "b", "c", "d", "e", "f", "g", "h"]
    base = {"representative": "a", "entities": [*core, "zzz1"]}
    grew = {"representative": "a", "entities": [*core, "zzz1", "zzz2"]}  # tail beyond top-8 grew
    changed = {"representative": "x", "entities": [*core, "zzz1"]}
    assert provenance_hash(_community_key(base)) == provenance_hash(_community_key(grew))
    assert provenance_hash(_community_key(base)) != provenance_hash(_community_key(changed))


def test_synthesize_refreshes_curated_projection_first(monkeypatch, tmp_cfg):
    from memo import dream_communities

    calls = []

    class _Mem:
        def rebuild_graph_if_due(self):
            calls.append("projection")

        def graph_discover(self, **kwargs):
            return {"available": True, "communities": []}

    monkeypatch.setenv("MEMO_DREAM_COMMUNITIES_ENABLED", "1")
    monkeypatch.setenv("MEMO_GRAPH_DISCOVERY_ENABLED", "1")
    dream_communities.run_synthesize_communities(tmp_cfg, _Mem())
    assert calls == ["projection"]
