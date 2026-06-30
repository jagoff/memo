"""Tests for graph-community synthesis (spec 3)."""

from __future__ import annotations

from memo.dream_communities import community_clusters, decide_syntheses, provenance_hash
from memo.navigation import Community


def test_provenance_hash_is_order_independent():
    assert provenance_hash(["a", "b"]) == provenance_hash(["b", "a"])
    assert provenance_hash(["a"]) != provenance_hash(["b"])


def test_community_clusters_maps_entities_to_memories():
    class _Nav:
        def detect_communities(self, *, min_size):
            return [Community(id=1, entities=["x", "y"], size=2, representative_entity="x")]

    class _Graph:
        def entity_memories(self, name, type_=None):
            return {"x": ["m1", "m2"], "y": ["m2", "m3"]}.get(name, [])

    class _Mem:
        navigator = _Nav()
        graph = _Graph()

    cl = community_clusters(_Mem(), min_size=2, max_communities=5)
    assert len(cl) == 1
    assert cl[0]["entities"] == ["x", "y"]
    assert set(cl[0]["memory_ids"]) == {"m1", "m2", "m3"}
    assert cl[0]["representative"] == "x"


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

    fail = decide_syntheses(
        clusters, synthesize_fn=lambda c: None, exists_fn=lambda h: False
    )
    assert fail[0]["status"] == "synth_failed"
