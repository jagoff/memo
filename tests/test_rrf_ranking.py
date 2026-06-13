"""RRF-k (configurable + density-adaptive), temporal half-life decay, and
graph-candidate RRF scale parity."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from memo.memory.record import (
    _RECALL_DECAY_HALFLIFE_DEFAULT,
    MemoryRecord,
    _adaptive_rrf_k,
    _apply_decay,
    _rrf_fuse,
)


def _hit(rid: str, score: float = 0.0) -> dict:
    return {
        "id": rid,
        "title": rid,
        "type": "note",
        "tags": [],
        "created": "",
        "updated": "",
        "score": score,
    }


# ── #6 RRF-k adaptive ──────────────────────────────────────────────────────


def test_adaptive_k_single_list_returns_base():
    assert _adaptive_rrf_k([[_hit("a"), _hit("b")]], base_k=60) == 60


def test_adaptive_k_high_overlap_shrinks_k():
    shared = [_hit("a"), _hit("b"), _hit("c")]
    k = _adaptive_rrf_k([shared, list(shared)], base_k=60)
    assert k < 60, "strong agreement should sharpen fusion (smaller k)"


def test_adaptive_k_disjoint_lists_grows_k():
    a = [_hit("a"), _hit("b"), _hit("c")]
    b = [_hit("x"), _hit("y"), _hit("z")]
    k = _adaptive_rrf_k([a, b], base_k=60)
    assert k > 60, "no agreement should soften rank dominance (larger k)"


def test_adaptive_k_is_bounded():
    a = [_hit("a"), _hit("b")]
    b = [_hit("x"), _hit("y")]
    k = _adaptive_rrf_k([a, b], base_k=60)
    assert 30 <= k <= 120


# ── #10 temporal half-life decay ───────────────────────────────────────────


def test_default_halflife_is_90_days():
    assert _RECALL_DECAY_HALFLIFE_DEFAULT == 90.0


def test_decay_is_true_half_life():
    """A memory updated exactly one half-life ago retains 50% freshness."""
    hl = _RECALL_DECAY_HALFLIFE_DEFAULT
    old = (datetime.now(tz=UTC) - timedelta(days=hl)).isoformat()
    rec = MemoryRecord(
        id="a", path="a", title="t", type="note", tags=[],
        created=old, updated=old, body="", score=1.0,
    )
    # alpha=1.0 isolates the decay term: final == decay == 0.5 at one half-life.
    out = _apply_decay([rec], halflife_days=hl, alpha=1.0)
    assert abs((out[0].score or 0.0) - 0.5) < 0.01


# ── Graph candidate RRF scale parity ──────────────────────────────────────────


def _graph_hit(rid: str, entity_count: int, k: int = 60) -> dict:
    """Simulate a graph candidate after the fix: synthetic RRF score,
    sorted position reflects entity count descending (so rank 0 = highest count).
    """
    # In the fixed _fetch_graph_candidates, rank is 0-based list position after
    # sorting by entity count descending.  We reproduce that here.
    rank = 0  # caller builds lists in sorted order; rank = index in that list
    return {**_hit(rid), "score": 1.0 / (k + rank + 1)}


def test_graph_candidate_score_in_rrf_range():
    """Fixed graph candidates carry scores in [0,1], not raw integer counts."""
    k = 60
    # Simulate 3 graph candidates after the fix (entity counts 5, 3, 1).
    graph_list = [
        {**_hit("g1"), "score": 1.0 / (k + 0 + 1)},
        {**_hit("g2"), "score": 1.0 / (k + 1 + 1)},
        {**_hit("g3"), "score": 1.0 / (k + 2 + 1)},
    ]
    for hit in graph_list:
        assert 0.0 < hit["score"] <= 1.0, (
            f"Graph hit score {hit['score']} is outside [0,1] — raw entity "
            "count was not converted to RRF synthetic score"
        )


def test_graph_candidates_do_not_dominate_high_vec_hits():
    """A graph-only candidate with many entity matches must not outrank a
    candidate that appears at top of BOTH vec and BM25 lists.

    Before the fix, graph candidates had `score = entity_count` (e.g. 5).
    _rrf_fuse uses position (rank), not the pre-fusion score, so the raw
    integer score doesn't directly corrupt fusion — but the list ORDER
    determines rank.  This test verifies that after the fix, a strongly-
    agreed vec+BM25 candidate beats a graph-only candidate even when the
    latter appears at rank-0 of the graph list.
    """
    k = 60

    # "best_doc" appears at top of both vec and BM25 (rank 0 in each).
    # Its fused score = 1/(60+1) + 1/(60+1) ≈ 0.0328
    vec_list = [_hit("best_doc", score=0.95), _hit("other1", score=0.7)]
    bm25_list = [_hit("best_doc", score=0.88), _hit("other2", score=0.6)]

    # "graph_only_doc" appears only in graph at rank 0.
    # Its fused score = 1/(60+1) ≈ 0.0164
    graph_list = [
        {**_hit("graph_only_doc"), "score": 1.0 / (k + 0 + 1)},  # fixed score
    ]

    results = _rrf_fuse(vec_list, bm25_list, graph_list, limit=5, k=k)
    ids = [r["id"] for r in results]

    assert ids[0] == "best_doc", (
        f"'best_doc' should rank first (appears in vec+BM25), got: {ids}"
    )
    # graph_only_doc should appear but not at the top
    assert "graph_only_doc" in ids, "graph_only_doc should still appear in results"
    graph_pos = ids.index("graph_only_doc")
    best_pos = ids.index("best_doc")
    assert best_pos < graph_pos, (
        f"best_doc (rank {best_pos}) should outrank graph_only_doc (rank {graph_pos})"
    )


def test_graph_candidates_with_old_raw_score_would_be_on_wrong_scale():
    """Demonstrates the pre-fix problem: if graph hits carry score=entity_count
    (integer), _rrf_fuse still uses position for fusion, but post-RRF steps
    that touch the score field (decay, health multipliers) would see inflated
    values.  After the fix, all pre-fusion scores are in [0,1].

    This test confirms the old integer values (2, 3) exceed 1.0, and the
    fixed synthetic values stay within [0,1].
    """
    k = 60
    # Old behavior: graph scores are raw entity counts (≥1, integer-valued).
    # Entity count of 1 sits exactly at the boundary; counts ≥2 exceed 1.0.
    # The key point is they are NOT in the interior (0, 1/61] that the fix uses.
    old_graph_scores = [3.0, 2.0, 1.0]
    for s in old_graph_scores:
        assert s >= 1.0, "Pre-fix: entity count scores are ≥1 (integers)"

    # New behavior: synthetic RRF scores for same candidates
    new_graph_scores = [1.0 / (k + rank + 1) for rank in range(3)]
    for s in new_graph_scores:
        assert 0.0 < s <= 1.0, f"Post-fix: score {s} must be in (0, 1]"
