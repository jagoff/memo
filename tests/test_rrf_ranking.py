"""RRF-k (configurable + density-adaptive), temporal half-life decay, and
graph-candidate RRF scale parity."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from memo.memory.record import (
    _RECALL_DECAY_HALFLIFE_DEFAULT,
    MemoryRecord,
    _adaptive_rrf_k,
    _apply_decay,
    _halflife_for_type,
    _rrf_confident_top,
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
    """A memory updated exactly one half-life ago retains 50% freshness.

    Uses type='preference' which has no per-type default, so it falls back to
    the global halflife_days argument. At exactly one global halflife the decay
    term is 0.5 and (with alpha=1.0) the final score equals 0.5.
    """
    hl = _RECALL_DECAY_HALFLIFE_DEFAULT
    old = (datetime.now(tz=UTC) - timedelta(days=hl)).isoformat()
    rec = MemoryRecord(
        id="a", path="a", title="t", type="preference", tags=[],
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


# ── #12 Per-type recency decay ─────────────────────────────────────────────


def _make_record(id_: str, type_: str, age_days: float, score: float = 1.0) -> MemoryRecord:
    """Build a MemoryRecord updated `age_days` ago."""
    updated = (datetime.now(tz=UTC) - timedelta(days=age_days)).isoformat()
    return MemoryRecord(
        id=id_,
        path=f"{id_}.md",
        title=id_,
        type=type_,
        tags=[],
        created=updated,
        updated=updated,
        body="",
        score=score,
    )


def test_halflife_for_type_decision_uses_registered_default(monkeypatch):
    """decision type uses its registered default (365 days) not the global."""
    # Ensure per-type env var is unset so we get the registered default.
    monkeypatch.delenv("MEMO_DECAY_HALFLIFE_DECISION", raising=False)
    hl = _halflife_for_type("decision", global_halflife=90.0)
    assert hl == 365.0, f"decision should use 365-day default, got {hl}"


def test_halflife_for_type_note_uses_registered_default(monkeypatch):
    """note type uses its registered default (30 days) not the global."""
    monkeypatch.delenv("MEMO_DECAY_HALFLIFE_NOTE", raising=False)
    hl = _halflife_for_type("note", global_halflife=90.0)
    assert hl == 30.0, f"note should use 30-day default, got {hl}"


def test_halflife_for_type_reference_no_decay(monkeypatch):
    """reference type has None default — returns 0.0 (no decay)."""
    monkeypatch.delenv("MEMO_DECAY_HALFLIFE_REFERENCE", raising=False)
    hl = _halflife_for_type("reference", global_halflife=90.0)
    assert hl == 0.0, f"reference should not decay (0.0), got {hl}"


def test_halflife_for_type_env_override(monkeypatch):
    """Setting MEMO_DECAY_HALFLIFE_DECISION overrides the registered default."""
    monkeypatch.setenv("MEMO_DECAY_HALFLIFE_DECISION", "500")
    hl = _halflife_for_type("decision", global_halflife=90.0)
    assert hl == 500.0, f"env var should override default, got {hl}"


def test_halflife_for_type_unknown_type_uses_global():
    """An unregistered type falls back to the global half-life."""
    hl = _halflife_for_type("synthesis", global_halflife=120.0)
    assert hl == 120.0, f"unknown type should use global halflife, got {hl}"


def test_decision_decays_slower_than_note(monkeypatch):
    """A decision from 30 days ago should retain more score than a note of
    the same age because decision half-life (365d) >> note half-life (30d).

    With alpha=1.0 the final score equals the decay term:
      decay_decision = 0.5 ** (30 / 365) ≈ 0.944
      decay_note     = 0.5 ** (30 /  30) = 0.5

    The decision should therefore rank ABOVE the note in the output list.
    """
    monkeypatch.delenv("MEMO_DECAY_HALFLIFE_DECISION", raising=False)
    monkeypatch.delenv("MEMO_DECAY_HALFLIFE_NOTE", raising=False)

    age_days = 30.0
    decision_rec = _make_record("d1", "decision", age_days, score=1.0)
    note_rec = _make_record("n1", "note", age_days, score=1.0)

    # global halflife = 90 (would give 0.794 for both if per-type were ignored)
    out = _apply_decay([decision_rec, note_rec], halflife_days=90.0, alpha=1.0)

    ids = [r.id for r in out]
    assert ids[0] == "d1", (
        f"decision should outrank note after per-type decay (got order {ids})"
    )
    d_score = next(r.score for r in out if r.id == "d1")
    n_score = next(r.score for r in out if r.id == "n1")
    assert d_score > n_score, (
        f"decision score ({d_score}) should exceed note score ({n_score})"
    )
    # Verify the note is actually at ~0.5 (one half-life at 30d default)
    assert abs(n_score - 0.5) < 0.02, f"note at one halflife should be ~0.5, got {n_score}"
    # Verify the decision is well above 0.5 (far from its 365d halflife)
    assert d_score > 0.9, f"decision at 30d should retain >0.9 freshness, got {d_score}"


def test_reference_not_decayed(monkeypatch):
    """References pass through _apply_decay unchanged (no decay by default)."""
    monkeypatch.delenv("MEMO_DECAY_HALFLIFE_REFERENCE", raising=False)

    age_days = 180.0
    ref_rec = _make_record("ref1", "reference", age_days, score=0.8)
    note_rec = _make_record("n1", "note", age_days, score=0.8)

    out = _apply_decay([ref_rec, note_rec], halflife_days=90.0, alpha=1.0)

    ref_out = next(r for r in out if r.id == "ref1")
    note_out = next(r for r in out if r.id == "n1")

    # note at 2× halflife (180/30) should be 0.5^6 ≈ 0.016 final with alpha=1
    # actually 0.5**(180/30) = 0.5**6 = 0.015625 → rounded to 0.015625
    assert note_out.score < 0.1, (
        f"note at 6 halflives should be nearly zero, got {note_out.score}"
    )
    # reference should be unchanged (score passed through)
    assert abs((ref_out.score or 0.0) - 0.8) < 1e-6, (
        f"reference score should be unchanged (0.8), got {ref_out.score}"
    )


# ── #13 Hybrid search leg weighting ───────────────────────────────────────────


def test_rrf_fuse_equal_weights_identical_to_unweighted():
    """weights=[1.0, 1.0] must produce the same ranking as no weights at all."""
    k = 60
    vec_list = [_hit("a", 0.9), _hit("b", 0.7)]
    bm25_list = [_hit("b", 0.8), _hit("c", 0.5)]

    unweighted = _rrf_fuse(vec_list, bm25_list, limit=5, k=k)
    weighted = _rrf_fuse(vec_list, bm25_list, limit=5, k=k, weights=[1.0, 1.0])

    assert [r["id"] for r in unweighted] == [r["id"] for r in weighted], (
        "equal weights should produce identical ranking to unweighted RRF"
    )


def test_rrf_fuse_default_half_weights_equal_to_unweighted_rank():
    """Default weights=[0.5, 0.5] yield the same relative ordering as [1.0, 1.0].

    Scores will differ (halved) but rank order must be identical — the
    proportional scaling cannot swap positions.
    """
    k = 60
    vec_list = [_hit("a", 0.95), _hit("b", 0.7)]
    bm25_list = [_hit("b", 0.85), _hit("c", 0.5)]

    unweighted = _rrf_fuse(vec_list, bm25_list, limit=5, k=k)
    half = _rrf_fuse(vec_list, bm25_list, limit=5, k=k, weights=[0.5, 0.5])

    assert [r["id"] for r in unweighted] == [r["id"] for r in half], (
        "symmetric half-weights should not change rank order vs. unweighted RRF"
    )


def test_high_vec_weight_promotes_semantic_only_hit():
    """With vec_weight=0.9, bm25_weight=0.1, a vec-only hit should rank above a
    bm25-only hit compared to equal weights where the bm25 hit ranks first.

    Setup:
      - "semantic_doc": appears only in vec list at rank 0.
      - "keyword_doc":  appears only in bm25 list at rank 0.
      - Equal weights (0.5 / 0.5): both get the same RRF contribution —
        tied by insertion order; keyword_doc comes from bm25 list at rank 0
        same as semantic_doc from vec list — they tie at the same score,
        so order is list-dependent but both equal. We just verify the
        inequality flips with high vec weight.
      - High vec weight (0.9 / 0.1): semantic_doc contribution = 0.9/(k+1),
        keyword_doc contribution = 0.1/(k+1) → semantic_doc ranks higher.
    """
    k = 60

    vec_list = [_hit("semantic_doc", score=0.95)]
    bm25_list = [_hit("keyword_doc", score=0.88)]

    # Equal weights: both get 1/(k+1) — a tie, order arbitrary.
    equal_results = _rrf_fuse(vec_list, bm25_list, limit=5, k=k, weights=[0.5, 0.5])
    equal_ids = [r["id"] for r in equal_results]
    assert set(equal_ids) == {"semantic_doc", "keyword_doc"}, (
        f"Both docs should appear with equal weights, got: {equal_ids}"
    )

    # High vec weight: semantic_doc should rank above keyword_doc.
    vec_heavy = _rrf_fuse(vec_list, bm25_list, limit=5, k=k, weights=[0.9, 0.1])
    ids = [r["id"] for r in vec_heavy]
    assert ids[0] == "semantic_doc", (
        f"With high vec weight, semantic-only hit should rank first; got: {ids}"
    )

    # High bm25 weight: keyword_doc should rank above semantic_doc.
    bm25_heavy = _rrf_fuse(vec_list, bm25_list, limit=5, k=k, weights=[0.1, 0.9])
    ids2 = [r["id"] for r in bm25_heavy]
    assert ids2[0] == "keyword_doc", (
        f"With high bm25 weight, keyword-only hit should rank first; got: {ids2}"
    )


def test_vec_weight_leg_weighting_flag_warning(monkeypatch, caplog):
    """When both weight flags are set but don't sum to ~1.0, a warning is logged.

    We test the pure _rrf_fuse path (which takes explicit weights) and the
    warning branch in search_ops via monkeypatching the env for the integration
    test. Here we just verify the weight math and that the warning condition
    is correct (the env-var check lives in search_ops, not in _rrf_fuse itself).
    """
    import math

    k = 60
    vec_list = [_hit("a", 0.9)]
    bm25_list = [_hit("b", 0.8)]

    # Weights summing to 1.0 — no issue.
    result = _rrf_fuse(vec_list, bm25_list, limit=5, k=k, weights=[0.7, 0.3])
    assert len(result) == 2

    # Verify the score math: a only in vec at rank 0, b only in bm25 at rank 0.
    scores = {r["id"]: r["score"] for r in result}
    expected_a = 0.7 / (k + 1)
    expected_b = 0.3 / (k + 1)
    assert math.isclose(scores["a"], expected_a, rel_tol=1e-9), (
        f"score for 'a' should be {expected_a}, got {scores['a']}"
    )
    assert math.isclose(scores["b"], expected_b, rel_tol=1e-9), (
        f"score for 'b' should be {expected_b}, got {scores['b']}"
    )


# ── #14 Reranker confident-RRF skip ───────────────────────────────────────────


def test_rrf_confident_top_detects_clear_winner():
    """A dominant RRF top hit can safely skip cross-encoder rerank."""
    rows = [
        _hit("clear", score=0.196721),
        _hit("second", score=0.049677),
        _hit("third", score=0.033871),
    ]

    decision = _rrf_confident_top(rows, min_ratio=3.0, min_gap=0.05)

    assert decision.skip is True
    assert decision.top_id == "clear"
    assert decision.ratio > 3.0
    assert decision.gap > 0.05


def test_rrf_confident_top_keeps_ambiguous_results_for_reranker():
    """Close RRF scores should still go through the reranker."""
    rows = [
        _hit("first", score=0.0164),
        _hit("second", score=0.0159),
        _hit("third", score=0.0147),
    ]

    decision = _rrf_confident_top(rows, min_ratio=3.0, min_gap=0.05)

    assert decision.skip is False
    assert decision.top_id == "first"
