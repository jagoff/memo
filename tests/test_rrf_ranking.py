"""RRF-k (configurable + density-adaptive) and temporal half-life decay."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from memo.memory.record import (
    _RECALL_DECAY_HALFLIFE_DEFAULT,
    MemoryRecord,
    _adaptive_rrf_k,
    _apply_decay,
)


def _hit(rid: str) -> dict:
    return {"id": rid, "title": rid, "type": "note", "tags": [], "created": "", "updated": ""}


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
