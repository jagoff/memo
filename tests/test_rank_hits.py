from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memo.recall_logic import RankKnobs, rank_hits


@dataclass
class _Hit:
    id: str
    score: float | None
    title: str = ""
    body: str = ""
    type: str = "note"
    extra: dict[str, Any] = field(default_factory=dict)


def _mk(id: str, score: float | None, **kw: Any) -> _Hit:
    """Build a hit with content unique per id, so dedup_hits only collapses
    genuine duplicates (same id), not distinct memories sharing a default body."""
    kw.setdefault("title", f"title {id}")
    kw.setdefault("body", f"distinct body for memory {id}, long enough to pass the gate")
    return _Hit(id=id, score=score, **kw)


def test_rank_hits_filters_below_min_sim_in_vec_mode() -> None:
    hits = [_mk("a", 0.9), _mk("b", 0.3), _mk("c", 0.7)]
    out = rank_hits(hits, RankKnobs(min_sim=0.5, mode="vec", min_body_chars=0))
    assert [h.id for h in out] == ["a", "c"]  # b dropped, order preserved


def test_rank_hits_dedups_by_id() -> None:
    # Two hits with the SAME id (same memory surfaced twice) collapse to one.
    hits = [_mk("a", 0.9), _mk("a", 0.8), _mk("b", 0.7)]
    out = rank_hits(hits, RankKnobs(min_sim=0.0, min_body_chars=0))
    assert [h.id for h in out] == ["a", "b"]


def test_rank_hits_hybrid_gate_uses_injected_vec_cosine() -> None:
    # In hybrid mode the RRF score is incomparable to min_sim; gate on cosine.
    hits = [_mk("a", 99.0), _mk("b", 99.0)]  # high RRF scores
    cos = {"a": 0.8, "b": 0.2}
    out = rank_hits(
        hits,
        RankKnobs(min_sim=0.5, mode="hybrid", min_body_chars=0),
        vec_cosine=lambda h: cos[h.id],
    )
    assert [h.id for h in out] == ["a"]  # b gated out by true cosine


def test_rank_hits_graph_boost_seam_runs_before_gate() -> None:
    hits = [_mk("a", 0.9), _mk("b", 0.8)]
    seen: list[str] = []

    def boost(raw: list[Any]) -> list[Any]:
        seen.extend(h.id for h in raw)
        return list(reversed(raw))

    out = rank_hits(hits, RankKnobs(min_sim=0.0, min_body_chars=0), graph_boost=boost)
    assert seen == ["a", "b"]  # boost saw the raw candidates
    assert [h.id for h in out] == ["b", "a"]  # boost reordering took effect


def test_rank_hits_drops_synthesis_sources() -> None:
    syn = _mk("syn", 0.9, type="synthesis", extra={"synthesis_sources": ["src1"]})
    src = _mk("src1", 0.85)
    other = _mk("o", 0.8)
    out = rank_hits([syn, src, other], RankKnobs(min_sim=0.0, min_body_chars=0))
    assert [h.id for h in out] == ["syn", "o"]  # src1 covered by syn, removed
