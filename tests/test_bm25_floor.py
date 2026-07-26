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


def test_bm25_mode_does_not_gate_hit_below_cosine_floor() -> None:
    # A bm25-scale score (0.156) is below the cosine-calibrated min_sim (0.5),
    # but bm25 hits are already relevance-ranked — they must NOT be gated out
    # (the cold-start vec->bm25 downgrade returned nothing before this fix).
    hits = [_mk("a", 0.156)]
    out = rank_hits(hits, RankKnobs(min_sim=0.5, mode="bm25", min_body_chars=0))
    assert [h.id for h in out] == ["a"]


def test_vec_mode_still_gates_same_low_score() -> None:
    # The SAME 0.156 score is a genuine cosine similarity in vec mode and stays
    # gated; a 0.87 cosine passes. Proves the fix is scoped to bm25 mode only.
    hits = [_mk("low", 0.156), _mk("high", 0.87)]
    out = rank_hits(hits, RankKnobs(min_sim=0.5, mode="vec", min_body_chars=0))
    assert [h.id for h in out] == ["high"]


def test_bm25_mode_still_applies_min_body_chars_gate() -> None:
    # The body-length gate is orthogonal to the cosine floor and must still
    # apply in bm25 mode.
    hits = [
        _mk("short", 0.156, body="tiny"),
        _mk("long", 0.156, body="a body that is comfortably longer than the min_body_chars gate"),
    ]
    out = rank_hits(hits, RankKnobs(min_sim=0.5, mode="bm25", min_body_chars=40))
    assert [h.id for h in out] == ["long"]  # short body dropped, low score kept
