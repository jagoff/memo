from __future__ import annotations

from dataclasses import replace

from memo.memory.record import MemoryRecord
from memo.memory.search_scoring_ops import _SearchScoringMixin
from memo.tiers import VerificationState


def _rec(id_: str, score: float, **extra):
    return MemoryRecord(
        id=id_,
        path=f"{id_}.md",
        title=id_,
        type="note",
        tags=[],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00",
        body="body",
        extra=dict(extra),
        score=score,
    )


class _Harness(_SearchScoringMixin):
    pass


def test_apply_quality_rerank_is_flag_gated(monkeypatch) -> None:
    hits = [_rec("old", 0.9, superseded_by="new"), _rec("new", 0.7)]
    monkeypatch.setenv("MEMO_QUALITY_RERANK", "0")
    assert [h.id for h in _Harness()._apply_quality_rerank(hits)] == ["old", "new"]

    monkeypatch.setenv("MEMO_QUALITY_RERANK", "1")
    out = _Harness()._apply_quality_rerank(hits)
    assert [h.id for h in out] == ["new", "old"]


def test_apply_quality_rerank_boosts_verified(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_QUALITY_RERANK", "1")
    verified = replace(_rec("verified", 0.7), verification_state=VerificationState.VERIFIED)
    out = _Harness()._apply_quality_rerank([_rec("plain", 0.72), verified])
    assert out[0].id == "verified"
