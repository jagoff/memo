from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from memo.errors import ValidationError
from memo.quality import apply_quality_rerank, classify_quality
from memo.tiers import VerificationState


@dataclass(frozen=True)
class _Hit:
    id: str
    score: float | None
    title: str = ""
    body: str = "durable body"
    type: str = "note"
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    verification_state: VerificationState = VerificationState.UNVERIFIED
    verified_at: int | None = None


def test_classify_quality_marks_superseded_as_stale() -> None:
    decision = classify_quality(_Hit("old", 0.9, extra={"superseded_by": "new"}))
    assert decision.bucket == "stale_or_conflicting"
    assert "superseded_by" in decision.reasons
    assert decision.multiplier < 1.0


def test_classify_quality_boosts_verified_and_supported_hit() -> None:
    hit = _Hit(
        "good",
        0.5,
        extra={"support_count": 3, "roi_score": 1.3},
        verification_state=VerificationState.VERIFIED,
    )
    decision = classify_quality(hit)
    assert decision.bucket == "current"
    assert "verified" in decision.reasons
    assert "support_count" in decision.reasons
    assert decision.multiplier > 1.0


def test_classify_quality_demotes_source_side_canonical_pointer() -> None:
    decision = classify_quality(_Hit("dup", 0.9, extra={"canonical_id": "canon"}))
    assert decision.bucket == "supporting"
    assert "redundant_source" in decision.reasons
    assert "canonical_or_synthesis" not in decision.reasons
    assert decision.multiplier < 1.0


def test_classify_quality_ignores_malformed_optional_quality_metadata() -> None:
    decision = classify_quality(
        _Hit("odd", 0.5, extra={"support_count": "many", "roi_score": "high"})
    )
    assert decision.bucket == "current"
    assert "support_count" not in decision.reasons
    assert "positive_roi" not in decision.reasons
    assert "low_roi" not in decision.reasons


def test_apply_quality_rerank_demotes_stale_but_keeps_it_visible() -> None:
    old = _Hit("old", 0.9, extra={"superseded_by": "new"})
    current = _Hit("new", 0.7, verification_state=VerificationState.VERIFIED)
    out = apply_quality_rerank([old, current])
    assert [h.id for h in out] == ["new", "old"]


def test_apply_quality_rerank_populates_explain() -> None:
    explain: dict[str, dict[str, Any]] = {}
    out = apply_quality_rerank(
        [_Hit("old", 0.9, extra={"invalidated": True}), _Hit("new", 0.7)],
        explain=explain,
    )
    assert [h.id for h in out] == ["new", "old"]
    assert explain["old"]["quality_bucket"] == "stale_or_conflicting"
    assert explain["old"]["quality_multiplier"] < 1.0
    assert "invalidated" in explain["old"]["quality_reasons"]


class _MutableHit:
    def __init__(self) -> None:
        self.id = "plain"
        self.score = 0.9
        self.type = "note"
        self.tags = []
        self.extra = {}
        self.verification_state = VerificationState.UNVERIFIED
        self.verified_at = None


def test_apply_quality_rerank_copies_non_dataclass_hit() -> None:
    hit = _MutableHit()
    out = apply_quality_rerank([hit])
    assert len(out) == 1
    assert out[0] is not hit
    assert out[0].score == 0.9
    assert hit.score == 0.9


class _CopyFailureHit:
    def __init__(self) -> None:
        self.id = "broken"
        self.score = 0.9
        self.type = "note"
        self.tags = []
        self.extra = {}
        self.verification_state = VerificationState.UNVERIFIED
        self.verified_at = None

    def __copy__(self) -> _CopyFailureHit:
        raise RuntimeError("copy intentionally fails")


def test_apply_quality_rerank_does_not_mutate_explain_when_copy_fails() -> None:
    explain: dict[str, dict[str, Any]] = {}
    with pytest.raises(ValidationError, match="Unsupported hit object"):
        apply_quality_rerank([_CopyFailureHit()], explain=explain)
    assert explain == {}
