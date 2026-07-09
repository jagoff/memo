from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dc_replace
from typing import Any

from memo.tiers import VerificationState


@dataclass(frozen=True)
class QualityDecision:
    """Pure quality classification for a single hit."""

    bucket: str
    multiplier: float
    reasons: tuple[str, ...]


def _extra(hit: Any) -> dict[str, Any]:
    raw = getattr(hit, "extra", None)
    return raw if isinstance(raw, dict) else {}


def _verification_value(hit: Any) -> str:
    value = getattr(hit, "verification_state", VerificationState.UNVERIFIED)
    return getattr(value, "value", str(value))


def classify_quality(hit: Any) -> QualityDecision:
    """Classify quality signals on a hit and return a multiplicative score factor."""

    extra = _extra(hit)
    reasons: list[str] = []
    multiplier = 1.0
    bucket = "current"

    if extra.get("superseded_by"):
        reasons.append("superseded_by")
        multiplier *= 0.35
        bucket = "stale_or_conflicting"

    if extra.get("invalidated") or extra.get("invalidated_at"):
        reasons.append("invalidated")
        multiplier *= 0.25
        bucket = "stale_or_conflicting"

    if extra.get("contradiction_status") in {"lost", "resolved_loser", "kept_other"}:
        reasons.append("contradiction_loser")
        multiplier *= 0.35
        bucket = "stale_or_conflicting"

    if bool(extra.get("secret")) or "secret" in {str(t) for t in getattr(hit, "tags", []) or []}:
        reasons.append("sensitive")

    verification = _verification_value(hit)
    if verification == "verified":
        reasons.append("verified")
        multiplier *= 1.10
    elif verification in {"rejected", "invalid"}:
        reasons.append("verification_rejected")
        multiplier *= 0.30
        bucket = "stale_or_conflicting"

    support_count = int(extra.get("support_count") or 0)
    if support_count > 0:
        reasons.append("support_count")
        multiplier *= min(1.20, 1.0 + support_count * 0.03)

    roi_score = extra.get("roi_score")
    if isinstance(roi_score, (int, float)):
        if roi_score > 1.0:
            reasons.append("positive_roi")
            multiplier *= min(1.15, float(roi_score))
        elif roi_score < 0.5:
            reasons.append("low_roi")
            multiplier *= max(0.50, float(roi_score))

    if extra.get("canonical_id") or extra.get("synthesis_source_memories"):
        reasons.append("canonical_or_synthesis")
        multiplier *= 1.05

    return QualityDecision(
        bucket=bucket,
        multiplier=round(multiplier, 6),
        reasons=tuple(reasons),
    )


def apply_quality_rerank(
    hits: list[Any],
    *,
    explain: dict[str, dict[str, Any]] | None = None,
) -> list[Any]:
    """Apply multiplicative quality-adjusted ranking without side effects."""

    scored: list[tuple[float, int, Any, QualityDecision]] = []
    for index, hit in enumerate(hits):
        decision = classify_quality(hit)
        base = float(getattr(hit, "score", None) or 0.0)
        score = base * decision.multiplier

        if explain is not None:
            hid = str(getattr(hit, "id", ""))
            entry = explain.setdefault(hid, {})
            entry["quality_bucket"] = decision.bucket
            entry["quality_multiplier"] = decision.multiplier
            entry["quality_reasons"] = list(decision.reasons)
            entry["quality_score"] = score

        try:
            hit = dc_replace(hit, score=score)
        except TypeError:
            hit.score = score

        scored.append((score, -index, hit, decision))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [hit for _score, _index, hit, _decision in scored]
