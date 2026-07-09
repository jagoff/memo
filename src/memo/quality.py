from __future__ import annotations

import copy
from dataclasses import dataclass, is_dataclass
from dataclasses import replace as dc_replace
from typing import Any, cast

from memo.errors import ValidationError
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


def _hit_id(hit: Any) -> str:
    return str(getattr(hit, "id", "") or "")


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def is_canonical_memory(hit: Any) -> bool:
    extra = _extra(hit)
    if str(getattr(hit, "type", "") or "") in {"synthesis", "profile"}:
        return True
    for key in ("synthesis_source_memories", "synthesis_sources"):
        value = extra.get(key)
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, (list, tuple, set)) and any(str(item).strip() for item in value):
            return True
    return False


def _source_side_canonical_id(hit: Any) -> str:
    canonical_id = str(_extra(hit).get("canonical_id") or "").strip()
    if canonical_id and canonical_id != _hit_id(hit):
        return canonical_id
    return ""


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

    support_count = _optional_int(extra.get("support_count"))
    if support_count is not None and support_count > 0:
        reasons.append("support_count")
        multiplier *= min(1.20, 1.0 + support_count * 0.03)

    roi_score = _optional_float(extra.get("roi_score"))
    if roi_score is not None:
        if roi_score > 1.0:
            reasons.append("positive_roi")
            multiplier *= min(1.15, roi_score)
        elif roi_score < 0.5:
            reasons.append("low_roi")
            multiplier *= max(0.50, roi_score)

    if _source_side_canonical_id(hit) and not is_canonical_memory(hit):
        reasons.append("redundant_source")
        multiplier *= 0.85
        if bucket == "current":
            bucket = "supporting"

    if is_canonical_memory(hit):
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

        try:
            if is_dataclass(hit):
                ranked = dc_replace(cast(Any, hit), score=score)
            else:
                ranked = copy.copy(hit)
                ranked.score = score
        except Exception as exc:
            raise ValidationError(
                "Unsupported hit object for quality rerank; expected a dataclass or "
                "a hit-shaped object with a writable `score` attribute."
            ) from exc

        if explain is not None:
            hid = str(getattr(hit, "id", ""))
            entry = explain.setdefault(hid, {})
            entry["quality_bucket"] = decision.bucket
            entry["quality_multiplier"] = decision.multiplier
            entry["quality_reasons"] = list(decision.reasons)
            entry["quality_score"] = score

        scored.append((score, -index, ranked, decision))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [hit for _score, _index, hit, _decision in scored]
