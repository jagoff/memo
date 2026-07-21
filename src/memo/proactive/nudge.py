from __future__ import annotations

import hashlib
from dataclasses import dataclass

KIND_CONTINUITY = "continuity"
KIND_RELIABILITY = "reliability"
KIND_DEJAVU = "dejavu"
KIND_HEALTH = "health"
KIND_ROI = "roi"


@dataclass(frozen=True)
class Nudge:
    id: str
    kind: str
    urgency: float
    value: float
    title: str
    evidence: tuple[str, ...]
    created_at: str
    detail: str = ""
    action: str | None = None
    ttl_days: int = 14

    @classmethod
    def make(
        cls,
        kind: str,
        *,
        subject_id: str,
        urgency: float,
        value: float,
        title: str,
        evidence: tuple[str, ...],
        created_at: str,
        detail: str = "",
        action: str | None = None,
        ttl_days: int = 14,
    ) -> Nudge:
        if not evidence:
            raise ValueError("Nudge.evidence must be non-empty (never fabricate)")
        nid = hashlib.sha256(f"{kind}:{subject_id}".encode()).hexdigest()[:16]
        return cls(
            id=nid,
            kind=kind,
            urgency=urgency,
            value=value,
            title=title,
            evidence=tuple(evidence),
            created_at=created_at,
            detail=detail,
            action=action,
            ttl_days=ttl_days,
        )
