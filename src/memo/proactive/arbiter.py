from __future__ import annotations

from dataclasses import dataclass

from .nudge import KIND_RELIABILITY, Nudge


def score(n: Nudge, mult: float, *, w_urgency: float = 0.6, w_value: float = 0.4) -> float:
    return (w_urgency * n.urgency + w_value * n.value) * mult


@dataclass(frozen=True)
class Routed:
    badge_count: int
    badge_kind: str | None
    digest: list[Nudge]
    urgent: Nudge | None


def route(
    candidates: list[Nudge],
    multipliers: dict[str, float],
    *,
    digest_top: int,
    urgent_min: float,
    can_push: bool,
    floor: float,
) -> Routed:
    def mult(n: Nudge) -> float:
        return multipliers.get(n.kind, 1.0)

    ranked = sorted(candidates, key=lambda n: score(n, mult(n)), reverse=True)
    urgent: Nudge | None = None
    if can_push:
        for n in ranked:
            if n.kind == KIND_RELIABILITY and score(n, mult(n)) >= urgent_min:
                urgent = n
                break
    return Routed(
        badge_count=len(ranked),
        badge_kind=ranked[0].kind if ranked else None,
        digest=ranked[:digest_top],
        urgent=urgent,
    )
