"""Guard — flag a prior decision the user looks to be reversing (pre-turn).

Pure Python, no LLM (v1). A cheap gate over already-ranked recall hits:
a hit is a guard candidate when it is a durable decision/preference, scored
above threshold, and the prompt carries a reversal signal. memo SURFACES the
prior decision; it never blocks or claims the user is wrong.
"""

from __future__ import annotations

import re
from typing import Any

_GUARD_TYPES = frozenset({"decision", "preference"})

# Reversal signals — the user is changing a prior direction, not asking fresh.
_REVERSAL = re.compile(
    r"\b(instead|switch|actually|redo|revert|rather than|changed my mind|"
    r"no longer|en vez|en lugar|cambi|mejor us|revert[íi]|volv[ae]mos)\b",
    re.IGNORECASE,
)


def has_reversal_signal(prompt: str) -> bool:
    """True when the prompt reads like reversing a prior direction."""
    return bool(_REVERSAL.search(prompt or ""))


def guard_candidates(prompt: str, hits: list[Any], *, sim_threshold: float) -> list[Any]:
    """Prior decisions/preferences the prompt looks to be reversing, score-desc."""
    if not has_reversal_signal(prompt):
        return []
    cand = [
        h
        for h in hits
        if getattr(h, "type", None) in _GUARD_TYPES and (getattr(h, "score", None) or 0.0) >= sim_threshold
    ]
    return sorted(cand, key=lambda h: getattr(h, "score", None) or 0.0, reverse=True)
