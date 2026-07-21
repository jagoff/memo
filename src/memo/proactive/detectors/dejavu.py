"""Déjà-vu detector — recurring queries that already have a matching memory.

Emits one `KIND_DEJAVU` `Nudge` per `(memo_id, pattern_text)` pair, sourced
from `mem.recurring_pattern_pairs(limit)`. Guarded: any exception returns
`[]` — this detector must never sink a proactive-surface call.

Limitation: true déjà-vu ("you just asked this again") wants the CURRENT
recall's live hits compared against history, but that context doesn't exist
outside a live recall call — this detector runs at dream/refresh time, not
inline with a query. `recurring_pattern_pairs` instead mines your
most-repeated PAST prompts from the recall log and re-runs `search()` to
find a real citable memory for each; a pattern with no match is dropped, so
this never fabricates a citation.
"""

from __future__ import annotations

import logging
from typing import Any

from ..nudge import KIND_DEJAVU, Nudge

_log = logging.getLogger(__name__)


def detect_dejavu(mem: Any, *, now: str, limit: int = 5) -> list[Nudge]:
    try:
        pairs = mem.recurring_pattern_pairs(limit=limit)
    except Exception as exc:  # guarded — never sink a surface
        _log.debug("proactive.dejavu failed: %s", exc)
        return []
    return [
        Nudge.make(
            KIND_DEJAVU,
            subject_id=memo_id,
            urgency=0.3,
            value=0.6,
            title=f"Recurring: {pattern_text} — you already have this",
            evidence=(memo_id,),
            created_at=now,
        )
        for memo_id, pattern_text in pairs
    ]
