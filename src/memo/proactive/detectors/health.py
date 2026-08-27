"""Health detector — low-confidence memories as an aggregate nudge.

Emits ONE `KIND_HEALTH` `Nudge` for low-confidence memories, sourced from
`mem.low_confidence_ids()`. Evidence is capped at `limit` ids while the title
reports the real corpus total (`mem.low_confidence_count()`). Guarded: any
exception returns `[]` — this detector must never sink a proactive-surface
call.
"""

from __future__ import annotations

import logging
from typing import Any

from ..nudge import KIND_HEALTH, Nudge
from ._totals import total_or

_log = logging.getLogger(__name__)


def detect_health(mem: Any, *, now: str, limit: int = 10) -> list[Nudge]:
    try:
        ids = mem.low_confidence_ids(threshold=0.4, limit=limit)
        total = total_or(lambda: mem.low_confidence_count(threshold=0.4), len(ids))
    except Exception as exc:  # guarded — never sink a surface
        _log.debug("proactive.health failed: %s", exc)
        return []
    if not ids:
        return []
    return [
        Nudge.make(
            KIND_HEALTH,
            subject_id="low-confidence",
            urgency=0.2,
            value=0.5,
            title=f"{total} low-confidence memories — worth reviewing",
            evidence=tuple(ids),
            action="memo maintain",
            created_at=now,
        )
    ]
