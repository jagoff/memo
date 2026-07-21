"""Health detector — low-confidence memories as an aggregate nudge.

Emits ONE `KIND_HEALTH` `Nudge` citing every low-confidence memory, sourced
from `mem.low_confidence_ids()`. Guarded: any exception returns `[]` — this
detector must never sink a proactive-surface call.
"""

from __future__ import annotations

import logging
from typing import Any

from ..nudge import KIND_HEALTH, Nudge

_log = logging.getLogger(__name__)


def detect_health(mem: Any, *, now: str, limit: int = 10) -> list[Nudge]:
    try:
        ids = mem.low_confidence_ids(threshold=0.4, limit=limit)
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
            title=f"{len(ids)} low-confidence memories — worth reviewing",
            evidence=tuple(ids),
            action="memo maintain",
            created_at=now,
        )
    ]
