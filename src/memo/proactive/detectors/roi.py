"""ROI detector — dead memories (never accessed) as an aggregate nudge.

Emits ONE `KIND_ROI` `Nudge` citing every durable memory with zero recorded
access, sourced from `mem.dead_memory_ids()`. Guarded: any exception returns
`[]` — this detector must never sink a proactive-surface call.
"""

from __future__ import annotations

import logging
from typing import Any

from ..nudge import KIND_ROI, Nudge

_log = logging.getLogger(__name__)


def detect_roi(mem: Any, *, now: str, limit: int = 10) -> list[Nudge]:
    try:
        ids = mem.dead_memory_ids(limit=limit)
    except Exception as exc:  # guarded — never sink a surface
        _log.debug("proactive.roi failed: %s", exc)
        return []
    if not ids:
        return []
    return [
        Nudge.make(
            KIND_ROI,
            subject_id="dead-memories",
            urgency=0.1,
            value=0.4,
            title=f"{len(ids)} memories never surfaced — candidates to prune",
            evidence=tuple(ids),
            created_at=now,
        )
    ]
