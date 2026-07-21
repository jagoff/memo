"""Continuity detector — open loops as nudges.

Emits one `KIND_CONTINUITY` `Nudge` per open loop, sourced from
`mem.open_loops(limit)`. Guarded: any exception returns `[]` — this detector
must never sink a proactive-surface call.
"""

from __future__ import annotations

import logging
from typing import Any

from ..nudge import KIND_CONTINUITY, Nudge

_log = logging.getLogger(__name__)


def detect_continuity(mem: Any, *, now: str, limit: int = 5) -> list[Nudge]:
    try:
        loops = mem.open_loops(limit)
    except Exception as exc:  # guarded — never sink a surface
        _log.debug("proactive.continuity failed: %s", exc)
        return []
    return [
        Nudge.make(
            KIND_CONTINUITY,
            subject_id=mid,
            urgency=0.4,
            value=0.7,
            title=f"Open loop: {text}",
            evidence=(mid,),
            created_at=now,
        )
        for mid, text in loops
    ]
