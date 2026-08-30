"""Reliability detector — superseded/contradicted memories as nudges.

Emits one `KIND_RELIABILITY` `Nudge` per durable memory that is superseded by
a live memory, sourced from `mem.superseded_pairs()`. Guarded: any exception
returns `[]` — this detector must never sink a proactive-surface call.
"""

from __future__ import annotations

import logging
from typing import Any

from ..nudge import KIND_RELIABILITY, Nudge

_log = logging.getLogger(__name__)


def detect_reliability(mem: Any, *, now: str, limit: int = 20) -> list[Nudge]:
    try:
        pairs = mem.superseded_pairs()[:limit]
    except Exception as exc:  # guarded — never sink a surface
        _log.debug("proactive.reliability failed: %s", exc)
        return []
    out: list[Nudge] = []
    for stale_id, superseding_id, title in pairs:
        out.append(
            Nudge.make(
                KIND_RELIABILITY,
                subject_id=stale_id,
                urgency=0.9,
                value=0.8,
                title=f"You may be relying on a superseded fact: {title}",
                evidence=(superseding_id, stale_id),
                # The SUCCESSOR, not the stale side: `superseded_pairs` sources
                # the stale id from `memory_dir/inactive/`, which `memo get`
                # does not read — every action this emitted answered "not
                # found". The replacement is the memory worth reading anyway.
                action=f"memo get {superseding_id}",
                created_at=now,
            )
        )
    return out
