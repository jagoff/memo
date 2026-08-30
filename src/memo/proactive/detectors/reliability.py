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
    out: list[Nudge] = []
    try:
        pairs = mem.superseded_pairs()[:limit]
        # A nudge whose action cannot resolve is worse than no nudge. Pointing
        # at the successor is only an improvement while the successor exists,
        # and `superseded_pairs` reads the archive from disk without checking:
        # measured 2026-08-30, only one of the first three successors the
        # refreshed digest offered still resolved — the other two had been
        # retired since they won. The lookup sits inside the guard because the
        # module contract is that this detector never sinks a surface, and a
        # `mem` that cannot answer `get` would otherwise raise past it.
        pairs = [p for p in pairs if mem.get(p[1]) is not None]
    except Exception as exc:  # guarded — never sink a surface
        _log.debug("proactive.reliability failed: %s", exc)
        return []
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
