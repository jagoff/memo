"""Best-effort corpus totals for aggregate nudges.

An aggregate detector cites at most `limit` ids as evidence, so `len(ids)` is
the evidence cap — not how many there are. These detectors read the real count
from a separate facade query, but `mem` is duck-typed (`Any`): a source that
predates the count method must still get its nudge rather than lose it to the
detector's outer guard. Hence this narrower fallback, called from *inside* that
guard so anything unforeseen still degrades to no nudge instead of raising.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable

_log = logging.getLogger(__name__)


def total_or(count: Callable[[], int], fallback: int) -> int:
    """`count()`, or `fallback` when the source can't answer.

    Catches the three ways a duck-typed source fails to produce a count — no
    such method (`AttributeError`), a different signature or a non-numeric
    return (`TypeError`/`ValueError`), or the underlying store erroring
    (`sqlite3.Error`) — because the count only sharpens a nudge's wording and
    must never cost the nudge itself.
    """
    try:
        return int(count())
    except (AttributeError, TypeError, ValueError, sqlite3.Error) as exc:
        _log.debug("proactive: corpus count unavailable (%s)", exc)
        return fallback
