"""Query-side NL date-range parsing (ES + EN) — offline, stdlib only.

Mirrors the relative-date vocabulary of
`memo.memory.consolidate_ops._normalize_relative_dates` (the capture side) but
resolves to a hard inclusive [start, end] ISO-date window instead of annotating
text. Deliberately regex-based (no dateparser dep) per memo's offline posture.
"""

from __future__ import annotations

import datetime as _dt
import re as _re


def parse_date_range(
    text: str, ref_date: _dt.date | None = None
) -> tuple[str | None, str | None]:
    """Inclusive [date_from, date_to] ISO pair, or (None, None). Never raises."""
    try:
        ref = ref_date or _dt.date.today()
        t = (text or "").lower()

        m = _re.search(r"hace\s+(\d+)\s+d[ií]as?|(\d+)\s+days?\s+ago", t)
        if m:
            d = ref - _dt.timedelta(days=int(m.group(1) or m.group(2)))
            return d.isoformat(), d.isoformat()
        if _re.search(r"\banteayer\b", t):
            d = ref - _dt.timedelta(days=2)
            return d.isoformat(), d.isoformat()
        if _re.search(r"\bayer\b|\byesterday\b", t):
            d = ref - _dt.timedelta(days=1)
            return d.isoformat(), d.isoformat()
        if _re.search(r"\bhoy\b|\btoday\b", t):
            return ref.isoformat(), ref.isoformat()
        if _re.search(r"\bla\s+semana\s+pasada\b|\blast\s+week\b", t):
            end = ref - _dt.timedelta(days=ref.weekday() + 1)  # last Sunday
            return (end - _dt.timedelta(days=6)).isoformat(), end.isoformat()
        if _re.search(r"\besta\s+semana\b|\bthis\s+week\b", t):
            return (ref - _dt.timedelta(days=ref.weekday())).isoformat(), ref.isoformat()
        if _re.search(r"\bel\s+mes\s+pasado\b|\blast\s+month\b", t):
            end = ref.replace(day=1) - _dt.timedelta(days=1)
            return end.replace(day=1).isoformat(), end.isoformat()
        return None, None
    except Exception:
        return None, None
