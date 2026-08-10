"""Valid-time ``as_of`` parsing at memo's user/agent boundaries.

Its own module because ``memo.util`` is an enforced pure-stdlib leaf
(``tests/test_architecture_boundaries.py``) and this raises a domain error.
"""

from __future__ import annotations

from datetime import datetime

from memo.errors import ValidationError


def validate_as_of(as_of: str | None) -> str | None:
    """Return ``as_of`` unchanged, or raise when it is not an ISO boundary.

    The store layer is deliberately lenient about a bound it cannot parse:
    ``store.bm25_queries._normalize_as_of`` falls back to the raw string and
    ``memory.search_ops._passes_validity_gate`` keeps the row rather than
    dropping it, so a garbled boundary never crashes a query mid-flight.

    At the EDGE that leniency reads as a lie: a valid-time query with a typo
    silently degrades to no filter at all and answers from the present while
    echoing the bad bound back as if it had been honoured. Call this wherever
    an ``as_of`` first arrives from a user or an agent, so the typo is refused
    where it can still be corrected. Accepts exactly what ``_normalize_as_of``
    understands — a bare ``YYYY-MM-DD``, a naive or offset-aware ISO 8601
    datetime, and ``Z`` as an offset alias.
    """
    if as_of is None:
        return None
    candidate = as_of.strip()
    try:
        if not candidate:
            raise ValueError("empty")
        datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(
            f"invalid --as-of value {as_of!r}: expected YYYY-MM-DD or a full ISO 8601 "
            "timestamp (e.g. 2026-01-31 or 2026-01-31T14:00:00-03:00)"
        ) from exc
    return as_of
