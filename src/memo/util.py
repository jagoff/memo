"""Shared low-level helpers consolidated from per-module duplicates.

Only genuinely-identical helpers live here. Note that the various
``_now_iso`` helpers across the codebase are intentionally NOT here:
they diverge on timezone and precision (UTC-with-Z vs local-with-ms vs
UTC-seconds) because callers depend on those differences — e.g. the
time-machine needs millisecond precision for within-second ordering.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any


def utc_now_iso() -> str:
    """UTC timestamp, ISO8601 with a ``Z`` suffix (e.g. ``2026-05-29T12:00:00Z``)."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@lru_cache(maxsize=8192)
def sha256_short(text: str) -> str:
    """First 16 hex chars of the SHA-256 of ``text`` (utf-8, lossy decode)."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def stable_hash(value: Any) -> str:
    """Deterministic full SHA-256 hex of ``value`` via canonical JSON.

    ``sort_keys`` + compact separators + ``default=str`` make the digest
    stable across dict ordering and tolerant of date/Path objects that
    YAML/frontmatter leave as native types.
    """
    raw = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
