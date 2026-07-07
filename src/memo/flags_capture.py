"""Crusher flags for JSON array compression on ingest.

Wave 1 token economy feature: L1 (SmartCrusher) drops low-relevance JSON rows
before indexing, saving 60–92% on structured data without losing retrieval quality.
"""

from __future__ import annotations

from memo.flags_base import FlagSpec, _spec

SPECS: tuple[FlagSpec, ...] = (
    _spec(
        "MEMO_CRUSHER_ENABLED",
        "bool",
        True,
        "capture",
        "Enable JSON array crushing on ingest (L1 SmartCrusher). When on, large JSON "
        "arrays are deduplicated before indexing, keeping only high-relevance rows and "
        "caching originals for later retrieval. Default: ON. Set to 0 to disable.",
        opt_out=True,
    ),
    _spec(
        "MEMO_CRUSHER_ROWS_KEEP_RATIO",
        "float",
        0.2,
        "capture",
        "Fraction of JSON array rows to keep after crushing (default: 0.2 = top 20%). "
        "Clamped to [0.05, 1.0]. Value of 1.0 disables crushing (keeps all rows).",
        min_val=0.05,
        max_val=1.0,
    ),
    _spec(
        "MEMO_CRUSHER_CACHE_TTL_DAYS",
        "int",
        30,
        "capture",
        "TTL for crush cache entries in days (default: 30). Older entries are evicted "
        "during maintenance. Minimum: 1 day.",
        min_val=1,
    ),
)


def flag_crusher_enabled() -> bool:
    """Enable JSON array crushing on ingest (default: ON)."""
    from memo.flags import flag_bool
    return flag_bool("MEMO_CRUSHER_ENABLED")


def flag_crusher_keep_ratio() -> float:
    """Fraction of JSON array rows to keep (default: 0.2 = top 20%).

    Returns float, clamped to [0.05, 1.0].
    """
    from memo.flags import flag_float
    val = flag_float("MEMO_CRUSHER_ROWS_KEEP_RATIO")
    return 0.2 if val is None else max(0.05, min(1.0, val))  # Clamp [0.05, 1.0]


def flag_crusher_cache_ttl_days() -> int:
    """TTL for crush cache entries (default: 30 days).

    Returns int >= 1.
    """
    from memo.flags import flag_int
    val = flag_int("MEMO_CRUSHER_CACHE_TTL_DAYS")
    return 30 if val is None else max(1, val)
