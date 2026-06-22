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
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from functools import lru_cache, wraps
from typing import Any


def utc_now_iso() -> str:
    """UTC timestamp, ISO8601 with a ``Z`` suffix (e.g. ``2026-05-29T12:00:00Z``)."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@lru_cache(maxsize=8192)
def sha256_short(text: str) -> str:
    """First 16 hex chars of the SHA-256 of ``text`` (utf-8)."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


@lru_cache(maxsize=8192)
def sha256_full(text: str) -> str:
    """Full SHA-256 hex of ``text`` (utf-8). Used for cache keys where
    64-bit collision risk is unacceptable."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


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


def safe_operation(
    *,
    fallback: Any = None,
    log_level: int = logging.ERROR,
    error_message: str | None = None,
    reraise: bool = False,
    expected_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for standardized error handling with structured logging.

    Replaces silent ``except Exception: pass`` patterns with observable logging.

    Args:
        fallback: Value to return on error (ignored if reraise=True)
        log_level: Logging level (default: ERROR)
        error_message: Custom error message (default: auto-generated from function name)
        reraise: If True, re-raise after logging (default: False)
        expected_exceptions: Exception types to catch (default: all Exceptions)

    Example:
        @safe_operation(fallback=[], log_level=logging.WARNING)
        def risky_operation():
            # May fail, but failure is non-critical
            return get_external_data()
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import logging as _logging

            logger = _logging.getLogger(func.__module__)
            msg = error_message or f"{func.__name__} failed"
            try:
                return func(*args, **kwargs)
            except expected_exceptions as exc:
                logger.log(log_level, "%s: %s", msg, exc)
                if reraise:
                    raise
                return fallback
        return wrapper
    return decorator
