"""Shared low-level helpers consolidated from per-module duplicates.

Only genuinely-identical helpers live here. Note that the various
``_now_iso`` helpers across the codebase are intentionally NOT here:
they diverge on timezone and precision (UTC-with-Z vs local-with-ms vs
UTC-seconds) because callers depend on those differences — e.g. the
time-machine needs millisecond precision for within-second ordering.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from functools import lru_cache, wraps
from typing import Any


def rename_legacy_table(conn: sqlite3.Connection, old: str, new: str) -> None:
    """Idempotently rename a legacy table ``old`` -> ``new``.

    No-op when ``old`` is absent or ``new`` already exists, so it is safe to
    call before every ``CREATE TABLE IF NOT EXISTS new`` on a connection that
    may hold a pre-rename schema. Must run BEFORE the CREATE so the rename
    isn't blocked by a freshly-created empty ``new`` table.
    """
    try:
        tables = {
            r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    except sqlite3.Error:
        return
    if old in tables and new not in tables:
        with contextlib.suppress(sqlite3.Error):
            conn.execute(f"ALTER TABLE {old} RENAME TO {new}")


def rename_legacy_columns(conn: sqlite3.Connection, table: str, renames: dict[str, str]) -> None:
    """Idempotently rename legacy columns on ``table`` (old -> new).

    No-op per column when the table/old-column is absent or the new column
    already exists. SQLite >= 3.25 ``RENAME COLUMN`` updates dependent indexes
    and constraints automatically. Call BEFORE ``CREATE TABLE IF NOT EXISTS``
    so an existing pre-rename table is migrated rather than skipped.
    """
    try:
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return
    if not cols:
        return
    for old, new in renames.items():
        if old in cols and new not in cols:
            with contextlib.suppress(sqlite3.Error):
                conn.execute(f'ALTER TABLE {table} RENAME COLUMN "{old}" TO "{new}"')


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
