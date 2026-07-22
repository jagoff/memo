"""memo domain error hierarchy.

One place for every exception memo raises. `MemoError` is the root: callers
can `except MemoError` to catch anything memo-domain. Each concrete error
also inherits its closest builtin (ValueError/KeyError/RuntimeError) so
pre-existing `except ValueError`-style handlers keep working during the
migration.

Named `MemoError` (not `MemoryError`) to avoid shadowing the builtin OOM
error. memory.py re-exports these names for back-compat with existing
`from memo.memory import AmbiguousIdError` imports.
"""

from __future__ import annotations

from typing import Any


class MemoError(Exception):
    """Base for all memo-domain errors."""


class NotFoundError(MemoError, KeyError):
    """A requested record / entity / resource does not exist."""


class ValidationError(MemoError, ValueError):
    """Caller-supplied input failed validation before any side effect."""


class StorageError(MemoError, RuntimeError):
    """A storage-layer operation (sqlite / filesystem) failed. Wraps the
    low-level error with operation context so callers don't see bare
    `sqlite3.OperationalError` with no hint of what memo was doing."""


class ConfigConflictError(MemoError, RuntimeError):
    """A persisted setting changed after the configuration session opened."""

    def __init__(self, keys: tuple[str, ...]) -> None:
        self.keys = keys
        super().__init__(f"configuration changed externally: {', '.join(keys)}")


class ConfigTransactionError(StorageError):
    """A staged configuration batch could not commit or roll back cleanly."""


class AmbiguousIdError(MemoError, ValueError):
    """Raised when an id prefix matches more than one record. Carries
    the candidate matches so the caller can surface them in an error."""

    def __init__(self, prefix: str, matches: list[str]) -> None:
        super().__init__(
            f"Ambiguous id prefix {prefix!r}: {len(matches)} matches "
            f"({', '.join(m[:8] for m in matches[:5])}{'...' if len(matches) > 5 else ''})",
        )
        self.prefix = prefix
        self.matches = matches


class WriteRefused(MemoError, RuntimeError):
    """Raised by `Memory.save()` when a synapse RealityConflict with
    `freeze_write=true` overlaps the topic of the pending write.

    Carries the offending conflict dict so callers (CLI / MCP / agent)
    can show the user the conflict id, severity, and summary before
    they decide to retry with `respect_synapse_freeze=False`.
    """

    def __init__(self, conflict: dict[str, Any]) -> None:
        cid = conflict.get("conflict_id") or "?"
        summary = conflict.get("summary") or "(no summary)"
        super().__init__(
            f"Synapse freeze-write active on conflict {cid}: {summary}. "
            f"Resolve the conflict in synapse or retry with "
            f"`respect_synapse_freeze=False`.",
        )
        self.conflict = dict(conflict)
