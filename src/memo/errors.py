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

from enum import StrEnum
from typing import Any


class MemoError(Exception):
    """Base for all memo-domain errors."""


class NotFoundError(MemoError, KeyError):
    """A requested record / entity / resource does not exist."""


class ValidationError(MemoError, ValueError):
    """Caller-supplied input failed validation before any side effect."""


class IdentityConflictError(ValidationError):
    """A requested write would make durable identity ambiguous.

    Structured attributes let CLI/MCP callers render a safe response without
    parsing exception text. The message deliberately contains no memory body.
    """

    def __init__(
        self,
        *,
        kind: str,
        incoming: dict[str, Any] | None = None,
        conflicts: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    ) -> None:
        self.kind = kind
        self.incoming = dict(incoming or {})
        self.conflicts = tuple(dict(item) for item in conflicts)
        ids = [str(item.get("id", ""))[:8] for item in self.conflicts if item.get("id")]
        suffix = f" ({', '.join(ids)})" if ids else ""
        super().__init__(f"memory identity conflict: {kind}{suffix}")


class RelationConflictError(ValidationError):
    """A relation already has an incompatible durable judgment."""

    def __init__(self, relation_id: str, existing: str, requested: str) -> None:
        self.relation_id = relation_id
        self.existing = existing
        self.requested = requested
        super().__init__(
            f"relation {relation_id} is already judged as {existing!r}; "
            f"cannot replace it with {requested!r}"
        )


class QueueFullError(MemoError, RuntimeError):
    """A bounded process-local coordinator rejected work before mutation."""

    retryable = True


class SetupError(MemoError, RuntimeError):
    """Declarative agent setup could not complete safely."""


class StorageError(MemoError, RuntimeError):
    """A storage-layer operation (sqlite / filesystem) failed. Wraps the
    low-level error with operation context so callers don't see bare
    `sqlite3.OperationalError` with no hint of what memo was doing."""


class OperationalErrorCode(StrEnum):
    """Stable public codes emitted by the operational runtime."""

    INVALID_EVENT = "invalid_event"
    UNKNOWN_SCHEMA = "unknown_schema"
    SEQUENCE_GAP = "sequence_gap"
    ANCHOR_CONFLICT = "anchor_conflict"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    SIGNATURE_INVALID = "signature_invalid"
    KEY_REVOKED = "key_revoked"
    EXPIRED = "expired"
    NOT_FOUND = "not_found"
    STORAGE_UNAVAILABLE = "storage_unavailable"


class OperationalError(MemoError, RuntimeError):
    """A typed operational failure safe to serialize across adapters."""

    def __init__(
        self,
        code: OperationalErrorCode,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = OperationalErrorCode(code)
        self.retryable = bool(retryable)
        self.details = dict(details or {})
        super().__init__(message)


class SignatureError(OperationalError):
    """An operational signature or its authorization context is invalid."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            OperationalErrorCode.SIGNATURE_INVALID,
            message,
            retryable=False,
            details=details,
        )


class KeyRevokedError(OperationalError):
    """A correctly identified signing key is no longer authorized."""

    def __init__(self, message: str) -> None:
        super().__init__(
            OperationalErrorCode.KEY_REVOKED,
            message,
            retryable=False,
        )


class AuthorityEpochError(OperationalError):
    """A request was not bound to the current durable authority marker."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            OperationalErrorCode.INVALID_EVENT,
            message,
            retryable=False,
            details=details,
        )


class FederationError(MemoError, RuntimeError):
    """A signed federation bundle failed ACL or integrity validation."""


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
    """Raised by `Memory.save()` when a native conflict or policy blocks a write.

    Carries the offending conflict dict so callers (CLI / MCP / agent)
    can show the user the conflict id, severity, and summary before
    they decide whether to resolve it or submit an explicit human override.
    """

    def __init__(self, conflict: dict[str, Any]) -> None:
        cid = conflict.get("conflict_id") or "?"
        summary = conflict.get("summary") or "(no summary)"
        super().__init__(
            f"Memo write policy blocked conflict {cid}: {summary}. "
            "Resolve the conflict or retry with an explicit override reason.",
        )
        self.conflict = dict(conflict)
