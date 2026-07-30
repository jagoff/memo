"""Exactly-once reconciliation for outcome-backed durable promotions."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from memo.errors import (
    IdentityConflictError,
    OperationalError,
    OperationalErrorCode,
    StorageError,
    ValidationError,
    WriteRefused,
)
from memo.identity import PrincipalIdentity
from memo.operational_event import (
    OperationalCommand,
    canonical_json_bytes,
)
from memo.operational_event_types import (
    DURABLE_PROMOTION_COMPLETED,
    DURABLE_PROMOTION_REJECTED,
    DURABLE_PROMOTION_REQUESTED,
    DURABLE_PROMOTION_RETRY_SCHEDULED,
)
from memo.util import utc_now_iso

if TYPE_CHECKING:
    from memo.memory.record import MemoryRecord
    from memo.operation_views import OperationalViewStore
    from memo.operational import OperationalStore
    from memo.operational_epoch import CommitContext

JsonScalar = str | int | float | bool | None
FrozenJson = JsonScalar | tuple["FrozenJson", ...] | Mapping[str, "FrozenJson"]
_PROMOTION_PREFIX = "promotion/"
_MAX_RETRY_SECONDS = 3600


def _canonical_timestamp(value: str, *, field: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValidationError(f"{field} must be a timezone-aware ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValidationError(f"{field} must be a timezone-aware ISO-8601 timestamp")
    return (
        parsed.astimezone(UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _json_value(value: object, *, path: str) -> JsonScalar | list[Any] | dict[str, Any]:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"{path} must not contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValidationError(f"{path} must contain only string mapping keys")
            result[key] = _json_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, path=f"{path}[]") for item in value]
    raise ValidationError(f"{path} contains unsupported value {type(value).__name__}")


def canonical_save_request(save_kwargs: Mapping[str, object]) -> dict[str, Any]:
    """Return a detached JSON request suitable for hashing and later save."""
    if not isinstance(save_kwargs, Mapping):
        raise ValidationError("save_kwargs must be a mapping")
    normalized = _json_value(save_kwargs, path="save_kwargs")
    if not isinstance(normalized, dict):
        raise ValidationError("save_kwargs must be a mapping")
    return normalized


def canonical_save_request_hash(save_kwargs: Mapping[str, object]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(canonical_save_request(save_kwargs))
    ).hexdigest()


def _freeze(value: object) -> FrozenJson:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"non-canonical frozen JSON value: {type(value).__name__}")


def _thaw(value: FrozenJson) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _normalized_idempotency_key(idempotency_key: str) -> str:
    if not isinstance(idempotency_key, str):
        raise ValidationError("idempotency_key must be a string")
    normalized = idempotency_key.strip()
    if not normalized:
        raise ValidationError("idempotency_key cannot be empty")
    return normalized


def promotion_operation_key(idempotency_key: str) -> str:
    normalized = _normalized_idempotency_key(idempotency_key)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{_PROMOTION_PREFIX}{digest}"


def _promotion_id(idempotency_key: str) -> str:
    return promotion_operation_key(idempotency_key).removeprefix(_PROMOTION_PREFIX)


def _normalized_source_event_ids(source_event_ids: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if not isinstance(source_event_ids, (tuple, list)) or not all(
        isinstance(value, str) for value in source_event_ids
    ):
        raise ValidationError("source_event_ids must be a list of strings")
    values = {value.strip() for value in source_event_ids}
    values.discard("")
    return tuple(sorted(values))


def _with_source_provenance(
    save_kwargs: Mapping[str, object],
    source_event_ids: tuple[str, ...],
) -> dict[str, Any]:
    normalized = canonical_save_request(save_kwargs)
    extra_raw = normalized.get("extra")
    extra = dict(extra_raw) if isinstance(extra_raw, dict) else {}
    provenance_raw = extra.get("provenance")
    provenance = dict(provenance_raw) if isinstance(provenance_raw, dict) else {}
    provenance["source_event_ids"] = list(source_event_ids)
    extra["provenance"] = provenance
    normalized["extra"] = extra
    return normalized


@dataclass(frozen=True)
class FrozenPromotionIntent:
    id: str
    idempotency_key: str
    operation_key: str
    request_hash: str
    save_kwargs: Mapping[str, FrozenJson]
    source_event_ids: tuple[str, ...]
    created_at: str
    attempts: int = 0

    def mutable_save_kwargs(self) -> dict[str, object]:
        return {key: _thaw(value) for key, value in self.save_kwargs.items()}

    def requested_payload(self) -> dict[str, object]:
        return {
            "promotion_id": self.id,
            "idempotency_key": self.idempotency_key,
            "operation_key": self.operation_key,
            "request_hash": self.request_hash,
            "save_kwargs": self.mutable_save_kwargs(),
            "source_event_ids": list(self.source_event_ids),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class OutboxRunReport:
    examined: int = 0
    completed: int = 0
    retried: int = 0
    quarantined: int = 0
    pending: int = 0


@dataclass(frozen=True)
class DurableOutboxAuthority:
    actor: PrincipalIdentity
    project: str
    workspace: str
    visibility: str = "owner"
    trace_id: str = ""

    def command(
        self,
        *,
        event_type: str,
        promotion_id: str,
        idempotency_key: str,
        payload: Mapping[str, object],
        caused_by: tuple[str, ...] = (),
    ) -> OperationalCommand:
        return OperationalCommand(
            event_type=event_type,
            actor=self.actor,
            target_id=promotion_id,
            project=self.project,
            workspace=self.workspace,
            expires_at=None,
            visibility=self.visibility,
            idempotency_key=idempotency_key,
            caused_by=caused_by,
            subject_uri=f"memo://durable-promotion/{promotion_id}",
            trace_id=self.trace_id,
            payload=dict(payload),
        )


def freeze_promotion_intent(
    *,
    idempotency_key: str,
    save_kwargs: Mapping[str, object],
    source_event_ids: tuple[str, ...] | list[str],
    created_at: str,
    attempts: int = 0,
) -> FrozenPromotionIntent:
    normalized_key = _normalized_idempotency_key(idempotency_key)
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0:
        raise ValidationError("attempts must be an integer >= 0")
    normalized_sources = _normalized_source_event_ids(source_event_ids)
    frozen_request = _with_source_provenance(save_kwargs, normalized_sources)
    return FrozenPromotionIntent(
        id=_promotion_id(normalized_key),
        idempotency_key=normalized_key,
        operation_key=promotion_operation_key(normalized_key),
        request_hash=canonical_save_request_hash(frozen_request),
        save_kwargs=MappingProxyType(
            {key: _freeze(value) for key, value in frozen_request.items()}
        ),
        source_event_ids=normalized_sources,
        created_at=_canonical_timestamp(created_at, field="created_at"),
        attempts=attempts,
    )


def frozen_intent_from_row(row: Mapping[str, object]) -> FrozenPromotionIntent:
    save_kwargs = row.get("save_kwargs")
    source_event_ids = row.get("source_event_ids")
    raw_idempotency_key = row.get("idempotency_key")
    if (
        not isinstance(save_kwargs, Mapping)
        or not isinstance(source_event_ids, list)
        or not all(isinstance(value, str) for value in source_event_ids)
        or not isinstance(raw_idempotency_key, str)
    ):
        raise ValidationError("stored durable promotion intent is malformed")
    normalized_key = _normalized_idempotency_key(raw_idempotency_key)
    normalized_sources = _normalized_source_event_ids(source_event_ids)
    normalized_request = canonical_save_request(save_kwargs)
    raw_attempts = row.get("attempts", 0)
    if (
        isinstance(raw_attempts, bool)
        or not isinstance(raw_attempts, int)
        or raw_attempts < 0
    ):
        raise ValidationError("stored durable promotion attempts are invalid")
    attempts = raw_attempts
    intent = FrozenPromotionIntent(
        id=_promotion_id(normalized_key),
        idempotency_key=normalized_key,
        operation_key=promotion_operation_key(normalized_key),
        request_hash=canonical_save_request_hash(normalized_request),
        save_kwargs=MappingProxyType(
            {key: _freeze(value) for key, value in normalized_request.items()}
        ),
        source_event_ids=normalized_sources,
        created_at=_canonical_timestamp(
            str(row.get("created_at") or ""),
            field="created_at",
        ),
        attempts=attempts,
    )
    if (
        intent.id != row.get("promotion_id")
        or intent.operation_key != row.get("operation_key")
        or intent.request_hash != row.get("request_hash")
    ):
        raise ValidationError("stored durable promotion identity is inconsistent")
    return intent


def deterministic_retry_at(created_at: str, attempt_number: int) -> str:
    if (
        isinstance(attempt_number, bool)
        or not isinstance(attempt_number, int)
        or attempt_number < 1
    ):
        raise ValidationError("attempt_number must be an integer >= 1")
    canonical = _canonical_timestamp(created_at, field="created_at")
    parsed = datetime.fromisoformat(canonical.replace("Z", "+00:00"))
    delay = min(2 ** (attempt_number - 1), _MAX_RETRY_SECONDS)
    return (
        (parsed + timedelta(seconds=delay))
        .astimezone(UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _safe_failure_reason(exc: BaseException) -> str:
    reason = " ".join(str(exc).split())
    return (reason or exc.__class__.__name__)[:512]


class DurableOutboxWorker:
    """Reconcile v2 promotion intents into authoritative Markdown exactly once."""

    def __init__(
        self,
        *,
        memory: Any,
        operational: OperationalStore,
        store: OperationalViewStore,
        authority: DurableOutboxAuthority,
        context_factory: Callable[[], CommitContext],
        clock: Callable[[], str] = utc_now_iso,
    ) -> None:
        self.memory = memory
        self.operational = operational
        self.store = store
        self.authority = authority
        self.context_factory = context_factory
        self.clock = clock

    def _commit(
        self,
        *,
        event_type: str,
        intent: FrozenPromotionIntent,
        idempotency_key: str,
        payload: Mapping[str, object],
        caused_by: tuple[str, ...] = (),
    ) -> None:
        command = self.authority.command(
            event_type=event_type,
            promotion_id=intent.id,
            idempotency_key=idempotency_key,
            payload=payload,
            caused_by=caused_by,
        )
        self.operational.commit(command, context=self.context_factory())

    def enqueue(
        self,
        *,
        idempotency_key: str,
        save_kwargs: Mapping[str, object],
        source_event_ids: tuple[str, ...] | list[str],
        created_at: str,
    ) -> FrozenPromotionIntent:
        normalized_key = _normalized_idempotency_key(idempotency_key)
        existing = self.store.outbox_intent(_promotion_id(normalized_key))
        if existing is not None:
            candidate = freeze_promotion_intent(
                idempotency_key=normalized_key,
                save_kwargs=save_kwargs,
                source_event_ids=source_event_ids,
                created_at=existing.created_at,
                attempts=existing.attempts,
            )
            if (
                candidate.request_hash != existing.request_hash
                or candidate.source_event_ids != existing.source_event_ids
            ):
                raise OperationalError(
                    OperationalErrorCode.IDEMPOTENCY_CONFLICT,
                    "durable promotion idempotency key identifies a different request",
                    retryable=False,
                )
            return existing
        intent = freeze_promotion_intent(
            idempotency_key=normalized_key,
            save_kwargs=save_kwargs,
            source_event_ids=source_event_ids,
            created_at=created_at,
        )
        self._commit(
            event_type=DURABLE_PROMOTION_REQUESTED,
            intent=intent,
            idempotency_key=f"durable-promotion/requested/{intent.id}",
            payload=intent.requested_payload(),
            caused_by=intent.source_event_ids,
        )
        return self.store.outbox_intent(intent.id) or intent

    def _reconcile_one(
        self,
        intent: FrozenPromotionIntent,
    ) -> tuple[MemoryRecord | None, BaseException | None]:
        try:
            saved: MemoryRecord = self.memory.save_operation(
                operation_key=intent.operation_key,
                request_hash=intent.request_hash,
                save_kwargs=intent.mutable_save_kwargs(),
            )
        except (IdentityConflictError, ValidationError, WriteRefused) as exc:
            self._commit(
                event_type=DURABLE_PROMOTION_REJECTED,
                intent=intent,
                idempotency_key=(
                    f"durable-promotion/rejected/{intent.id}/{intent.request_hash}"
                ),
                payload={
                    "promotion_id": intent.id,
                    "operation_key": intent.operation_key,
                    "request_hash": intent.request_hash,
                    "failure_class": exc.__class__.__name__,
                    "reason": _safe_failure_reason(exc),
                },
            )
            return None, exc
        except Exception as exc:
            attempt_number = intent.attempts + 1
            self._commit(
                event_type=DURABLE_PROMOTION_RETRY_SCHEDULED,
                intent=intent,
                idempotency_key=(
                    f"durable-promotion/retry/{intent.id}/{attempt_number}"
                ),
                payload={
                    "promotion_id": intent.id,
                    "operation_key": intent.operation_key,
                    "request_hash": intent.request_hash,
                    "attempt_number": attempt_number,
                    "failure_class": exc.__class__.__name__,
                    "retry_at": deterministic_retry_at(
                        intent.created_at,
                        attempt_number,
                    ),
                },
            )
            raise
        self._commit(
            event_type=DURABLE_PROMOTION_COMPLETED,
            intent=intent,
            idempotency_key=(
                f"durable-promotion/completed/{intent.id}/{intent.request_hash}"
            ),
            payload={
                "promotion_id": intent.id,
                "operation_key": intent.operation_key,
                "request_hash": intent.request_hash,
                "memory_id": saved.id,
            },
        )
        return saved, None

    def reconcile(self, intent: FrozenPromotionIntent) -> MemoryRecord:
        """Synchronously reconcile one caller-owned intent."""
        status = self.store.outbox_status(intent.id)
        if status is None:
            raise StorageError("durable promotion intent is absent from the view")
        state = str(status.get("status") or "")
        if state == "completed":
            existing = self.memory.find_by_operation_key(
                intent.operation_key,
                intent.request_hash,
            )
            if existing is None:
                raise StorageError(
                    "completed durable promotion has no authoritative memory"
                )
            return existing
        if state == "rejected":
            raise ValidationError(
                str(status.get("reason") or "durable promotion was rejected")
            )
        if state == "retry_scheduled" and _canonical_timestamp(
            self.clock(),
            field="clock",
        ) < _canonical_timestamp(str(status.get("retry_at") or ""), field="retry_at"):
            raise OperationalError(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                "durable promotion retry is not due",
                retryable=True,
                details={"retry_at": str(status.get("retry_at") or "")},
            )
        current = self.store.outbox_intent(intent.id)
        if current is None:
            raise StorageError("durable promotion intent disappeared from the view")
        saved, permanent = self._reconcile_one(current)
        if permanent is not None:
            raise permanent
        if saved is None:
            raise StorageError("durable promotion reconciliation produced no memory")
        return saved

    def run_once(self, *, limit: int = 100) -> OutboxRunReport:
        if isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be >= 1")
        intents = self.store.pending_outbox(limit=limit, now=self.clock())
        completed = 0
        quarantined = 0
        retried = 0
        for intent in intents:
            try:
                saved, permanent = self._reconcile_one(intent)
            except Exception:
                retried += 1
                raise
            if permanent is not None:
                quarantined += 1
                continue
            if saved is not None:
                completed += 1
        aggregate = self.store.outbox_report()
        return OutboxRunReport(
            examined=len(intents),
            completed=completed,
            retried=retried,
            quarantined=quarantined,
            pending=aggregate.pending,
        )


__all__ = [
    "DurableOutboxAuthority",
    "DurableOutboxWorker",
    "FrozenPromotionIntent",
    "OutboxRunReport",
    "canonical_save_request",
    "canonical_save_request_hash",
    "deterministic_retry_at",
    "freeze_promotion_intent",
    "frozen_intent_from_row",
    "promotion_operation_key",
]
