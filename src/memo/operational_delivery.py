"""Event-derived delivery, acknowledgement, retry, and cursor state."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

from memo.contracts import Visibility
from memo.errors import OperationalError, OperationalErrorCode
from memo.identity import PrincipalIdentity
from memo.operational import OperationalStore
from memo.operational_epoch import CommitContext
from memo.operational_event import OperationalCommand
from memo.operational_event_types import (
    DELIVERY_ACK_RECORDED,
    DELIVERY_CURSOR_ADVANCED,
    DELIVERY_EXPIRED,
    DELIVERY_KNOWN_FAILED,
    DELIVERY_PRESENTED,
    DELIVERY_RESERVED,
    DELIVERY_UNCERTAIN,
    MESSAGE_SENT,
)

ContextFactory = Callable[[PrincipalIdentity], CommitContext]
_TERMINAL = frozenset({"acknowledged", "expired"})
_ALLOWED = {
    "pending": frozenset({"reserved", "acknowledged", "expired"}),
    "reserved": frozenset(
        {"presented", "acknowledged", "known_failed", "uncertain", "expired"}
    ),
    "presented": frozenset({"acknowledged", "expired"}),
    "known_failed": frozenset({"reserved", "expired"}),
    "uncertain": frozenset({"presented", "known_failed", "expired"}),
    "acknowledged": frozenset(),
    "expired": frozenset(),
}
_TRANSITION_EVENTS = {
    "reserved": DELIVERY_RESERVED,
    "presented": DELIVERY_PRESENTED,
    "known_failed": DELIVERY_KNOWN_FAILED,
    "uncertain": DELIVERY_UNCERTAIN,
    "expired": DELIVERY_EXPIRED,
}


def _invalid(message: str) -> OperationalError:
    return OperationalError(
        OperationalErrorCode.INVALID_EVENT,
        message,
        retryable=False,
    )


def _canonical_time(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("delivery timestamps must include a timezone")
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _delivery_id(event_id: str, target_id: str) -> str:
    return hashlib.sha256(f"{event_id}\0{target_id}".encode()).hexdigest()


def _clock_key(value: str) -> tuple[int, int, str]:
    try:
        first, second, origin = value.split("-", 2)
        return int(first), int(second), origin
    except (TypeError, ValueError) as exc:
        raise ValueError("logical_clock must be '<counter>-<subcounter>-<origin>'") from exc


@dataclass(frozen=True)
class RetryPolicy:
    initial_delay_seconds: float = 1.0
    multiplier: float = 2.0
    max_delay_seconds: float = 60.0
    max_attempts: int = 8

    def __post_init__(self) -> None:
        values = (
            self.initial_delay_seconds,
            self.multiplier,
            self.max_delay_seconds,
        )
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("retry delays and multiplier must be positive finite values")
        if isinstance(self.max_attempts, bool) or not 1 <= self.max_attempts <= 32:
            raise ValueError("max_attempts must be between 1 and 32")

    def delay(self, attempt: int) -> float:
        return min(
            self.initial_delay_seconds * self.multiplier ** max(0, attempt - 1),
            self.max_delay_seconds,
        )


@dataclass(frozen=True)
class DeliveryView:
    id: str
    message_id: str
    target_id: str
    state: str
    terminal_id: str | None
    attempt_count: int
    next_attempt_at: str | None
    deadline_at: str | None
    last_error_code: str | None
    ack_actor_id: str | None
    ack_event_id: str | None
    channel: str = ""
    message_event_id: str = ""


@dataclass(frozen=True)
class CursorView:
    consumer_id: str
    channel: str
    logical_clock: str
    event_id: str


class DeliveryService:
    def __init__(
        self,
        store: OperationalStore,
        *,
        context_factory: ContextFactory,
        retry_policy: RetryPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if store.backend_version != 2:
            raise _invalid("native delivery requires operational ledger v2")
        self.store = store
        self.context_factory = context_factory
        self.retry_policy = retry_policy or RetryPolicy()
        self.clock = clock or (lambda: datetime.now(UTC))

    def _commit(
        self,
        identity: PrincipalIdentity,
        *,
        event_type: str,
        target_id: str,
        project: str,
        subject_uri: str,
        payload: Mapping[str, object],
        idempotency_key: str,
        caused_by: tuple[str, ...] = (),
    ):
        key = idempotency_key.strip()
        if not key:
            raise ValueError("idempotency_key must be non-empty")
        return self.store.commit(
            OperationalCommand(
                event_type=event_type,
                actor=identity,
                target_id=target_id,
                project=project or "_global",
                workspace="",
                expires_at=None,
                visibility=Visibility.LOCAL_ONLY.value,
                idempotency_key=key,
                caused_by=caused_by,
                subject_uri=subject_uri,
                trace_id="",
                payload=payload,
            ),
            context=self.context_factory(identity),
        )

    def deliveries(self, *, message_id: str | None = None) -> list[DeliveryView]:
        rows: dict[str, DeliveryView] = {}
        for event in self.store.ledger.validated_events():
            payload = event.payload
            if event.event_type == MESSAGE_SENT:
                for target in payload["target_ids"]:
                    target_id = str(target)
                    delivery_id = _delivery_id(event.event_id, target_id)
                    rows[delivery_id] = DeliveryView(
                        id=delivery_id,
                        message_id=str(payload["message_id"]),
                        target_id=target_id,
                        state="pending",
                        terminal_id=None,
                        attempt_count=0,
                        next_attempt_at=str(payload["created_at"]),
                        deadline_at=event.expires_at,
                        last_error_code=None,
                        ack_actor_id=None,
                        ack_event_id=None,
                        channel=str(payload["channel"]),
                        message_event_id=event.event_id,
                    )
                continue
            if event.event_type not in {
                DELIVERY_RESERVED,
                DELIVERY_PRESENTED,
                DELIVERY_ACK_RECORDED,
                DELIVERY_KNOWN_FAILED,
                DELIVERY_UNCERTAIN,
                DELIVERY_EXPIRED,
            }:
                continue
            key = str(payload["delivery_id"])
            current = rows.get(key)
            if current is None:
                continue
            if event.event_type == DELIVERY_ACK_RECORDED:
                rows[key] = replace(
                    current,
                    state="acknowledged",
                    next_attempt_at=None,
                    ack_actor_id=str(payload["ack_actor_id"]),
                    ack_event_id=str(payload["ack_event_id"]),
                )
                continue
            transition = {
                DELIVERY_RESERVED: "reserved",
                DELIVERY_PRESENTED: "presented",
                DELIVERY_KNOWN_FAILED: "known_failed",
                DELIVERY_UNCERTAIN: "uncertain",
                DELIVERY_EXPIRED: "expired",
            }[event.event_type]
            attempt = int(payload["attempt_count"])
            transitioned_at = _parse_time(str(payload["transitioned_at"]))
            next_attempt: str | None = None
            if transition == "known_failed" and attempt < self.retry_policy.max_attempts:
                next_attempt = _canonical_time(
                    transitioned_at + timedelta(seconds=self.retry_policy.delay(attempt))
                )
            rows[key] = replace(
                current,
                state=transition,
                terminal_id=str(payload["terminal_id"]) or current.terminal_id,
                attempt_count=attempt,
                next_attempt_at=next_attempt,
                last_error_code=str(payload["error_code"]) or None,
            )
        selected = [row for row in rows.values() if message_id is None or row.message_id == message_id]
        return sorted(selected, key=lambda row: (row.message_event_id, row.target_id))

    def status(self, delivery_id: str) -> DeliveryView:
        row = next((item for item in self.deliveries() if item.id == delivery_id), None)
        if row is None:
            raise _invalid(f"unknown delivery: {delivery_id}")
        return row

    def transition(
        self,
        *,
        identity: PrincipalIdentity,
        delivery_id: str,
        state: str,
        idempotency_key: str,
        terminal_id: str = "",
        error_code: str = "",
        at: datetime | None = None,
    ) -> DeliveryView:
        current = self.status(delivery_id)
        if state not in _ALLOWED.get(current.state, frozenset()):
            if current.state == state:
                return current
            raise _invalid(f"delivery transition {current.state}->{state} is not allowed")
        event_type = _TRANSITION_EVENTS.get(state)
        if event_type is None:
            raise _invalid(f"delivery transition requires dedicated operation: {state}")
        attempt = current.attempt_count + (1 if state == "reserved" else 0)
        if attempt > self.retry_policy.max_attempts:
            raise _invalid("delivery retry attempts are exhausted")
        self._commit(
            identity,
            event_type=event_type,
            target_id=delivery_id,
            project="_global",
            subject_uri=f"memo://delivery/{delivery_id}",
            payload={
                "delivery_id": delivery_id,
                "message_id": current.message_id,
                "target_id": current.target_id,
                "transitioned_at": _canonical_time(at or self.clock()),
                "attempt_count": attempt,
                "terminal_id": terminal_id,
                "error_code": error_code,
            },
            idempotency_key=idempotency_key,
            caused_by=(current.message_event_id,),
        )
        return self.status(delivery_id)

    def reserve_due(
        self,
        *,
        identity: PrincipalIdentity,
        now: datetime,
        limit: int = 100,
    ) -> list[DeliveryView]:
        if isinstance(limit, bool) or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        current_time = now.astimezone(UTC)
        reserved: list[DeliveryView] = []
        for row in self.deliveries():
            if len(reserved) >= limit:
                break
            if row.deadline_at and _parse_time(row.deadline_at) <= current_time:
                self.transition(
                    identity=identity,
                    delivery_id=row.id,
                    state="expired",
                    idempotency_key=f"expire/{row.id}/{row.attempt_count}",
                    at=now,
                )
                continue
            if row.state not in {"pending", "known_failed"}:
                continue
            if row.attempt_count >= self.retry_policy.max_attempts:
                continue
            if row.next_attempt_at and _parse_time(row.next_attempt_at) > current_time:
                continue
            reserved.append(
                self.transition(
                    identity=identity,
                    delivery_id=row.id,
                    state="reserved",
                    idempotency_key=f"reserve/{row.id}/{row.attempt_count + 1}",
                    at=now,
                )
            )
        return reserved

    def acknowledge(
        self,
        *,
        identity: PrincipalIdentity,
        message_id: str,
        idempotency_key: str,
    ) -> DeliveryView:
        candidates = [
            item
            for item in self.deliveries(message_id=message_id)
            if item.target_id in {identity.actor_id, identity.principal_id}
        ]
        if len(candidates) != 1:
            raise _invalid("delivery ACK target differs from authenticated actor")
        current = candidates[0]
        if current.state == "acknowledged":
            return current
        if "acknowledged" not in _ALLOWED[current.state]:
            raise _invalid(f"delivery cannot be acknowledged from {current.state}")
        result = self._commit(
            identity,
            event_type=DELIVERY_ACK_RECORDED,
            target_id=current.id,
            project="_global",
            subject_uri=f"memo://delivery/{current.id}/ack",
            payload={
                "delivery_id": current.id,
                "message_id": current.message_id,
                "target_id": current.target_id,
                "ack_actor_id": identity.actor_id,
                "ack_event_id": hashlib.sha256(
                    f"{current.id}\0{idempotency_key}".encode()
                ).hexdigest(),
                "transitioned_at": _canonical_time(self.clock()),
            },
            idempotency_key=idempotency_key,
            caused_by=(current.message_event_id,),
        )
        del result
        return self.status(current.id)

    def cursors(self) -> dict[tuple[str, str], CursorView]:
        rows: dict[tuple[str, str], CursorView] = {}
        for event in self.store.ledger.validated_events():
            if event.event_type != DELIVERY_CURSOR_ADVANCED:
                continue
            payload = event.payload
            key = (str(payload["consumer_id"]), str(payload["channel"]))
            candidate = CursorView(
                consumer_id=key[0],
                channel=key[1],
                logical_clock=str(payload["logical_clock"]),
                event_id=str(payload["event_id"]),
            )
            current = rows.get(key)
            if current is None or _clock_key(candidate.logical_clock) > _clock_key(
                current.logical_clock
            ):
                rows[key] = candidate
        return rows

    def advance_cursor(
        self,
        *,
        identity: PrincipalIdentity,
        channel: str,
        logical_clock: str,
        event_id: str,
        idempotency_key: str,
    ) -> CursorView:
        key = (identity.principal_id, channel)
        current = self.cursors().get(key)
        if current is not None and _clock_key(logical_clock) <= _clock_key(
            current.logical_clock
        ):
            if current.logical_clock == logical_clock and current.event_id == event_id:
                return current
            raise _invalid("delivery cursor cannot regress")
        self._commit(
            identity,
            event_type=DELIVERY_CURSOR_ADVANCED,
            target_id=identity.principal_id,
            project="_global",
            subject_uri=f"memo://delivery/cursor/{identity.principal_id}/{channel}",
            payload={
                "consumer_id": identity.principal_id,
                "channel": channel,
                "logical_clock": logical_clock,
                "event_id": event_id,
            },
            idempotency_key=idempotency_key,
        )
        return self.cursors()[key]

    def unread_count(self, *, identity: PrincipalIdentity, channel: str) -> int:
        cursor = self.cursors().get((identity.principal_id, channel))
        rows = [
            event
            for event in self.store.ledger.validated_events()
            if event.event_type == MESSAGE_SENT and str(event.payload["channel"]) == channel
        ]
        if cursor is None:
            return len(rows)
        seen = next((index for index, event in enumerate(rows) if event.event_id == cursor.event_id), -1)
        return len(rows) if seen < 0 else len(rows) - seen - 1


__all__ = ["CursorView", "DeliveryService", "DeliveryView", "RetryPolicy"]
