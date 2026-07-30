"""Transactional, rebuildable SQLite projections for operational ledger v2."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterable, Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from memo.contracts import MEMO_OPERATIONAL_SCHEMA
from memo.errors import OperationalError, OperationalErrorCode
from memo.operation_view_schema import connect_operational_db, ensure_operational_schema
from memo.operational_event import (
    OperationalEventV2,
    canonical_json_bytes,
    operational_wire_dict,
)
from memo.operational_event_types import (
    ATTENTION_ACKNOWLEDGED,
    ATTENTION_ADDED,
    CONFLICT_OPENED,
    CONFLICT_RESOLVED,
    DURABLE_PROMOTION_COMPLETED,
    DURABLE_PROMOTION_REJECTED,
    DURABLE_PROMOTION_REQUESTED,
    DURABLE_PROMOTION_RETRY_SCHEDULED,
    FOCUS_CLEARED,
    FOCUS_SET,
    HANDOFF_CONSUMED,
    HANDOFF_CREATED,
    OUTCOME_RECORDED,
    SESSION_CHECKPOINTED,
    SESSION_STATUS_CHANGED,
)

if TYPE_CHECKING:
    from memo.durable_outbox import FrozenPromotionIntent, OutboxRunReport


@dataclass(frozen=True)
class ApplyReport:
    applied: int
    duplicates: int
    quarantined: int
    state_sha256: str = ""


@dataclass(frozen=True)
class IdempotencyRecord:
    scope: str
    idempotency_key: str
    request_hash: str
    event_id: str
    result: Mapping[str, object]


EventReducer = Callable[[sqlite3.Connection, OperationalEventV2], Mapping[str, object]]

_UPSERT_PAYLOAD_SQL = {
    "conflicts": """
        INSERT INTO conflicts(id, row_json, updated_event_id)
        VALUES(?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          row_json = excluded.row_json,
          updated_event_id = excluded.updated_event_id
    """,
    "outcomes": """
        INSERT INTO outcomes(task_id, row_json, updated_event_id)
        VALUES(?, ?, ?)
        ON CONFLICT(task_id) DO UPDATE SET
          row_json = excluded.row_json,
          updated_event_id = excluded.updated_event_id
    """,
}
_UPSERT_PROJECT_PAYLOAD_SQL = {
    "attention": """
        INSERT INTO attention(id, project, row_json, updated_event_id)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          project = excluded.project,
          row_json = excluded.row_json,
          updated_event_id = excluded.updated_event_id
    """,
    "handoffs": """
        INSERT INTO handoffs(id, project, row_json, updated_event_id)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          project = excluded.project,
          row_json = excluded.row_json,
          updated_event_id = excluded.updated_event_id
    """,
}
_RESET_STATEMENTS = (
    "DELETE FROM idempotency",
    "DELETE FROM origin_cursors",
    "DELETE FROM applied_events",
    "DELETE FROM focus",
    "DELETE FROM handoffs",
    "DELETE FROM attention",
    "DELETE FROM conflicts",
    "DELETE FROM outcomes",
    "DELETE FROM sessions",
    "DELETE FROM session_local_artifacts",
    "DELETE FROM durable_outbox",
    "DELETE FROM quarantined_events",
)


def _failure(code: OperationalErrorCode, message: str) -> OperationalError:
    return OperationalError(code, message, retryable=False)


def _json(value: object) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _canonical_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _failure(
            OperationalErrorCode.INVALID_EVENT,
            f"operational event timestamp is invalid: {value!r}",
        ) from exc
    if parsed.tzinfo is None:
        raise _failure(
            OperationalErrorCode.INVALID_EVENT,
            f"operational event timestamp has no timezone: {value!r}",
        )
    return (
        parsed.astimezone(UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _payload(event: OperationalEventV2) -> dict[str, object]:
    if not isinstance(event.payload, Mapping) or not all(
        isinstance(key, str) for key in event.payload
    ):
        raise _failure(
            OperationalErrorCode.INVALID_EVENT,
            f"event payload is not a string-keyed mapping: {event.event_id}",
        )
    return dict(event.payload)


def _read_row(
    connection: sqlite3.Connection,
    query: str,
    key: str,
    *,
    description: str,
) -> dict[str, object] | None:
    row = connection.execute(query, (key,)).fetchone()
    if row is None:
        return None
    value = json.loads(row["row_json"])
    if not isinstance(value, dict):
        raise _failure(
            OperationalErrorCode.STORAGE_UNAVAILABLE,
            f"derived operational row is invalid: {description}/{key}",
        )
    return value


def _focus_set(
    connection: sqlite3.Connection,
    event: OperationalEventV2,
) -> Mapping[str, object]:
    payload = _payload(event)
    project = str(payload["project"])
    connection.execute(
        """
        INSERT INTO focus(project, row_json, updated_event_id)
        VALUES(?, ?, ?)
        ON CONFLICT(project) DO UPDATE SET
          row_json = excluded.row_json,
          updated_event_id = excluded.updated_event_id
        """,
        (project, _json(payload), event.event_id),
    )
    return payload


def _focus_cleared(
    connection: sqlite3.Connection,
    event: OperationalEventV2,
) -> Mapping[str, object]:
    project = str(_payload(event)["project"])
    existed = connection.execute(
        "SELECT 1 FROM focus WHERE project = ?",
        (project,),
    ).fetchone()
    connection.execute("DELETE FROM focus WHERE project = ?", (project,))
    return {"project": project, "cleared": existed is not None}


def _upsert_payload(
    connection: sqlite3.Connection,
    event: OperationalEventV2,
    *,
    table: str,
    key: str,
    project: str | None = None,
) -> Mapping[str, object]:
    payload = _payload(event)
    if project is None:
        statement = _UPSERT_PAYLOAD_SQL.get(table)
        parameters: tuple[object, ...] = (key, _json(payload), event.event_id)
    else:
        statement = _UPSERT_PROJECT_PAYLOAD_SQL.get(table)
        parameters = (
            key,
            project,
            _json(payload),
            event.event_id,
        )
    if statement is None:
        raise AssertionError(f"unsupported operational projection table: {table}")
    connection.execute(statement, parameters)
    return payload


def _handoff_created(
    connection: sqlite3.Connection,
    event: OperationalEventV2,
) -> Mapping[str, object]:
    payload = _payload(event)
    return _upsert_payload(
        connection,
        event,
        table="handoffs",
        key=str(payload["id"]),
        project=str(payload["project"]),
    )


def _handoff_consumed(
    connection: sqlite3.Connection,
    event: OperationalEventV2,
) -> Mapping[str, object]:
    payload = _payload(event)
    key = str(payload["id"])
    row = _read_row(
        connection,
        "SELECT row_json FROM handoffs WHERE id = ?",
        key,
        description="handoffs",
    )
    if row is None:
        return {"id": key, "consumed": False}
    row["consumed_at"] = payload["consumed_at"]
    connection.execute(
        "UPDATE handoffs SET row_json = ?, updated_event_id = ? WHERE id = ?",
        (_json(row), event.event_id, key),
    )
    return row


def _attention_added(
    connection: sqlite3.Connection,
    event: OperationalEventV2,
) -> Mapping[str, object]:
    payload = _payload(event)
    return _upsert_payload(
        connection,
        event,
        table="attention",
        key=str(payload["id"]),
        project=str(payload["project"]),
    )


def _attention_acknowledged(
    connection: sqlite3.Connection,
    event: OperationalEventV2,
) -> Mapping[str, object]:
    payload = _payload(event)
    key = str(payload["id"])
    row = _read_row(
        connection,
        "SELECT row_json FROM attention WHERE id = ?",
        key,
        description="attention",
    )
    if row is None:
        return {"id": key, "acknowledged": False}
    row["acknowledged_at"] = payload["acknowledged_at"]
    connection.execute(
        "UPDATE attention SET row_json = ?, updated_event_id = ? WHERE id = ?",
        (_json(row), event.event_id, key),
    )
    return row


def _conflict_opened(
    connection: sqlite3.Connection,
    event: OperationalEventV2,
) -> Mapping[str, object]:
    payload = _payload(event)
    return _upsert_payload(
        connection,
        event,
        table="conflicts",
        key=str(payload["id"]),
    )


def _conflict_resolved(
    connection: sqlite3.Connection,
    event: OperationalEventV2,
) -> Mapping[str, object]:
    payload = _payload(event)
    key = str(payload["id"])
    row = _read_row(
        connection,
        "SELECT row_json FROM conflicts WHERE id = ?",
        key,
        description="conflicts",
    )
    if row is None:
        return {"id": key, "resolved": False}
    row.update(
        {
            "lifecycle_state": "resolved",
            "resolved_at": payload["resolved_at"],
            "resolution": payload["resolution"],
        }
    )
    connection.execute(
        "UPDATE conflicts SET row_json = ?, updated_event_id = ? WHERE id = ?",
        (_json(row), event.event_id, key),
    )
    return row


def _outcome_recorded(
    connection: sqlite3.Connection,
    event: OperationalEventV2,
) -> Mapping[str, object]:
    payload = _payload(event)
    return _upsert_payload(
        connection,
        event,
        table="outcomes",
        key=str(payload["task_id"]),
    )


def _session_checkpointed(
    connection: sqlite3.Connection,
    event: OperationalEventV2,
) -> Mapping[str, object]:
    payload = _payload(event)
    session_id = str(payload["session_id"])
    connection.execute(
        """
        INSERT INTO sessions(
          session_id, project, workspace, status, row_json, updated_event_id
        ) VALUES(?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
          project = excluded.project,
          workspace = excluded.workspace,
          status = excluded.status,
          row_json = excluded.row_json,
          updated_event_id = excluded.updated_event_id
        """,
        (
            session_id,
            str(payload.get("project") or event.project),
            str(payload.get("workspace") or event.workspace),
            str(payload["status"]),
            _json(payload),
            event.event_id,
        ),
    )
    return payload


def _session_status_changed(
    connection: sqlite3.Connection,
    event: OperationalEventV2,
) -> Mapping[str, object]:
    payload = _payload(event)
    session_id = str(payload["session_id"])
    row = _read_row(
        connection,
        "SELECT row_json FROM sessions WHERE session_id = ?",
        session_id,
        description="sessions",
    )
    if row is None:
        return {"session_id": session_id, "changed": False}
    row["status"] = payload["status"]
    connection.execute(
        """
        UPDATE sessions
        SET status = ?, row_json = ?, updated_event_id = ?
        WHERE session_id = ?
        """,
        (str(payload["status"]), _json(row), event.event_id, session_id),
    )
    return row


def _promotion_row(
    connection: sqlite3.Connection,
    promotion_id: str,
) -> dict[str, object] | None:
    return _read_row(
        connection,
        "SELECT row_json FROM durable_outbox WHERE promotion_id = ?",
        promotion_id,
        description="durable_outbox",
    )


def _assert_promotion_binding(
    row: Mapping[str, object],
    payload: Mapping[str, object],
) -> None:
    if (
        row.get("operation_key") != payload.get("operation_key")
        or row.get("request_hash") != payload.get("request_hash")
    ):
        raise _failure(
            OperationalErrorCode.IDEMPOTENCY_CONFLICT,
            "durable promotion event identifies a different request",
        )


def _assert_promotion_active(
    row: Mapping[str, object],
    promotion_id: str,
) -> None:
    if row.get("status") in {"completed", "rejected"}:
        raise _failure(
            OperationalErrorCode.INVALID_EVENT,
            f"durable promotion is already terminal: {promotion_id}",
        )


def _promotion_requested(
    connection: sqlite3.Connection,
    event: OperationalEventV2,
) -> Mapping[str, object]:
    payload = _payload(event)
    promotion_id = str(payload["promotion_id"])
    created_at = _canonical_timestamp(str(payload["created_at"]))
    row = {
        **payload,
        "created_at": created_at,
        "status": "pending",
        "attempts": 0,
        "retry_at": "",
        "memory_id": "",
        "failure_class": "",
        "reason": "",
    }
    existing = _promotion_row(connection, promotion_id)
    if existing is not None:
        immutable_fields = (
            "promotion_id",
            "idempotency_key",
            "operation_key",
            "request_hash",
            "save_kwargs",
            "source_event_ids",
            "created_at",
        )
        if all(existing.get(field) == row.get(field) for field in immutable_fields):
            return existing
        raise _failure(
            OperationalErrorCode.IDEMPOTENCY_CONFLICT,
            f"durable promotion request identity conflict: {promotion_id}",
        )
    operation_collision = connection.execute(
        """
        SELECT promotion_id FROM durable_outbox
        WHERE operation_key = ? AND promotion_id != ?
        """,
        (str(payload["operation_key"]), promotion_id),
    ).fetchone()
    if operation_collision is not None:
        raise _failure(
            OperationalErrorCode.IDEMPOTENCY_CONFLICT,
            "durable promotion operation key collision",
        )
    connection.execute(
        """
        INSERT INTO durable_outbox(
          promotion_id, operation_key, status, attempt_count, retry_at,
          memory_id, row_json, updated_event_id
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            promotion_id,
            str(payload["operation_key"]),
            "pending",
            0,
            "",
            "",
            _json(row),
            event.event_id,
        ),
    )
    return row


def _promotion_retry_scheduled(
    connection: sqlite3.Connection,
    event: OperationalEventV2,
) -> Mapping[str, object]:
    from memo.durable_outbox import deterministic_retry_at

    payload = _payload(event)
    promotion_id = str(payload["promotion_id"])
    row = _promotion_row(connection, promotion_id)
    if row is None:
        raise _failure(
            OperationalErrorCode.INVALID_EVENT,
            f"durable promotion retry has no intent: {promotion_id}",
        )
    _assert_promotion_binding(row, payload)
    _assert_promotion_active(row, promotion_id)
    attempts = row.get("attempts")
    if isinstance(attempts, bool) or not isinstance(attempts, int):
        raise _failure(
            OperationalErrorCode.STORAGE_UNAVAILABLE,
            f"durable promotion attempts are invalid: {promotion_id}",
        )
    requested_attempt = payload["attempt_number"]
    if requested_attempt != attempts + 1:
        raise _failure(
            OperationalErrorCode.INVALID_EVENT,
            f"durable promotion retry attempt is not monotonic: {promotion_id}",
        )
    retry_at = _canonical_timestamp(str(payload["retry_at"]))
    expected_retry_at = deterministic_retry_at(str(row["created_at"]), requested_attempt)
    if retry_at != expected_retry_at:
        raise _failure(
            OperationalErrorCode.INVALID_EVENT,
            f"durable promotion retry timing is not deterministic: {promotion_id}",
        )
    row.update(
        {
            "status": "retry_scheduled",
            "attempts": requested_attempt,
            "retry_at": retry_at,
            "failure_class": payload["failure_class"],
        }
    )
    connection.execute(
        """
        UPDATE durable_outbox
        SET status = 'retry_scheduled', attempt_count = ?, retry_at = ?,
            row_json = ?, updated_event_id = ?
        WHERE promotion_id = ?
        """,
        (requested_attempt, retry_at, _json(row), event.event_id, promotion_id),
    )
    return row


def _promotion_completed(
    connection: sqlite3.Connection,
    event: OperationalEventV2,
) -> Mapping[str, object]:
    payload = _payload(event)
    promotion_id = str(payload["promotion_id"])
    row = _promotion_row(connection, promotion_id)
    if row is None:
        raise _failure(
            OperationalErrorCode.INVALID_EVENT,
            f"durable promotion completion has no intent: {promotion_id}",
        )
    _assert_promotion_binding(row, payload)
    _assert_promotion_active(row, promotion_id)
    row.update({"status": "completed", "memory_id": payload["memory_id"], "retry_at": ""})
    connection.execute(
        """
        UPDATE durable_outbox
        SET status = 'completed', memory_id = ?, row_json = ?, updated_event_id = ?
        WHERE promotion_id = ?
        """,
        (str(payload["memory_id"]), _json(row), event.event_id, promotion_id),
    )
    return row


def _promotion_rejected(
    connection: sqlite3.Connection,
    event: OperationalEventV2,
) -> Mapping[str, object]:
    payload = _payload(event)
    promotion_id = str(payload["promotion_id"])
    row = _promotion_row(connection, promotion_id)
    if row is None:
        raise _failure(
            OperationalErrorCode.INVALID_EVENT,
            f"durable promotion rejection has no intent: {promotion_id}",
        )
    _assert_promotion_binding(row, payload)
    _assert_promotion_active(row, promotion_id)
    row.update(
        {
            "status": "rejected",
            "failure_class": payload["failure_class"],
            "reason": payload["reason"],
            "retry_at": "",
        }
    )
    connection.execute(
        """
        UPDATE durable_outbox
        SET status = 'rejected', retry_at = '', row_json = ?, updated_event_id = ?
        WHERE promotion_id = ?
        """,
        (_json(row), event.event_id, promotion_id),
    )
    return row


EVENT_REDUCERS: dict[str, EventReducer] = {
    FOCUS_SET: _focus_set,
    FOCUS_CLEARED: _focus_cleared,
    HANDOFF_CREATED: _handoff_created,
    HANDOFF_CONSUMED: _handoff_consumed,
    ATTENTION_ADDED: _attention_added,
    ATTENTION_ACKNOWLEDGED: _attention_acknowledged,
    CONFLICT_OPENED: _conflict_opened,
    CONFLICT_RESOLVED: _conflict_resolved,
    OUTCOME_RECORDED: _outcome_recorded,
    SESSION_CHECKPOINTED: _session_checkpointed,
    SESSION_STATUS_CHANGED: _session_status_changed,
    DURABLE_PROMOTION_REQUESTED: _promotion_requested,
    DURABLE_PROMOTION_RETRY_SCHEDULED: _promotion_retry_scheduled,
    DURABLE_PROMOTION_COMPLETED: _promotion_completed,
    DURABLE_PROMOTION_REJECTED: _promotion_rejected,
}


class OperationalViewStore:
    """One transactional projection database derived from verified v2 events."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        with closing(connect_operational_db(self.path)) as connection:
            ensure_operational_schema(connection)

    @staticmethod
    def _result(event: OperationalEventV2, value: Mapping[str, object]) -> dict[str, object]:
        return {
            "event_id": event.event_id,
            "event_hash": event.event_hash,
            "event_type": event.event_type,
            "value": dict(value),
        }

    def _apply_events(
        self,
        connection: sqlite3.Connection,
        events: Iterable[OperationalEventV2],
    ) -> ApplyReport:
        applied = 0
        duplicates = 0
        quarantined = 0
        blocked_origins: set[str] = set()
        for event in events:
            if not isinstance(event, OperationalEventV2):
                raise _failure(
                    OperationalErrorCode.INVALID_EVENT,
                    "OperationalEventV2 is required for view application",
                )
            existing_event = connection.execute(
                """
                SELECT origin_device, origin_sequence, event_hash
                FROM applied_events WHERE event_id = ?
                """,
                (event.event_id,),
            ).fetchone()
            if existing_event is not None:
                if (
                    existing_event["origin_device"] != event.origin_device
                    or existing_event["origin_sequence"] != event.origin_sequence
                    or existing_event["event_hash"] != event.event_hash
                ):
                    raise _failure(
                        OperationalErrorCode.ANCHOR_CONFLICT,
                        f"applied event identity collision: {event.event_id}",
                    )
                connection.execute(
                    "DELETE FROM quarantined_events WHERE event_id = ?",
                    (event.event_id,),
                )
                duplicates += 1
                continue
            origin_event = connection.execute(
                """
                SELECT event_id, event_hash FROM applied_events
                WHERE origin_device = ? AND origin_sequence = ?
                """,
                (event.origin_device, event.origin_sequence),
            ).fetchone()
            if origin_event is not None:
                raise _failure(
                    OperationalErrorCode.ANCHOR_CONFLICT,
                    (
                        "origin sequence collision: "
                        f"{event.origin_device}/{event.origin_sequence}"
                    ),
                )
            reducer = EVENT_REDUCERS.get(event.event_type)
            if reducer is None or event.origin_device in blocked_origins:
                reason = (
                    "unsupported operational view reducer"
                    if reducer is None
                    else "origin blocked by an earlier unsupported event"
                )
                connection.execute(
                    """
                    INSERT INTO quarantined_events(
                      event_id, event_type, reason, event_json, quarantined_at
                    ) VALUES(?, ?, ?, ?, ?)
                    ON CONFLICT(event_id) DO UPDATE SET
                      event_type = excluded.event_type,
                      reason = excluded.reason,
                      event_json = excluded.event_json,
                      quarantined_at = excluded.quarantined_at
                    """,
                    (
                        event.event_id,
                        event.event_type,
                        reason,
                        _json(operational_wire_dict(event)),
                        event.created_at,
                    ),
                )
                blocked_origins.add(event.origin_device)
                quarantined += 1
                continue
            cursor = connection.execute(
                """
                SELECT origin_sequence, event_hash FROM origin_cursors
                WHERE origin_device = ?
                """,
                (event.origin_device,),
            ).fetchone()
            if cursor is not None and event.origin_sequence <= cursor["origin_sequence"]:
                raise _failure(
                    OperationalErrorCode.SEQUENCE_GAP,
                    (
                        "event is behind its derived origin cursor: "
                        f"{event.origin_device}/{event.origin_sequence}"
                    ),
                )
            value = reducer(connection, event)
            result = self._result(event, value)
            connection.execute(
                """
                INSERT INTO applied_events(
                  event_id, origin_device, origin_sequence, event_hash, applied_at
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.origin_device,
                    event.origin_sequence,
                    event.event_hash,
                    _canonical_timestamp(event.created_at),
                ),
            )
            connection.execute(
                """
                INSERT INTO origin_cursors(origin_device, origin_sequence, event_hash)
                VALUES(?, ?, ?)
                ON CONFLICT(origin_device) DO UPDATE SET
                  origin_sequence = excluded.origin_sequence,
                  event_hash = excluded.event_hash
                """,
                (event.origin_device, event.origin_sequence, event.event_hash),
            )
            connection.execute(
                "DELETE FROM quarantined_events WHERE event_id = ?",
                (event.event_id,),
            )
            if event.idempotency_key:
                existing_key = connection.execute(
                    """
                    SELECT request_hash, event_id FROM idempotency
                    WHERE scope = ? AND idempotency_key = ?
                    """,
                    (event.project, event.idempotency_key),
                ).fetchone()
                if existing_key is not None:
                    if (
                        existing_key["request_hash"] != event.content_hash
                        or existing_key["event_id"] != event.event_id
                    ):
                        raise _failure(
                            OperationalErrorCode.IDEMPOTENCY_CONFLICT,
                            (
                                "idempotency key identifies a different request: "
                                f"{event.project}/{event.idempotency_key}"
                            ),
                        )
                else:
                    connection.execute(
                        """
                        INSERT INTO idempotency(
                          scope, idempotency_key, request_hash, event_id, result_json
                        ) VALUES(?, ?, ?, ?, ?)
                        """,
                        (
                            event.project,
                            event.idempotency_key,
                            event.content_hash,
                            event.event_id,
                            _json(result),
                        ),
                    )
            applied += 1
        return ApplyReport(
            applied=applied,
            duplicates=duplicates,
            quarantined=quarantined,
        )

    def apply_events(self, events: Iterable[OperationalEventV2]) -> ApplyReport:
        with closing(connect_operational_db(self.path)) as connection:
            ensure_operational_schema(connection)
            connection.execute("BEGIN IMMEDIATE")
            try:
                report = self._apply_events(connection, events)
            except BaseException:
                connection.rollback()
                raise
            connection.commit()
            return report

    def catch_up(self, ledger: object) -> ApplyReport:
        validated_events = getattr(ledger, "validated_events", None)
        if not callable(validated_events):
            raise TypeError("operational ledger must expose validated_events()")
        return self.apply_events(validated_events())

    @staticmethod
    def supports(event_type: str) -> bool:
        return event_type in EVENT_REDUCERS

    def rebuild(self, events: Iterable[OperationalEventV2]) -> ApplyReport:
        materialized = tuple(events)
        with closing(connect_operational_db(self.path)) as connection:
            ensure_operational_schema(connection)
            connection.execute("BEGIN IMMEDIATE")
            try:
                for statement in _RESET_STATEMENTS:
                    connection.execute(statement)
                report = self._apply_events(connection, materialized)
                state = self._state(connection)
            except BaseException:
                connection.rollback()
                raise
            connection.commit()
        return ApplyReport(
            applied=report.applied,
            duplicates=report.duplicates,
            quarantined=report.quarantined,
            state_sha256=hashlib.sha256(canonical_json_bytes(state)).hexdigest(),
        )

    @staticmethod
    def _rows(
        connection: sqlite3.Connection,
        query: str,
        key_field: str,
        *,
        parameters: tuple[object, ...] = (),
    ) -> dict[str, object]:
        rows = connection.execute(query, parameters).fetchall()
        return {str(row[key_field]): json.loads(row["row_json"]) for row in rows}

    def _state(
        self,
        connection: sqlite3.Connection,
        *,
        project: str | None = None,
    ) -> dict[str, object]:
        cursors = connection.execute(
            "SELECT origin_device, event_hash FROM origin_cursors"
        ).fetchall()
        last = connection.execute(
            """
            SELECT event_hash FROM applied_events
            ORDER BY applied_at DESC, origin_device DESC, origin_sequence DESC
            LIMIT 1
            """
        ).fetchone()
        return {
            "schema": MEMO_OPERATIONAL_SCHEMA,
            "focus": self._rows(
                connection,
                (
                    "SELECT project, row_json FROM focus WHERE project = ?"
                    if project is not None
                    else "SELECT project, row_json FROM focus"
                ),
                "project",
                parameters=(project,) if project is not None else (),
            ),
            "handoffs": self._rows(
                connection,
                (
                    "SELECT id, row_json FROM handoffs WHERE project = ?"
                    if project is not None
                    else "SELECT id, row_json FROM handoffs"
                ),
                "id",
                parameters=(project,) if project is not None else (),
            ),
            "attention": self._rows(
                connection,
                (
                    "SELECT id, row_json FROM attention WHERE project = ?"
                    if project is not None
                    else "SELECT id, row_json FROM attention"
                ),
                "id",
                parameters=(project,) if project is not None else (),
            ),
            "conflicts": self._rows(
                connection,
                "SELECT id, row_json FROM conflicts",
                "id",
            ),
            "outcomes": self._rows(
                connection,
                "SELECT task_id, row_json FROM outcomes",
                "task_id",
            ),
            "sessions": self._rows(
                connection,
                "SELECT session_id, row_json FROM sessions",
                "session_id",
            ),
            "durable_outbox": self._rows(
                connection,
                "SELECT promotion_id, row_json FROM durable_outbox",
                "promotion_id",
            ),
            "last_event_hash": str(last["event_hash"]) if last is not None else "",
            "journal_heads": {
                str(row["origin_device"]): str(row["event_hash"]) for row in cursors
            },
        }

    def state(self, *, project: str | None = None) -> dict[str, object]:
        with closing(connect_operational_db(self.path)) as connection:
            ensure_operational_schema(connection)
            return self._state(connection, project=project)

    def outbox_intent(self, promotion_id: str) -> FrozenPromotionIntent | None:
        from memo.durable_outbox import frozen_intent_from_row

        body = self.outbox_status(promotion_id)
        return frozen_intent_from_row(body) if body is not None else None

    def outbox_status(self, promotion_id: str) -> dict[str, object] | None:
        with closing(connect_operational_db(self.path)) as connection:
            ensure_operational_schema(connection)
            row = connection.execute(
                "SELECT row_json FROM durable_outbox WHERE promotion_id = ?",
                (promotion_id,),
            ).fetchone()
        if row is None:
            return None
        body = json.loads(row["row_json"])
        if not isinstance(body, dict):
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"stored durable promotion row is invalid: {promotion_id}",
            )
        return body

    def pending_outbox(
        self,
        *,
        limit: int,
        now: str | None = None,
    ) -> list[FrozenPromotionIntent]:
        from memo.durable_outbox import frozen_intent_from_row
        from memo.util import utc_now_iso

        if isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be >= 1")
        canonical_now = _canonical_timestamp(now or utc_now_iso())
        with closing(connect_operational_db(self.path)) as connection:
            ensure_operational_schema(connection)
            rows = connection.execute(
                """
                SELECT row_json FROM durable_outbox
                WHERE status = 'pending'
                   OR (status = 'retry_scheduled' AND retry_at <= ?)
                ORDER BY
                  CASE
                    WHEN retry_at = '' THEN json_extract(row_json, '$.created_at')
                    ELSE retry_at
                  END,
                  json_extract(row_json, '$.created_at'),
                  promotion_id
                LIMIT ?
                """,
                (canonical_now, limit),
            ).fetchall()
        intents: list[FrozenPromotionIntent] = []
        for row in rows:
            body = json.loads(row["row_json"])
            if not isinstance(body, dict):
                raise _failure(
                    OperationalErrorCode.STORAGE_UNAVAILABLE,
                    "stored durable promotion row is invalid",
                )
            intents.append(frozen_intent_from_row(body))
        return intents

    def outbox_report(self) -> OutboxRunReport:
        from memo.durable_outbox import OutboxRunReport

        with closing(connect_operational_db(self.path)) as connection:
            ensure_operational_schema(connection)
            rows = connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM durable_outbox
                GROUP BY status
                """
            ).fetchall()
        counts = {str(row["status"]): int(row["count"]) for row in rows}
        return OutboxRunReport(
            examined=sum(counts.values()),
            completed=counts.get("completed", 0),
            retried=counts.get("retry_scheduled", 0),
            quarantined=counts.get("rejected", 0),
            pending=counts.get("pending", 0) + counts.get("retry_scheduled", 0),
        )

    def idempotency(
        self,
        scope: str,
        idempotency_key: str,
    ) -> IdempotencyRecord | None:
        with closing(connect_operational_db(self.path)) as connection:
            ensure_operational_schema(connection)
            row = connection.execute(
                """
                SELECT request_hash, event_id, result_json FROM idempotency
                WHERE scope = ? AND idempotency_key = ?
                """,
                (scope, idempotency_key),
            ).fetchone()
        if row is None:
            return None
        result = json.loads(row["result_json"])
        if not isinstance(result, dict):
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                "stored operational idempotency result is invalid",
            )
        return IdempotencyRecord(
            scope=scope,
            idempotency_key=idempotency_key,
            request_hash=str(row["request_hash"]),
            event_id=str(row["event_id"]),
            result=result,
        )


__all__ = [
    "EVENT_REDUCERS",
    "ApplyReport",
    "IdempotencyRecord",
    "OperationalViewStore",
]
