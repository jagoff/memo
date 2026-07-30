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
    DURABLE_PROMOTION_ENQUEUED,
    FOCUS_CLEARED,
    FOCUS_SET,
    HANDOFF_CONSUMED,
    HANDOFF_CREATED,
    OUTCOME_RECORDED,
    SESSION_CHECKPOINTED,
    SESSION_STATUS_CHANGED,
)


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


def _promotion_enqueued(
    connection: sqlite3.Connection,
    event: OperationalEventV2,
) -> Mapping[str, object]:
    payload = _payload(event)
    promotion_id = str(payload["promotion_id"])
    attempt_count = payload.get("attempt_count", 0)
    if isinstance(attempt_count, bool) or not isinstance(attempt_count, int):
        raise _failure(
            OperationalErrorCode.INVALID_EVENT,
            "durable promotion attempt_count must be an integer",
        )
    row = {
        **payload,
        "status": "pending",
        "attempt_count": attempt_count,
        "retry_at": str(payload.get("retry_at") or ""),
        "memory_id": "",
    }
    connection.execute(
        """
        INSERT INTO durable_outbox(
          promotion_id, operation_key, status, attempt_count, retry_at,
          memory_id, row_json, updated_event_id
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(promotion_id) DO UPDATE SET
          operation_key = excluded.operation_key,
          status = excluded.status,
          attempt_count = excluded.attempt_count,
          retry_at = excluded.retry_at,
          memory_id = excluded.memory_id,
          row_json = excluded.row_json,
          updated_event_id = excluded.updated_event_id
        """,
        (
            promotion_id,
            str(payload["operation_key"]),
            "pending",
            row["attempt_count"],
            row["retry_at"],
            "",
            _json(row),
            event.event_id,
        ),
    )
    return row


def _promotion_completed(
    connection: sqlite3.Connection,
    event: OperationalEventV2,
) -> Mapping[str, object]:
    payload = _payload(event)
    promotion_id = str(payload["promotion_id"])
    row = connection.execute(
        "SELECT row_json FROM durable_outbox WHERE promotion_id = ?",
        (promotion_id,),
    ).fetchone()
    if row is None:
        raise _failure(
            OperationalErrorCode.INVALID_EVENT,
            f"durable promotion completion has no intent: {promotion_id}",
        )
    body: dict[str, object] = json.loads(row["row_json"])
    body.update({"status": "completed", "memory_id": payload["memory_id"]})
    connection.execute(
        """
        UPDATE durable_outbox
        SET status = 'completed', memory_id = ?, row_json = ?, updated_event_id = ?
        WHERE promotion_id = ?
        """,
        (str(payload["memory_id"]), _json(body), event.event_id, promotion_id),
    )
    return body


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
    DURABLE_PROMOTION_ENQUEUED: _promotion_enqueued,
    DURABLE_PROMOTION_COMPLETED: _promotion_completed,
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
            "last_event_hash": str(last["event_hash"]) if last is not None else "",
            "journal_heads": {
                str(row["origin_device"]): str(row["event_hash"]) for row in cursors
            },
        }

    def state(self, *, project: str | None = None) -> dict[str, object]:
        with closing(connect_operational_db(self.path)) as connection:
            ensure_operational_schema(connection)
            return self._state(connection, project=project)

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
