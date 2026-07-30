from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

import memo.operation_views as operation_views
from memo.errors import OperationalError
from memo.identity import PrincipalIdentity
from memo.operation_view_schema import connect_operational_db, ensure_operational_schema
from memo.operation_views import OperationalViewStore
from memo.operational_event import OperationalEventV2
from memo.operational_event_types import (
    ATTENTION_ACKNOWLEDGED,
    ATTENTION_ADDED,
    CONFLICT_OPENED,
    CONFLICT_RESOLVED,
    COORDINATION_CREATED,
    DURABLE_PROMOTION_COMPLETED,
    FOCUS_CLEARED,
    FOCUS_SET,
    HANDOFF_CONSUMED,
    HANDOFF_CREATED,
    OUTCOME_RECORDED,
    SESSION_CHECKPOINTED,
    SESSION_STATUS_CHANGED,
)


def _identity() -> PrincipalIdentity:
    return PrincipalIdentity(
        principal_id="device-a:session-a",
        actor_id="agent-a",
        kind="agent",
        device_id="device-a",
        session_id="session-a",
        source_client="codex",
    )


def _event(
    sequence: int,
    event_type: str,
    payload: dict[str, object],
    *,
    origin: str = "device-a",
    idempotency_key: str | None = None,
) -> OperationalEventV2:
    return OperationalEventV2(
        schema="memo.operational_event.v2",
        schema_version=2,
        event_id=f"event-{origin}-{sequence}",
        event_type=event_type,
        actor=_identity(),
        target_id=None,
        project="demo",
        workspace="/tmp/demo",
        origin_device=origin,
        origin_sequence=sequence,
        logical_clock=f"0:{sequence}",
        authority_epoch=0,
        control_oid="control-0",
        created_at=f"2026-07-30T12:00:{sequence:02d}Z",
        expires_at=None,
        visibility="owner",
        idempotency_key=idempotency_key or f"idem-{origin}-{sequence}",
        caused_by=(),
        subject_uri=f"memo://event/{origin}/{sequence}",
        trace_id=f"trace-{sequence}",
        payload=payload,
        content_hash=f"{sequence:064x}",
        previous_hash="" if sequence == 1 else f"{sequence - 1:064x}",
        event_hash=f"{sequence + 100:064x}",
        source_proof=None,
        roster_version=1,
        key_id="key-1",
        signature="signature",
    )


def test_schema_creation_is_idempotent_and_rejects_unknown_version(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operational.db"
    with connect_operational_db(path) as connection:
        ensure_operational_schema(connection)
        ensure_operational_schema(connection)
        connection.execute(
            "UPDATE view_meta SET value = '999' WHERE key = 'schema_version'"
        )
        connection.commit()
        with pytest.raises(OperationalError, match="schema"):
            ensure_operational_schema(connection)


def test_apply_is_one_transaction_and_exact_replay_is_noop(tmp_path: Path) -> None:
    store = OperationalViewStore(tmp_path / "operational.db")
    event = _event(
        1,
        FOCUS_SET,
        {
            "id": "focus-1",
            "project": "demo",
            "summary": "Ship native views",
            "updated_at": "2026-07-30T12:00:01Z",
            "actor_id": "agent-a",
            "metadata": {},
        },
    )

    first = store.apply_events((event,))
    second = store.apply_events((event,))

    assert first.applied == 1
    assert second.applied == 0
    assert second.duplicates == 1
    assert store.state()["focus"]["demo"]["summary"] == "Ship native views"


def test_reducer_failure_rolls_back_event_cursor_domain_and_idempotency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = OperationalViewStore(tmp_path / "operational.db")
    event = _event(1, FOCUS_SET, {"project": "demo", "summary": "boom"})

    def fail(_connection: sqlite3.Connection, _event: OperationalEventV2) -> object:
        raise RuntimeError("reducer exploded")

    monkeypatch.setitem(operation_views.EVENT_REDUCERS, FOCUS_SET, fail)
    with pytest.raises(RuntimeError, match="reducer exploded"):
        store.apply_events((event,))

    with connect_operational_db(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM applied_events").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM origin_cursors").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM focus").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM idempotency").fetchone()[0] == 0


def test_origin_sequence_collision_with_different_event_fails_closed(
    tmp_path: Path,
) -> None:
    store = OperationalViewStore(tmp_path / "operational.db")
    first = _event(1, FOCUS_SET, {"project": "demo", "summary": "first"})
    changed = replace(
        first,
        event_id="event-changed",
        event_hash="f" * 64,
        payload={"project": "demo", "summary": "changed"},
    )
    store.apply_events((first,))

    with pytest.raises(OperationalError, match=r"origin|sequence|collision"):
        store.apply_events((changed,))


def test_core_reducers_preserve_public_state_shape(tmp_path: Path) -> None:
    store = OperationalViewStore(tmp_path / "operational.db")
    events = (
        _event(1, FOCUS_SET, {"project": "demo", "summary": "focus"}),
        _event(
            2,
            HANDOFF_CREATED,
            {
                "id": "handoff-1",
                "project": "demo",
                "summary": "continue",
                "from_actor": "agent-a",
                "to_actor": "agent-b",
                "created_at": "2026-07-30T12:00:02Z",
                "consumed_at": "",
                "metadata": {},
            },
        ),
        _event(
            3,
            HANDOFF_CONSUMED,
            {"id": "handoff-1", "consumed_at": "2026-07-30T12:00:03Z"},
        ),
        _event(
            4,
            ATTENTION_ADDED,
            {
                "id": "attention-1",
                "project": "demo",
                "summary": "review",
                "severity": "high",
                "created_at": "2026-07-30T12:00:04Z",
                "acknowledged_at": "",
                "metadata": {},
            },
        ),
        _event(
            5,
            ATTENTION_ACKNOWLEDGED,
            {"id": "attention-1", "acknowledged_at": "2026-07-30T12:00:05Z"},
        ),
        _event(
            6,
            CONFLICT_OPENED,
            {
                "id": "conflict-1",
                "topic": "authority",
                "summary": "choose one",
                "freeze_write": True,
                "evidence_uris": [],
                "lifecycle_state": "detected",
                "created_at": "2026-07-30T12:00:06Z",
                "resolved_at": "",
                "resolution": "",
                "metadata": {},
            },
        ),
        _event(
            7,
            CONFLICT_RESOLVED,
            {
                "id": "conflict-1",
                "resolution": "memo",
                "resolved_at": "2026-07-30T12:00:07Z",
            },
        ),
        _event(
            8,
            OUTCOME_RECORDED,
            {
                "task_id": "task-1",
                "status": "success",
                "memory_ids": ["memory-1"],
                "artifacts": [],
                "environment": {},
                "actor_id": "agent-a",
                "idempotency_key": "outcome-1",
                "recorded_at": "2026-07-30T12:00:08Z",
            },
        ),
        _event(
            9,
            SESSION_CHECKPOINTED,
            {
                "session_id": "session-a",
                "principal_id": "device-a:session-a",
                "project": "demo",
                "workspace": "/tmp/demo",
                "status": "active",
                "branch": "main",
                "head": "abc",
                "summary": "working",
                "checkpointed_at": "2026-07-30T12:00:09Z",
                "source_event_id": "source-1",
            },
        ),
        _event(
            10,
            SESSION_STATUS_CHANGED,
            {"session_id": "session-a", "status": "recoverable"},
        ),
    )

    store.apply_events(events)
    state = store.state()

    expected_handoff = {
        **events[1].payload,
        "consumed_at": "2026-07-30T12:00:03Z",
    }
    expected_attention = {
        **events[3].payload,
        "acknowledged_at": "2026-07-30T12:00:05Z",
    }
    expected_conflict = {
        **events[5].payload,
        "lifecycle_state": "resolved",
        "resolved_at": "2026-07-30T12:00:07Z",
        "resolution": "memo",
    }
    expected_session = {
        **events[8].payload,
        "status": "recoverable",
    }

    assert state["focus"] == {"demo": dict(events[0].payload)}
    assert state["handoffs"] == {"handoff-1": expected_handoff}
    assert state["attention"] == {"attention-1": expected_attention}
    assert state["conflicts"] == {"conflict-1": expected_conflict}
    assert state["outcomes"] == {"task-1": dict(events[7].payload)}
    assert state["sessions"] == {"session-a": expected_session}


def test_clear_and_rebuild_are_deterministic(tmp_path: Path) -> None:
    events = (
        _event(1, FOCUS_SET, {"project": "demo", "summary": "focus"}),
        _event(2, FOCUS_CLEARED, {"project": "demo"}),
    )
    first = OperationalViewStore(tmp_path / "first.db")
    second = OperationalViewStore(tmp_path / "second.db")

    first.apply_events(events)
    first_report = first.rebuild(events)
    second_report = second.rebuild(events)

    assert first.state() == second.state()
    assert first_report.state_sha256 == second_report.state_sha256
    assert first.state()["focus"] == {}


def test_catch_up_reduces_every_validated_event_in_ledger_order(
    tmp_path: Path,
) -> None:
    events = (
        _event(1, FOCUS_SET, {"project": "demo", "summary": "first"}),
        _event(2, FOCUS_SET, {"project": "demo", "summary": "second"}),
    )

    class _ValidatedLedger:
        def __init__(self) -> None:
            self.calls = 0

        def validated_events(self) -> tuple[OperationalEventV2, ...]:
            self.calls += 1
            return events

    ledger = _ValidatedLedger()
    store = OperationalViewStore(tmp_path / "operational.db")

    report = store.catch_up(ledger)

    assert report.applied == 2
    assert ledger.calls == 1
    assert store.state()["focus"]["demo"]["summary"] == "second"


def test_global_last_event_uses_normalized_event_time_not_arrival_order(
    tmp_path: Path,
) -> None:
    store = OperationalViewStore(tmp_path / "operational.db")
    newest = replace(
        _event(
            1,
            FOCUS_SET,
            {"project": "demo", "summary": "newest"},
            origin="device-a",
        ),
        created_at="2026-07-30T10:00:00-02:00",
    )
    older_late_arrival = replace(
        _event(
            1,
            ATTENTION_ADDED,
            {
                "id": "attention-1",
                "project": "demo",
                "summary": "older",
                "severity": "low",
            },
            origin="device-b",
        ),
        created_at="2026-07-30T11:30:00Z",
    )

    store.apply_events((newest,))
    store.apply_events((older_late_arrival,))

    assert store.state()["last_event_hash"] == newest.event_hash


def test_unsupported_event_is_quarantined_without_partial_reduction(tmp_path: Path) -> None:
    store = OperationalViewStore(tmp_path / "operational.db")
    unsupported = _event(
        1,
        COORDINATION_CREATED,
        {"task_id": "task-1", "summary": "coordinate", "status": "pending"},
    )

    report = store.apply_events((unsupported,))

    assert report.applied == 0
    assert report.quarantined == 1
    with connect_operational_db(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM quarantined_events").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM applied_events").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM origin_cursors").fetchone()[0] == 0


def test_unsupported_event_blocks_later_origin_events_until_reducer_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = OperationalViewStore(tmp_path / "operational.db")
    unsupported = _event(
        1,
        COORDINATION_CREATED,
        {"task_id": "task-1", "summary": "coordinate", "status": "pending"},
    )
    later = _event(
        2,
        FOCUS_SET,
        {"project": "demo", "summary": "must wait"},
    )

    blocked = store.apply_events((unsupported, later))

    assert blocked.applied == 0
    assert blocked.quarantined == 2
    assert store.state()["focus"] == {}

    monkeypatch.setitem(
        operation_views.EVENT_REDUCERS,
        COORDINATION_CREATED,
        lambda _connection, event: dict(event.payload),
    )
    recovered = store.apply_events((unsupported, later))

    assert recovered.applied == 2
    assert store.state()["focus"]["demo"]["summary"] == "must wait"
    with connect_operational_db(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM quarantined_events").fetchone()[0] == 0


def test_durable_completion_without_intent_rolls_back(tmp_path: Path) -> None:
    store = OperationalViewStore(tmp_path / "operational.db")
    completion = _event(
        1,
        DURABLE_PROMOTION_COMPLETED,
        {"promotion_id": "promotion-1", "memory_id": "memory-1"},
    )

    with pytest.raises(OperationalError, match="no intent"):
        store.apply_events((completion,))

    with connect_operational_db(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM applied_events").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM durable_outbox").fetchone()[0] == 0
