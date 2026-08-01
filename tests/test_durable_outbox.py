from __future__ import annotations

import hashlib
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import pytest

from memo.durable_outbox import (
    DurableOutboxAuthority,
    DurableOutboxWorker,
    canonical_save_request_hash,
    deterministic_retry_at,
    freeze_promotion_intent,
    promotion_operation_key,
)
from memo.errors import IdentityConflictError, OperationalError, OperationalErrorCode, WriteRefused
from memo.identity import PrincipalIdentity
from memo.operation_views import OperationalViewStore
from memo.operational_epoch import CommitContext
from memo.operational_event import (
    CommandResult,
    OperationalCommand,
    OperationalEventV2,
    canonical_json_bytes,
)
from memo.operational_event_types import validate_event_payload


def _identity() -> PrincipalIdentity:
    return PrincipalIdentity(
        principal_id="device-a:session-a",
        actor_id="agent-a",
        kind="agent",
        device_id="device-a",
        session_id="session-a",
        source_client="codex",
    )


def _context() -> CommitContext:
    return CommitContext(
        identity=_identity(),
        authority_epoch=0,
        control_oid="control-0",
        origin_device="device-a",
    )


class _Operational:
    def __init__(self, views: OperationalViewStore) -> None:
        self.views = views
        self.events: list[OperationalEventV2] = []
        self._commands: dict[
            tuple[str, str], tuple[str, OperationalEventV2, dict[str, object]]
        ] = {}

    def commit(
        self,
        command: OperationalCommand,
        *,
        context: CommitContext,
    ) -> CommandResult:
        assert command.actor == context.identity
        validate_event_payload(command.event_type, command.payload)
        request_hash = hashlib.sha256(canonical_json_bytes(asdict(command))).hexdigest()
        key = (command.project, command.idempotency_key)
        existing = self._commands.get(key)
        if existing is not None:
            if existing[0] != request_hash:
                raise OperationalError(
                    OperationalErrorCode.IDEMPOTENCY_CONFLICT,
                    "operational idempotency conflict",
                )
            return CommandResult(event=existing[1], replayed=True, result=existing[2])
        sequence = len(self.events) + 1
        event = OperationalEventV2(
            schema="memo.operational_event.v2",
            schema_version=2,
            event_id=f"event-{sequence}",
            event_type=command.event_type,
            actor=command.actor,
            target_id=command.target_id,
            project=command.project,
            workspace=command.workspace,
            origin_device=context.origin_device,
            origin_sequence=sequence,
            logical_clock=f"0:{sequence}",
            authority_epoch=context.authority_epoch,
            control_oid=context.control_oid,
            created_at=f"2026-07-30T12:00:{sequence:02d}Z",
            expires_at=command.expires_at,
            visibility=command.visibility,
            idempotency_key=command.idempotency_key,
            caused_by=command.caused_by,
            subject_uri=command.subject_uri,
            trace_id=command.trace_id,
            payload=command.payload,
            content_hash=request_hash,
            previous_hash=self.events[-1].event_hash if self.events else "",
            event_hash=f"{sequence + 100:064x}",
            source_proof=command.source_proof,
            roster_version=1,
            key_id="key-1",
            signature="signature",
        )
        self.events.append(event)
        report = self.views.apply_events((event,))
        assert report.applied == 1
        result = {
            "event_id": event.event_id,
            "event_hash": event.event_hash,
            "event_type": event.event_type,
            "value": dict(command.payload),
        }
        self._commands[key] = (request_hash, event, result)
        return CommandResult(event=event, replayed=False, result=result)


class _Memory:
    def __init__(self) -> None:
        self.calls = 0
        self.fail: str | None = None
        self.records: dict[str, tuple[str, SimpleNamespace]] = {}

    def find_by_operation_key(
        self,
        operation_key: str,
        request_hash: str,
    ) -> SimpleNamespace | None:
        existing = self.records.get(operation_key)
        if existing is None:
            return None
        if existing[0] != request_hash:
            raise IdentityConflictError(
                kind="durable_operation",
                incoming={"operation_key": operation_key, "request_hash": request_hash},
                conflicts=[{"id": existing[1].id, "request_hash": existing[0]}],
            )
        return existing[1]

    def save_operation(
        self,
        *,
        operation_key: str,
        request_hash: str,
        save_kwargs: dict[str, object],
    ) -> SimpleNamespace:
        self.calls += 1
        existing = self.find_by_operation_key(operation_key, request_hash)
        if existing is not None:
            return existing
        if self.fail == "before":
            self.fail = None
            raise RuntimeError("crash before save")
        if self.fail == "policy":
            raise WriteRefused({"conflict_id": "conflict-1", "summary": "blocked"})
        record = SimpleNamespace(
            id=f"memory-{len(self.records) + 1}",
            extra=dict(save_kwargs.get("extra") or {}),
        )
        self.records[operation_key] = (request_hash, record)
        if self.fail == "after":
            self.fail = None
            raise RuntimeError("crash after save")
        return record


class _Clock:
    def __init__(self, value: str) -> None:
        self.value = value

    def __call__(self) -> str:
        return self.value


def _worker(
    tmp_path: Any,
    *,
    clock: _Clock | None = None,
) -> tuple[DurableOutboxWorker, _Memory, _Operational, OperationalViewStore]:
    views = OperationalViewStore(tmp_path / "operational.db")
    operational = _Operational(views)
    memory = _Memory()
    authority = DurableOutboxAuthority(
        actor=_identity(),
        project="memo",
        workspace="/tmp/memo",
        visibility="owner",
        trace_id="trace-outbox",
    )
    worker = DurableOutboxWorker(
        memory=memory,
        operational=operational,
        store=views,
        authority=authority,
        context_factory=_context,
        clock=clock or _Clock("2026-07-30T12:00:10Z"),
    )
    return worker, memory, operational, views


def _save_kwargs(body: str = "Verify the journal before sync.") -> dict[str, object]:
    return {
        "content": body,
        "title": "Verify journal",
        "type_": "procedure",
        "tags": ["procedural", "outcome-backed"],
        "extra": {
            "learning": {
                "promoted_at": "2026-07-30T12:00:00Z",
                "source_memory_ids": ["memory-source"],
            }
        },
        "auto_project": False,
    }


def _enqueue(worker: DurableOutboxWorker, *, key: str = "promotion-1") -> Any:
    return worker.enqueue(
        idempotency_key=key,
        save_kwargs=_save_kwargs(),
        source_event_ids=("outcome-event-2", "outcome-event-1", "outcome-event-2"),
        created_at="2026-07-30T12:00:00Z",
    )


def test_intent_identity_and_request_are_canonical_and_deeply_immutable() -> None:
    kwargs = _save_kwargs()
    intent = freeze_promotion_intent(
        idempotency_key="  promotion-1  ",
        save_kwargs=kwargs,
        source_event_ids=("outcome-event-2", "outcome-event-1", "outcome-event-2"),
        created_at="2026-07-30T12:00:00Z",
    )

    assert intent.id == hashlib.sha256(b"promotion-1").hexdigest()
    assert intent.operation_key == promotion_operation_key("promotion-1")
    assert intent.request_hash == canonical_save_request_hash(intent.mutable_save_kwargs())
    assert intent.source_event_ids == ("outcome-event-1", "outcome-event-2")
    kwargs["content"] = "changed after freeze"
    assert intent.save_kwargs["content"] == "Verify the journal before sync."
    with pytest.raises(TypeError):
        intent.save_kwargs["content"] = "cannot mutate"  # type: ignore[index]
    with pytest.raises(TypeError):
        intent.save_kwargs["extra"]["learning"]["promoted_at"] = "changed"  # type: ignore[index]


def test_outbox_reuses_memory_after_crash_post_save(tmp_path: Any) -> None:
    clock = _Clock("2026-07-30T12:00:00Z")
    worker, memory, _operational, views = _worker(tmp_path, clock=clock)
    intent = _enqueue(worker)
    memory.fail = "after"

    first = worker.run_once()

    assert first.retried == 1
    assert views.outbox_report().retried == 1
    clock.value = "2026-07-30T12:00:01Z"
    report = worker.run_once()

    assert report.examined == 1
    assert report.completed == 1
    assert report.pending == 0
    record = memory.find_by_operation_key(intent.operation_key, intent.request_hash)
    assert record is not None
    assert record.id == "memory-1"
    assert len(memory.records) == 1


def test_outbox_crash_before_save_retries_only_when_due(tmp_path: Any) -> None:
    clock = _Clock("2026-07-30T12:00:00Z")
    worker, memory, _operational, views = _worker(tmp_path, clock=clock)
    _enqueue(worker)
    memory.fail = "before"

    first = worker.run_once()

    assert first.retried == 1
    assert memory.records == {}
    assert views.outbox_report().retried == 1
    assert worker.run_once().examined == 0
    clock.value = "2026-07-30T12:00:01Z"
    assert worker.run_once().completed == 1
    assert len(memory.records) == 1


def test_old_intent_retry_is_scheduled_from_failure_time(tmp_path: Any) -> None:
    clock = _Clock("2026-07-31T12:00:00Z")
    worker, memory, _operational, views = _worker(tmp_path, clock=clock)
    intent = _enqueue(worker)
    memory.fail = "before"

    report = worker.run_once()

    status = views.outbox_status(intent.id)
    assert report.retried == 1
    assert status is not None
    assert status["failure_at"] == "2026-07-31T12:00:00.000000Z"
    assert status["retry_at"] == "2026-07-31T12:00:01.000000Z"
    assert worker.run_once().examined == 0


def test_retry_delay_caps_before_exponentiation_for_large_attempt_count() -> None:
    assert deterministic_retry_at(
        "2026-07-30T12:00:00Z",
        1_000_000,
    ) == "2026-07-30T13:00:00.000000Z"


def test_transient_failure_does_not_starve_later_intents(tmp_path: Any) -> None:
    clock = _Clock("2026-07-30T12:00:00Z")
    worker, memory, _operational, views = _worker(tmp_path, clock=clock)
    intents = sorted(
        (
            _enqueue(worker, key="promotion-1"),
            _enqueue(worker, key="promotion-2"),
        ),
        key=lambda intent: intent.id,
    )
    failed, completed = intents
    memory.fail = "before"

    report = worker.run_once(limit=2)

    assert report.examined == 2
    assert report.retried == 1
    assert report.completed == 1
    failed_status = views.outbox_status(failed.id)
    completed_status = views.outbox_status(completed.id)
    assert failed_status is not None
    assert completed_status is not None
    assert failed_status["status"] == "retry_scheduled"
    assert completed_status["status"] == "completed"
    assert len(memory.records) == 1


def test_write_policy_rejection_is_terminal_and_quarantined(tmp_path: Any) -> None:
    worker, memory, _operational, views = _worker(tmp_path)
    _enqueue(worker)
    memory.fail = "policy"

    report = worker.run_once()

    assert report.examined == 1
    assert report.quarantined == 1
    assert report.pending == 0
    assert views.outbox_report().quarantined == 1
    assert memory.calls == 1
    assert worker.run_once().examined == 0


def test_bounded_batch_leaves_unexamined_intents_pending(tmp_path: Any) -> None:
    worker, memory, _operational, _views = _worker(tmp_path)
    for index in range(3):
        _enqueue(worker, key=f"promotion-{index}")

    first = worker.run_once(limit=2)
    second = worker.run_once(limit=2)

    assert first.examined == 2
    assert first.completed == 2
    assert first.pending == 1
    assert second.examined == 1
    assert second.completed == 1
    assert second.pending == 0
    assert len(memory.records) == 3


def test_changed_payload_for_same_promotion_key_never_overwrites_intent(
    tmp_path: Any,
) -> None:
    worker, _memory, _operational, views = _worker(tmp_path)
    original = _enqueue(worker)

    with pytest.raises(OperationalError) as raised:
        worker.enqueue(
            idempotency_key="promotion-1",
            save_kwargs=_save_kwargs("different"),
            source_event_ids=("outcome-event-1",),
            created_at="2026-07-30T12:00:00Z",
        )

    assert raised.value.code is OperationalErrorCode.IDEMPOTENCY_CONFLICT
    pending = views.pending_outbox(limit=10, now="2026-07-30T12:00:10Z")
    assert len(pending) == 1
    assert pending[0].request_hash == original.request_hash


def test_rebuild_between_requested_and_completed_preserves_exactly_once(
    tmp_path: Any,
) -> None:
    worker, memory, operational, views = _worker(tmp_path)
    intent = _enqueue(worker)

    rebuilt = views.rebuild(tuple(operational.events))
    report = worker.run_once()

    assert rebuilt.applied == 1
    assert report.completed == 1
    assert memory.find_by_operation_key(intent.operation_key, intent.request_hash) is not None
    assert len(memory.records) == 1


def test_source_event_provenance_reaches_the_saved_request(tmp_path: Any) -> None:
    worker, memory, _operational, _views = _worker(tmp_path)
    intent = _enqueue(worker)

    report = worker.run_once()

    assert report.completed == 1
    saved = memory.find_by_operation_key(intent.operation_key, intent.request_hash)
    assert saved is not None
    assert saved.extra["provenance"]["source_event_ids"] == [
        "outcome-event-1",
        "outcome-event-2",
    ]


def test_synchronous_reconcile_returns_and_replays_the_requested_memory(
    tmp_path: Any,
) -> None:
    worker, memory, _operational, _views = _worker(tmp_path)
    intent = _enqueue(worker)

    first = worker.reconcile(intent)
    replayed_intent = worker.enqueue(
        idempotency_key="promotion-1",
        save_kwargs=_save_kwargs(),
        source_event_ids=("outcome-event-1", "outcome-event-2"),
        created_at="2026-07-30T13:00:00Z",
    )
    replay = worker.reconcile(replayed_intent)

    assert replay.id == first.id
    assert memory.calls == 1
    assert len(memory.records) == 1


def test_run_once_rejects_invalid_limit_without_touching_state(tmp_path: Any) -> None:
    worker, memory, _operational, views = _worker(tmp_path)
    _enqueue(worker)

    with pytest.raises(ValueError, match="limit"):
        worker.run_once(limit=0)

    assert memory.calls == 0
    assert views.outbox_report().pending == 1
