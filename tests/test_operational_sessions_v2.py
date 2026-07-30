from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from memo.errors import OperationalError, OperationalErrorCode
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
from memo.operational_sessions import (
    LegacySessionMigrator,
    OperationalSessionService,
)


def _identity(
    *,
    principal_id: str = "device-a:session-a",
    session_id: str = "session-a",
) -> PrincipalIdentity:
    return PrincipalIdentity(
        principal_id=principal_id,
        actor_id="agent-a",
        kind="agent",
        device_id="device-a",
        session_id=session_id,
        source_client="codex",
    )


def _context(identity: PrincipalIdentity) -> CommitContext:
    return CommitContext(
        identity=identity,
        authority_epoch=0,
        control_oid="control-0",
        origin_device="device-a",
    )


class _Clock:
    def __init__(self, value: str = "2026-07-30T12:00:00Z") -> None:
        self.value = value

    def __call__(self) -> str:
        return self.value


class _Operational:
    def __init__(self, views: OperationalViewStore) -> None:
        self.views = views
        self.ledger = self
        self.events: list[OperationalEventV2] = []
        self.defer_next_view = False
        self._commands: dict[
            tuple[str, str],
            tuple[str, OperationalEventV2, dict[str, object]],
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
                    retryable=False,
                )
            return CommandResult(
                event=existing[1],
                replayed=True,
                result=existing[2],
            )
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
        if self.defer_next_view:
            self.defer_next_view = False
        else:
            report = self.views.apply_events((event,))
            assert report.applied == 1
        result = {
            "event_id": event.event_id,
            "event_hash": event.event_hash,
            "event_type": event.event_type,
            "value": self.views.session(str(command.target_id)).to_dict(),
        }
        self._commands[key] = (request_hash, event, result)
        return CommandResult(event=event, replayed=False, result=result)

    def validated_events(self) -> tuple[OperationalEventV2, ...]:
        return tuple(self.events)


def _service(
    tmp_path: Path,
) -> tuple[
    OperationalSessionService,
    _Operational,
    OperationalViewStore,
    _Clock,
]:
    views = OperationalViewStore(tmp_path / "operational.db")
    operational = _Operational(views)
    clock = _Clock()
    service = OperationalSessionService(
        operational=operational,
        views=views,
        context_factory=_context,
        clock=clock,
    )
    return service, operational, views, clock


def _checkpoint(
    service: OperationalSessionService,
    *,
    identity: PrincipalIdentity | None = None,
    session_id: str = "session-a",
    project: str = "memo",
    workspace: str = "/work/memo",
    summary: str = "working",
    idempotency_key: str = "checkpoint-1",
    checkpointed_at: str = "2026-07-30T12:00:00Z",
):
    return service.checkpoint(
        identity=identity or _identity(),
        session_id=session_id,
        project=project,
        workspace=workspace,
        summary=summary,
        branch="main",
        head="abc123",
        source_event_id=f"source-{idempotency_key}",
        checkpointed_at=checkpointed_at,
        idempotency_key=idempotency_key,
    )


def test_checkpoint_replay_recovers_original_request_but_rejects_explicit_drift(
    tmp_path: Path,
) -> None:
    service, operational, _views, _clock = _service(tmp_path)
    original = _checkpoint(service)

    replayed = service.replay_checkpoint(
        identity=_identity(),
        session_id="session-a",
        project="memo",
        workspace="/work/memo",
        source_event_id="source-checkpoint-1",
        checkpointed_at="2026-07-30T12:00:00Z",
        idempotency_key="checkpoint-1",
    )

    assert replayed == original
    assert len(operational.events) == 1

    with pytest.raises(OperationalError) as source_conflict:
        service.replay_checkpoint(
            identity=_identity(),
            session_id="session-a",
            project="memo",
            workspace="/work/memo",
            source_event_id="different-explicit-source",
            checkpointed_at="2026-07-30T12:00:00Z",
            idempotency_key="checkpoint-1",
        )
    assert source_conflict.value.code == OperationalErrorCode.IDEMPOTENCY_CONFLICT

    with pytest.raises(OperationalError) as workspace_conflict:
        service.replay_checkpoint(
            identity=_identity(),
            session_id="session-a",
            project="memo",
            workspace="/work/other",
            source_event_id="source-checkpoint-1",
            checkpointed_at="2026-07-30T12:00:00Z",
            idempotency_key="checkpoint-1",
        )
    assert workspace_conflict.value.code == OperationalErrorCode.IDEMPOTENCY_CONFLICT


def test_session_lifecycle_is_monotonic_and_terminal_is_final(
    tmp_path: Path,
) -> None:
    service, _operational, _views, _clock = _service(tmp_path)
    checkpoint = _checkpoint(service)
    terminated = service.terminate(
        identity=_identity(),
        session_id="session-a",
        summary="done",
        terminated_at="2026-07-30T12:01:00Z",
        idempotency_key="terminate-1",
    )

    assert checkpoint.status == "active"
    assert terminated.status == "terminated"
    assert terminated.summary == "done"

    with pytest.raises(OperationalError, match=r"terminal|terminated"):
        _checkpoint(
            service,
            summary="regress",
            idempotency_key="checkpoint-2",
            checkpointed_at="2026-07-30T12:02:00Z",
        )
    with pytest.raises(OperationalError, match=r"terminal|terminated"):
        service.mark_recoverable(
            identity=_identity(),
            session_id="session-a",
            reason="should fail",
            recoverable_at="2026-07-30T12:02:00Z",
            idempotency_key="recoverable-1",
        )


def test_recoverable_session_cannot_regress_to_active_and_can_terminate(
    tmp_path: Path,
) -> None:
    service, _operational, _views, _clock = _service(tmp_path)
    _checkpoint(service)
    recoverable = service.mark_recoverable(
        identity=_identity(),
        session_id="session-a",
        reason="client disconnected",
        recoverable_at="2026-07-30T12:01:00Z",
        idempotency_key="recoverable-1",
    )

    assert recoverable.status == "recoverable"
    assert recoverable.recoverable_reason == "client disconnected"
    with pytest.raises(OperationalError, match="recoverable"):
        _checkpoint(
            service,
            idempotency_key="checkpoint-2",
            checkpointed_at="2026-07-30T12:02:00Z",
        )

    terminated = service.terminate(
        identity=_identity(),
        session_id="session-a",
        summary="closed after recovery",
        terminated_at="2026-07-30T12:03:00Z",
        idempotency_key="terminate-1",
    )
    assert terminated.status == "terminated"


def test_session_identity_project_and_workspace_are_immutable(
    tmp_path: Path,
) -> None:
    service, _operational, _views, _clock = _service(tmp_path)
    _checkpoint(service)

    changes = (
        {"identity": _identity(principal_id="device-b:session-a")},
        {"project": "other"},
        {"workspace": "/work/other"},
    )
    for index, changed in enumerate(changes, start=2):
        with pytest.raises(
            OperationalError,
            match=r"identity|project|workspace",
        ):
            _checkpoint(
                service,
                idempotency_key=f"checkpoint-{index}",
                checkpointed_at=f"2026-07-30T12:0{index}:00Z",
                **changed,
            )


def test_command_replay_is_exact_and_changed_request_conflicts(
    tmp_path: Path,
) -> None:
    service, operational, _views, _clock = _service(tmp_path)
    first = _checkpoint(service)
    replay = _checkpoint(service)

    assert replay == first
    assert len(operational.events) == 1

    with pytest.raises(OperationalError) as raised:
        _checkpoint(service, summary="changed under same key")
    assert raised.value.code is OperationalErrorCode.IDEMPOTENCY_CONFLICT


def test_terminal_command_replays_after_session_is_terminal(
    tmp_path: Path,
) -> None:
    service, operational, _views, _clock = _service(tmp_path)
    _checkpoint(service)
    first = service.terminate(
        identity=_identity(),
        session_id="session-a",
        summary="done",
        terminated_at="2026-07-30T12:01:00Z",
        idempotency_key="terminate-1",
    )
    replay = service.terminate(
        identity=_identity(),
        session_id="session-a",
        summary="done",
        terminated_at="2026-07-30T12:01:00Z",
        idempotency_key="terminate-1",
    )

    assert replay == first
    assert len(operational.events) == 2

    with pytest.raises(OperationalError) as raised:
        service.terminate(
            identity=_identity(),
            session_id="session-a",
            summary="changed",
            terminated_at="2026-07-30T12:01:00Z",
            idempotency_key="terminate-1",
        )
    assert raised.value.code is OperationalErrorCode.IDEMPOTENCY_CONFLICT


def test_replay_with_omitted_timestamp_reuses_durable_command_time(
    tmp_path: Path,
) -> None:
    service, operational, _views, clock = _service(tmp_path)
    first_checkpoint = _checkpoint(service, checkpointed_at=None)
    clock.value = "2026-07-30T13:00:00Z"
    replayed_checkpoint = _checkpoint(service, checkpointed_at=None)
    first_termination = service.terminate(
        identity=_identity(),
        session_id="session-a",
        summary="done",
        terminated_at=None,
        idempotency_key="terminate-1",
    )
    clock.value = "2026-07-30T14:00:00Z"
    replayed_termination = service.terminate(
        identity=_identity(),
        session_id="session-a",
        summary="done",
        terminated_at=None,
        idempotency_key="terminate-1",
    )

    assert replayed_checkpoint == first_checkpoint
    assert replayed_termination == first_termination
    assert replayed_termination.terminated_at == "2026-07-30T13:00:00.000000Z"
    assert len(operational.events) == 2


def test_preconditions_catch_up_durable_events_before_append(
    tmp_path: Path,
) -> None:
    service, operational, views, _clock = _service(tmp_path)
    _checkpoint(service)
    operational.defer_next_view = True
    service.terminate(
        identity=_identity(),
        session_id="session-a",
        summary="durable but not projected",
        terminated_at="2026-07-30T12:01:00Z",
        idempotency_key="terminate-1",
    )

    assert views.session("session-a").status == "active"
    with pytest.raises(OperationalError, match="terminal"):
        _checkpoint(
            service,
            summary="must not append after catch-up",
            idempotency_key="checkpoint-2",
            checkpointed_at="2026-07-30T12:02:00Z",
        )
    assert views.session("session-a").status == "terminated"
    assert len(operational.events) == 2


def test_canonical_reads_catch_up_durable_unprojected_events(
    tmp_path: Path,
) -> None:
    service, operational, views, _clock = _service(tmp_path)
    _checkpoint(service)
    operational.defer_next_view = True
    service.mark_recoverable(
        identity=_identity(),
        session_id="session-a",
        reason="durable but not projected",
        recoverable_at="2026-07-30T12:01:00Z",
        idempotency_key="recoverable-1",
    )

    assert views.session("session-a").status == "active"
    recovered = service.get("session-a")
    assert recovered is not None
    assert recovered.status == "recoverable"
    assert service.latest_recoverable(project="memo").session_id == "session-a"


def test_latest_recoverable_excludes_active_and_terminated_and_is_stable(
    tmp_path: Path,
) -> None:
    service, _operational, _views, _clock = _service(tmp_path)
    _checkpoint(
        service,
        session_id="session-a",
        idempotency_key="checkpoint-a",
    )
    _checkpoint(
        service,
        identity=_identity(
            principal_id="device-a:session-b",
            session_id="session-b",
        ),
        session_id="session-b",
        idempotency_key="checkpoint-b",
    )
    _checkpoint(
        service,
        identity=_identity(
            principal_id="device-a:session-c",
            session_id="session-c",
        ),
        session_id="session-c",
        idempotency_key="checkpoint-c",
    )
    service.mark_recoverable(
        identity=_identity(),
        session_id="session-a",
        reason="older",
        recoverable_at="2026-07-30T12:01:00Z",
        idempotency_key="recoverable-a",
    )
    service.mark_recoverable(
        identity=_identity(
            principal_id="device-a:session-b",
            session_id="session-b",
        ),
        session_id="session-b",
        reason="newer",
        recoverable_at="2026-07-30T12:02:00Z",
        idempotency_key="recoverable-b",
    )
    service.terminate(
        identity=_identity(
            principal_id="device-a:session-b",
            session_id="session-b",
        ),
        session_id="session-b",
        summary="done",
        terminated_at="2026-07-30T12:03:00Z",
        idempotency_key="terminate-b",
    )

    latest = service.latest_recoverable(project="memo", workspace="/work/memo")
    assert latest is not None
    assert latest.session_id == "session-a"
    assert latest.status == "recoverable"


def test_local_artifacts_survive_view_rebuild_and_never_enter_state(
    tmp_path: Path,
) -> None:
    service, operational, views, _clock = _service(tmp_path)
    session = _checkpoint(service)
    artifacts = {
        "transcript_path": "/private/transcript.jsonl",
        "prompt_trail": ["secret local prompt"],
        "turn_count": 4,
    }
    views.replace_session_local_artifacts(session.session_id, artifacts)

    assert views.session_local_artifacts(session.session_id) == artifacts
    state = views.state()
    assert "session_local_artifacts" not in state
    assert "/private/transcript.jsonl" not in str(state)

    rebuilt = views.rebuild(tuple(operational.events))
    assert rebuilt.applied == 1
    assert views.session_local_artifacts(session.session_id) == artifacts
    assert "/private/transcript.jsonl" not in str(views.state())
    assert all(
        "transcript_path" not in event.payload and "prompt_trail" not in event.payload
        for event in operational.events
    )


def test_legacy_merge_separates_portable_and_local_fields() -> None:
    merged = LegacySessionMigrator.merge_legacy(
        json_checkpoint={
            "session_id": "session-a",
            "cwd": "/work/memo",
            "project": "memo",
            "branch": "main",
            "head_commit": "abc123",
            "summary": "working",
            "updated": "2026-07-30T12:00:00Z",
            "transcript_path": "/private/transcript.jsonl",
            "prompt_trail": ["one", "two"],
            "last_user_msg": "local prompt",
        },
        sqlite_row={
            "id": "session-a",
            "project": "memo",
            "directory": "/work/memo",
            "status": "recoverable",
            "summary": "working",
            "started_at": "2026-07-30T11:00:00Z",
        },
    )

    portable = asdict(merged.checkpoint)
    assert portable["session_id"] == "session-a"
    assert portable["project"] == "memo"
    assert portable["workspace"] == "/work/memo"
    assert portable["status"] == "recoverable"
    assert "transcript_path" not in portable
    assert "prompt_trail" not in portable
    assert merged.local_artifacts["transcript_path"] == "/private/transcript.jsonl"
    assert merged.local_artifacts["prompt_trail"] == ["one", "two"]
    assert merged.local_artifacts["last_user_msg"] == "local prompt"


@pytest.mark.parametrize(
    ("json_checkpoint", "sqlite_row", "match"),
    [
        (
            {"session_id": "session-a", "project": "memo", "cwd": "/work/memo"},
            {"id": "session-b", "project": "memo", "directory": "/work/memo"},
            "session",
        ),
        (
            {"session_id": "session-a", "project": "memo", "cwd": "/work/memo"},
            {"id": "session-a", "project": "other", "directory": "/work/memo"},
            "project",
        ),
        (
            {"session_id": "session-a", "project": "memo", "cwd": "/work/memo"},
            {"id": "session-a", "project": "memo", "directory": "/work/other"},
            "workspace",
        ),
    ],
)
def test_legacy_merge_rejects_incompatible_sources(
    json_checkpoint: dict[str, Any],
    sqlite_row: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(OperationalError, match=match):
        LegacySessionMigrator.merge_legacy(json_checkpoint, sqlite_row)
