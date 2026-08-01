from __future__ import annotations

import hashlib
from contextlib import closing
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from memo.errors import AuthorityEpochError, OperationalError, OperationalErrorCode
from memo.identity import PrincipalIdentity
from memo.operation_ledger_v2 import OperationLedgerV2
from memo.operation_view_schema import connect_operational_db
from memo.operation_views import OperationalViewStore
from memo.operational import OperationalStore
from memo.operational_epoch import CommitContext, EpochFence
from memo.operational_event import (
    CommandResult,
    EpochMarkerAuthorization,
    OperationalCommand,
    OperationalEventV2,
    canonical_json_bytes,
    canonical_signed_bytes,
)
from memo.operational_event_types import COORDINATION_CREATED, FOCUS_SET
from memo.operational_key_store import (
    AuthorityPinStore,
    DeviceKeyStore,
    InMemoryAuthorityPinProvider,
)
from memo.operational_roster import VerificationRoster
from memo.operational_signing import OperationalSigner, OperationalVerifier


def _identity() -> PrincipalIdentity:
    return PrincipalIdentity(
        principal_id="device-a:session-a",
        actor_id="agent-a",
        kind="agent",
        device_id="device-a",
        session_id="session-a",
        source_client="codex",
    )


def _context(identity: PrincipalIdentity | None = None) -> CommitContext:
    return CommitContext(
        identity=identity or _identity(),
        authority_epoch=0,
        control_oid="control-0",
        origin_device="device-a",
    )


def _command(**changes: object) -> OperationalCommand:
    value = OperationalCommand(
        event_type=FOCUS_SET,
        actor=_identity(),
        target_id=None,
        project="demo",
        workspace="/tmp/demo",
        expires_at=None,
        visibility="owner",
        idempotency_key="focus-1",
        caused_by=(),
        subject_uri="memo://focus/demo",
        trace_id="trace-1",
        payload={"project": "demo", "summary": "native memo"},
    )
    return replace(value, **changes)


class _Fence:
    def __init__(self) -> None:
        self.error: Exception | None = None
        self.calls = 0

    def verify(self, _context: CommitContext) -> None:
        self.calls += 1
        if self.error is not None:
            raise self.error


class _Ledger:
    def __init__(self) -> None:
        self.events: list[OperationalEventV2] = []
        self.append_calls = 0

    def validated_events(self) -> list[OperationalEventV2]:
        return list(self.events)

    def append(
        self,
        command: OperationalCommand,
        *,
        context: CommitContext,
    ) -> OperationalEventV2:
        self.append_calls += 1
        sequence = len(self.events) + 1
        content_hash = hashlib.sha256(canonical_json_bytes(asdict(command))).hexdigest()
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
            content_hash=content_hash,
            previous_hash=self.events[-1].event_hash if self.events else "",
            event_hash=f"{sequence:064x}",
            source_proof=command.source_proof,
            roster_version=1,
            key_id="key-1",
            signature="signature",
        )
        self.events.append(event)
        return event


def _store(
    tmp_path: Path,
    *,
    ledger: _Ledger | None = None,
) -> tuple[OperationalStore, _Ledger, _Fence]:
    ledger = ledger or _Ledger()
    fence = _Fence()
    views = OperationalViewStore(tmp_path / "operational.db")
    store = OperationalStore.for_v2(
        ledger=ledger,
        views=views,
        epoch_fence=fence,
        transaction_root=tmp_path / "operational-transactions",
    )
    return store, ledger, fence


def _real_store(tmp_path: Path) -> tuple[OperationalStore, OperationLedgerV2, CommitContext]:
    root = tmp_path.resolve()
    keys = DeviceKeyStore.in_memory()
    origin_key = keys.generate(device_id="device-a", roles=("origin",))
    pin_store = AuthorityPinStore._for_test(
        root,
        provider=InMemoryAuthorityPinProvider(),
    )
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=origin_key,
        root=root,
        pin_store=pin_store,
    )
    signer = OperationalSigner(keys, roster_version=roster.version)
    verifier = OperationalVerifier()
    fence = EpochFence(
        root,
        roster=roster,
        verifier=verifier,
        pin_store=pin_store,
    )
    unsigned = EpochMarkerAuthorization(
        schema="memo.operational_epoch_authorization.v1",
        attempt_id="attempt-0",
        device_id="device-a",
        epoch=0,
        control_oid="control-0",
        artifact_digests={
            "bootstrap_roster": "a" * 64,
            "empty_anchor": "b" * 64,
        },
        roster_version=roster.version,
        key_id=origin_key.key_id,
        signature=None,  # type: ignore[arg-type]
    )
    authorization = replace(
        unsigned,
        signature=signer.sign(
            domain="memo.operational_epoch_authorization.v1",
            payload=canonical_signed_bytes(unsigned),
            key_id=origin_key.key_id,
        ),
    )
    fence.bootstrap(
        authorization=authorization,
        observed_artifact_digests=authorization.artifact_digests,
    )
    ledger = OperationLedgerV2(
        root,
        device_id="device-a",
        clock=lambda: "2026-07-30T12:00:00Z",
        signer=signer,
        verifier=verifier,
        roster=roster,
        roster_root=root,
        pin_store=pin_store,
        epoch_fence=fence,
    )
    store = OperationalStore.for_v2(
        ledger=ledger,
        views=OperationalViewStore(root / "operational-v2" / "operational.db"),
        epoch_fence=fence,
        transaction_root=root / "operational-transactions",
    )
    context = fence.context(
        _identity(),
        request_epoch=0,
        request_control_oid="control-0",
    )
    return store, ledger, context


def test_commit_integrates_real_ledger_fence_and_views(tmp_path: Path) -> None:
    store, ledger, context = _real_store(tmp_path)

    first = store.commit(_command(), context=context)
    replay = store.commit(_command(), context=context)

    assert first.replayed is False
    assert replay.replayed is True
    assert replay.event == first.event
    assert len(ledger.validated_events()) == 1
    assert store.views.state()["focus"]["demo"]["summary"] == "native memo"


@pytest.mark.parametrize(
    "context_change",
    (
        {"authority_epoch": -1},
        {"authority_epoch": 1},
        {"control_oid": "wrong-control"},
    ),
)
def test_commit_rejects_invalid_epoch_or_control_before_real_append(
    tmp_path: Path,
    context_change: dict[str, object],
) -> None:
    store, ledger, context = _real_store(tmp_path)

    with pytest.raises(AuthorityEpochError):
        store.commit(_command(), context=replace(context, **context_change))

    assert ledger.validated_events() == []


def test_commit_exact_replay_returns_stored_result(tmp_path: Path) -> None:
    store, ledger, _fence = _store(tmp_path)
    command = _command()

    first = store.commit(command, context=_context())
    replay = store.commit(command, context=_context())

    assert first.replayed is False
    assert replay.replayed is True
    assert replay.event == first.event
    assert replay.result == first.result
    assert ledger.append_calls == 1


def test_commit_replays_append_before_view_crash_without_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, ledger, _fence = _store(tmp_path)
    command = _command()
    real_apply = store.views.apply_events

    def fail_after_append(events: object) -> object:
        materialized = tuple(events)  # type: ignore[arg-type]
        if materialized:
            raise OSError("simulated view crash")
        return real_apply(materialized)

    monkeypatch.setattr(store.views, "apply_events", fail_after_append)
    with pytest.raises(OSError, match="simulated view crash"):
        store.commit(command, context=_context())
    assert len(ledger.events) == 1

    monkeypatch.setattr(store.views, "apply_events", real_apply)
    replay = store.commit(command, context=_context())

    assert isinstance(replay, CommandResult)
    assert replay.replayed is True
    assert ledger.append_calls == 1
    assert len(ledger.events) == 1


def test_commit_catches_up_an_interposed_same_origin_event(tmp_path: Path) -> None:
    class _InterleavingLedger(_Ledger):
        def append(
            self,
            command: OperationalCommand,
            *,
            context: CommitContext,
        ) -> OperationalEventV2:
            if not self.events:
                super().append(
                    _command(
                        idempotency_key="interposed-1",
                        payload={"project": "demo", "summary": "interposed"},
                    ),
                    context=context,
                )
            return super().append(command, context=context)

    interleaving = _InterleavingLedger()
    store, ledger, _fence = _store(tmp_path, ledger=interleaving)

    result = store.commit(_command(), context=_context())

    assert result.replayed is False
    assert len(ledger.events) == 2
    assert store.views.state()["focus"]["demo"]["summary"] == "native memo"
    with closing(connect_operational_db(store.views.path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM applied_events").fetchone()[0] == 2


def test_idempotency_key_with_different_request_is_rejected(tmp_path: Path) -> None:
    store, ledger, _fence = _store(tmp_path)
    store.commit(_command(), context=_context())

    with pytest.raises(OperationalError) as raised:
        store.commit(
            _command(payload={"project": "demo", "summary": "different"}),
            context=_context(),
        )

    assert raised.value.code == OperationalErrorCode.IDEMPOTENCY_CONFLICT
    assert ledger.append_calls == 1


def test_local_commit_rejects_event_without_view_reducer_before_append(
    tmp_path: Path,
) -> None:
    store, ledger, _fence = _store(tmp_path)

    with pytest.raises(OperationalError, match="view reducer"):
        store.commit(
            _command(
                event_type=COORDINATION_CREATED,
                payload={
                    "task_id": "task-1",
                    "summary": "coordinate",
                    "status": "pending",
                },
            ),
            context=_context(),
        )

    assert ledger.append_calls == 0


def test_missing_empty_or_mismatched_authority_fails_before_append(
    tmp_path: Path,
) -> None:
    store, ledger, fence = _store(tmp_path)

    with pytest.raises(OperationalError):
        store.commit(_command(), context=None)  # type: ignore[arg-type]
    with pytest.raises(OperationalError, match="idempotency"):
        store.commit(_command(idempotency_key=""), context=_context())
    with pytest.raises(OperationalError, match=r"actor|principal"):
        store.commit(
            _command(actor=replace(_identity(), actor_id="other")),
            context=_context(),
        )
    fence.error = AuthorityEpochError("stale authority epoch")
    with pytest.raises(AuthorityEpochError, match="stale"):
        store.commit(_command(), context=_context())

    assert ledger.append_calls == 0


def test_default_operational_store_remains_v1_until_activation(tmp_path: Path) -> None:
    store = OperationalStore(tmp_path, device_id="device-a")

    assert not hasattr(store, "views")
    with pytest.raises(OperationalError, match="v2"):
        store.commit(_command(), context=_context())
