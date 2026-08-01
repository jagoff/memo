from __future__ import annotations

import hashlib
import json
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest

import memo.operation_migration as operation_migration
from memo.contracts import MemoEvent
from memo.errors import OperationalError
from memo.operation_ledger_v1 import LegacyOperationLedger
from memo.operation_ledger_v2 import OperationLedgerV2
from memo.operation_migration import (
    MigrationParityReport,
    V1MigrationAuthority,
    apply_v1_migration,
    migrate_v1,
    plan_v1_migration,
    verify_v1_parity,
)
from memo.operational_epoch import EpochFence
from memo.operational_event import (
    EpochMarkerAuthorization,
    canonical_json_bytes,
    canonical_signed_bytes,
)
from memo.operational_key_store import (
    AuthorityPinStore,
    DeviceKeyStore,
    InMemoryAuthorityPinProvider,
)
from memo.operational_roster import VerificationRoster
from memo.operational_signing import OperationalSigner, OperationalVerifier

_LOCAL = "device-a"
_REMOTE = "device-b"
_ISSUED_AT = "2026-07-29T12:00:00Z"
_EXPIRES_AT = "2026-07-29T13:00:00Z"


def _authorization(
    signer: OperationalSigner,
    *,
    key_id: str,
) -> EpochMarkerAuthorization:
    unsigned = EpochMarkerAuthorization(
        schema="memo.operational_epoch_authorization.v1",
        attempt_id="attempt-0",
        device_id=_LOCAL,
        epoch=0,
        control_oid="control-0",
        artifact_digests={
            "bootstrap_roster": "a" * 64,
            "empty_anchor": "b" * 64,
        },
        roster_version=signer.roster_version,
        key_id=key_id,
        signature=None,  # type: ignore[arg-type]
    )
    return replace(
        unsigned,
        signature=signer.sign(
            domain="memo.operational_epoch_authorization.v1",
            payload=canonical_signed_bytes(unsigned),
            key_id=key_id,
        ),
    )


def _authority(root: Path) -> V1MigrationAuthority:
    keys = DeviceKeyStore.in_memory()
    local_key = keys.generate(device_id=_LOCAL, roles=("origin",))
    pins = AuthorityPinStore._for_test(
        root,
        provider=InMemoryAuthorityPinProvider(),
    )
    roster = VerificationRoster.bootstrap(
        device_id=_LOCAL,
        key=local_key,
        root=root,
        pin_store=pins,
    )
    signer = OperationalSigner(keys, roster_version=roster.version)
    verifier = OperationalVerifier()
    fence = EpochFence(
        root,
        roster=roster,
        verifier=verifier,
        pin_store=pins,
    )
    authorization = _authorization(signer, key_id=local_key.key_id)
    fence.bootstrap(
        authorization=authorization,
        observed_artifact_digests=authorization.artifact_digests,
    )

    remote_key = keys.generate(
        device_id=_REMOTE,
        roles=("origin",),
        enrollment_sequence=2,
    )
    attestor = keys.generate(
        device_id=_LOCAL,
        roles=("migration_attestor",),
        enrollment_sequence=2,
    )
    roster = roster.with_keys(
        version=2,
        peers=(_LOCAL, _REMOTE),
        keys=(local_key, remote_key, attestor),
        signer=signer,
        root=root,
        pin_store=pins,
    )
    signer = OperationalSigner(keys, roster_version=roster.version)
    return V1MigrationAuthority(
        signer=signer,
        verifier=verifier,
        roster=roster,
        roster_root=root,
        pin_store=pins,
        epoch_fence=fence,
        attestor_key_id=attestor.key_id,
        capability_manifest_sha256="c" * 64,
        authority_epoch=0,
        control_oid="control-0",
        issued_at=_ISSUED_AT,
        expires_at=_EXPIRES_AT,
    )


def _append(
    ledger: LegacyOperationLedger,
    op: str,
    *,
    ts: str,
    subject: str,
    payload: dict[str, object],
) -> None:
    ledger.append(
        op,
        subject_uri=subject,
        payload=payload,
        content_hash=hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        ts=ts,
    )


def _populate_v1(source: Path) -> None:
    local = LegacyOperationLedger(source, device_id=_LOCAL)
    remote = LegacyOperationLedger(source, device_id=_REMOTE)
    _append(
        local,
        "focus.set",
        ts="2026-07-29T10:00:00Z",
        subject="memo://focus/memo",
        payload={
            "id": "focus-local",
            "project": "memo",
            "summary": "local focus",
            "updated_at": "2026-07-29T10:00:00Z",
            "actor_id": "codex",
            "metadata": {},
        },
    )
    _append(
        local,
        "handoff.create",
        ts="2026-07-29T10:01:00Z",
        subject="memo://handoff/handoff-1",
        payload={
            "id": "handoff-1",
            "project": "memo",
            "summary": "continue migration",
            "from_actor": "codex",
            "to_actor": "claude",
            "created_at": "2026-07-29T10:01:00Z",
            "consumed_at": "",
            "metadata": {"priority": "high"},
        },
    )
    _append(
        local,
        "attention.add",
        ts="2026-07-29T10:02:00Z",
        subject="memo://attention/attention-1",
        payload={
            "id": "attention-1",
            "project": "memo",
            "summary": "review parity",
            "severity": "high",
            "created_at": "2026-07-29T10:02:00Z",
            "acknowledged_at": "",
            "metadata": {},
        },
    )
    _append(
        local,
        "conflict.open",
        ts="2026-07-29T10:03:00Z",
        subject="memo://conflict/conflict-1",
        payload={
            "id": "conflict-1",
            "topic": "migration",
            "summary": "manual conflict",
            "lifecycle_state": "detected",
            "freeze_write": True,
            "created_at": "2026-07-29T10:03:00Z",
            "resolved_at": "",
            "resolution": "",
            "evidence_uris": ["memo://memory/aaaaaaaa"],
            "metadata": {},
        },
    )
    _append(
        local,
        "outcome.record",
        ts="2026-07-29T10:04:00Z",
        subject="memo://outcome/task-1",
        payload={
            "task_id": "task-1",
            "status": "partial",
            "memory_ids": ["aaaaaaaa"],
            "artifacts": [],
            "environment": {"runner": "local"},
            "actor_id": "codex",
            "idempotency_key": "outcome-1",
            "recorded_at": "2026-07-29T10:04:00Z",
        },
    )
    _append(
        local,
        "anomaly.record",
        ts="2026-07-29T10:05:00Z",
        subject="memo://anomaly/anomaly-1",
        payload={
            "kind": "semantic_contradiction",
            "anomaly_id": "anomaly-1",
            "state": "detected",
            "memory_id_a": "aaaaaaaa",
            "memory_id_b": "bbbbbbbb",
            "relationship": "contradicts",
            "summary": "semantic conflict",
            "severity": "high",
            "confidence": 0.99,
            "created_at": "2026-07-29T10:05:00Z",
            "evidence_uris": [
                "memo://memory/aaaaaaaa",
                "memo://memory/bbbbbbbb",
            ],
        },
    )
    _append(
        remote,
        "focus.set",
        ts="2026-07-29T11:00:00Z",
        subject="memo://focus/memo",
        payload={
            "id": "focus-remote",
            "project": "memo",
            "summary": "remote focus wins",
            "updated_at": "2026-07-29T11:00:00Z",
            "actor_id": "claude",
            "metadata": {"origin": "remote"},
        },
    )
    _append(
        remote,
        "handoff.consume",
        ts="2026-07-29T11:01:00Z",
        subject="memo://handoff/handoff-1",
        payload={
            "id": "handoff-1",
            "consumed_at": "2026-07-29T11:01:00Z",
        },
    )
    _append(
        remote,
        "attention.ack",
        ts="2026-07-29T11:02:00Z",
        subject="memo://attention/attention-1",
        payload={
            "id": "attention-1",
            "acknowledged_at": "2026-07-29T11:02:00Z",
        },
    )
    _append(
        remote,
        "conflict.resolve",
        ts="2026-07-29T11:03:00Z",
        subject="memo://conflict/conflict-1",
        payload={
            "id": "conflict-1",
            "resolved_at": "2026-07-29T11:03:00Z",
            "resolution": "accepted",
        },
    )
    _append(
        remote,
        "anomaly.record",
        ts="2026-07-29T11:04:00Z",
        subject="memo://anomaly/anomaly-1",
        payload={
            "kind": "semantic_contradiction",
            "anomaly_id": "anomaly-1",
            "state": "resolved",
            "status": "superseded",
            "created_at": "2026-07-29T11:04:00Z",
        },
    )
    _append(
        remote,
        "outcome.record",
        ts="2026-07-29T11:05:00Z",
        subject="memo://outcome/task-1",
        payload={
            "task_id": "task-1",
            "status": "success",
            "memory_ids": ["aaaaaaaa", "bbbbbbbb"],
            "artifacts": ["artifact://report"],
            "environment": {"runner": "remote"},
            "actor_id": "claude",
            "idempotency_key": "outcome-2",
            "recorded_at": "2026-07-29T11:05:00Z",
        },
    )


def _open_target(
    target: Path,
    authority: V1MigrationAuthority,
) -> OperationLedgerV2:
    return OperationLedgerV2(
        target,
        device_id=_LOCAL,
        clock=lambda: _ISSUED_AT,
        signer=authority.signer,
        verifier=authority.verifier,
        roster=authority.roster,
        roster_root=authority.roster_root,
        pin_store=authority.pin_store,
        epoch_fence=authority.epoch_fence,
    )


def _journal_digest(source: Path) -> str:
    root = source / "journal"
    return hashlib.sha256(
        canonical_json_bytes(
            [
                {
                    "path": str(path.relative_to(root)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in sorted(root.rglob("*"))
                if path.is_file()
            ]
        )
    ).hexdigest()


def test_v1_migration_is_deterministic_idempotent_and_exact(tmp_path: Path) -> None:
    source = tmp_path / "v1"
    _populate_v1(source)
    source_before = _journal_digest(source)
    authority = _authority(tmp_path / "authority")
    plan = plan_v1_migration(
        source,
        device_id=_LOCAL,
        workspace="/tmp/memo",
    )
    assert canonical_json_bytes(plan) == canonical_json_bytes(
        plan_v1_migration(
            source,
            device_id=_LOCAL,
            workspace="/tmp/memo",
        )
    )

    first = apply_v1_migration(
        plan,
        tmp_path / "v2-first",
        authority=authority,
    )
    replay = apply_v1_migration(
        plan,
        tmp_path / "v2-first",
        authority=authority,
    )
    second = apply_v1_migration(
        plan,
        tmp_path / "v2-second",
        authority=authority,
    )

    assert first.events_inserted == len(plan.seeds)
    assert replay.events_inserted == 0
    assert first.source_manifest_sha256 == replay.source_manifest_sha256
    assert first.target_generation_sha256 == replay.target_generation_sha256
    assert first.target_generation_sha256 == second.target_generation_sha256
    assert first.seed_event_ids == second.seed_event_ids
    assert first.v1_state_sha256 == first.v2_state_sha256
    assert first.parity.equal is True
    assert first.seed_event_ids == tuple(seed.event_id for seed in plan.seeds)
    assert all(
        event_id.startswith(f"memo-v1/{plan.source_manifest_sha256}/")
        for event_id in first.seed_event_ids
    )

    ledger = _open_target(tmp_path / "v2-first", authority)
    assert ledger.verify().ok is True
    assert {
        bundle.anchor.origin_device: bundle.anchor.base_event_hash
        for bundle in ledger.export_bundles()
    } == dict(plan.source_heads)
    events = ledger.validated_events()
    assert [event.event_id for event in events] == list(first.seed_event_ids)
    assert all(event.source_proof is not None for event in events)
    assert all(
        event.source_proof is not None
        and event.source_proof.authentication is not None
        and event.migration_origin is not None
        for event in events
    )
    assert any(
        event.source_proof is not None
        and event.source_proof.source_origin == _REMOTE
        for event in events
    )
    assert (tmp_path / "v2-first" / "migration-v1.json").is_file()
    assert not (tmp_path / "v2-first" / "operational-v2-activated.json").exists()
    assert json.loads(
        (tmp_path / "v2-first" / "migration-v1.json").read_text(encoding="utf-8")
    )["schema"] == "memo.operational_migration_prepared.v1"
    assert _journal_digest(source) == source_before


def test_existing_target_rejects_a_different_migration_plan(
    tmp_path: Path,
) -> None:
    source = tmp_path / "v1"
    target = tmp_path / "v2"
    _populate_v1(source)
    authority = _authority(tmp_path / "authority")
    original = plan_v1_migration(
        source,
        device_id=_LOCAL,
        workspace="/tmp/original",
    )
    apply_v1_migration(original, target, authority=authority)
    changed = plan_v1_migration(
        source,
        device_id=_LOCAL,
        workspace="/tmp/changed",
    )

    with pytest.raises(OperationalError, match="plan"):
        apply_v1_migration(changed, target, authority=authority)


def test_empty_v1_source_prepares_no_fabricated_origin(tmp_path: Path) -> None:
    source = tmp_path / "v1"
    target = tmp_path / "v2"
    authority = _authority(tmp_path / "authority")
    plan = plan_v1_migration(source, device_id=_LOCAL)

    report = apply_v1_migration(plan, target, authority=authority)
    ledger = _open_target(target, authority)

    assert plan.source_heads == ()
    assert plan.seeds == ()
    assert report.events_inserted == 0
    assert report.parity.equal is True
    assert ledger.export_bundles() == ()
    assert (target / "migration-v1.json").is_file()
    assert not (target / "operational-v2-activated.json").exists()


def test_corrupt_v1_aborts_before_any_v2_write(tmp_path: Path) -> None:
    source = tmp_path / "v1"
    target = tmp_path / "v2"
    _populate_v1(source)
    segment = next((source / "journal" / "events" / _LOCAL).glob("*.jsonl"))
    rows = segment.read_text(encoding="utf-8").splitlines()
    body = json.loads(rows[0])
    body["payload"]["summary"] = "tampered"
    rows[0] = json.dumps(body, ensure_ascii=False, sort_keys=True)
    segment.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(OperationalError):
        migrate_v1(
            source,
            target,
            device_id=_LOCAL,
            workspace="/tmp/memo",
            authority=_authority(tmp_path / "authority"),
        )

    assert not target.exists()


def test_changed_source_after_plan_is_hard_manifest_failure(tmp_path: Path) -> None:
    source = tmp_path / "v1"
    target = tmp_path / "v2"
    _populate_v1(source)
    plan = plan_v1_migration(source, device_id=_LOCAL)
    local = LegacyOperationLedger(source, device_id=_LOCAL)
    _append(
        local,
        "focus.set",
        ts="2026-07-29T12:00:00Z",
        subject="memo://focus/memo",
        payload={
            "id": "focus-late",
            "project": "memo",
            "summary": "source changed",
            "updated_at": "2026-07-29T12:00:00Z",
            "actor_id": "codex",
            "metadata": {},
        },
    )

    with pytest.raises(OperationalError, match="manifest"):
        apply_v1_migration(
            plan,
            target,
            authority=_authority(tmp_path / "authority"),
        )

    assert not target.exists()


def test_plan_uses_one_verified_legacy_snapshot_without_rereading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "v1"
    _populate_v1(source)
    legacy = LegacyOperationLedger(source, device_id=_LOCAL)
    expected_manifest = OperationLedgerV2.legacy_manifest_sha256(legacy)
    segment = next((legacy.root / "events" / _LOCAL).glob("*.jsonl"))
    original_bytes = segment.read_bytes()
    validated_events = LegacyOperationLedger.validated_events

    def mutate_after_validation(instance: LegacyOperationLedger) -> list[MemoEvent]:
        events = validated_events(instance)
        segment.write_bytes(original_bytes + b"{}\n")
        return events

    monkeypatch.setattr(
        LegacyOperationLedger,
        "validated_events",
        mutate_after_validation,
    )

    plan = plan_v1_migration(source, device_id=_LOCAL)

    assert plan.source_manifest_sha256 == expected_manifest
    assert segment.read_bytes() == original_bytes


def test_tampered_plan_is_rederived_and_rejected_before_target_write(
    tmp_path: Path,
) -> None:
    source = tmp_path / "v1"
    target = tmp_path / "v2"
    _populate_v1(source)
    plan = plan_v1_migration(source, device_id=_LOCAL)
    first = plan.seeds[0]
    tampered_seed = replace(
        first,
        payload={**first.payload, "summary": "forged seed"},
    )
    tampered = replace(plan, seeds=(tampered_seed, *plan.seeds[1:]))

    with pytest.raises(OperationalError, match="plan"):
        apply_v1_migration(
            tampered,
            target,
            authority=_authority(tmp_path / "authority"),
        )

    assert not target.exists()


def test_current_v1_snapshot_is_only_a_parity_oracle(tmp_path: Path) -> None:
    source = tmp_path / "v1"
    _populate_v1(source)
    plan = plan_v1_migration(source, device_id=_LOCAL)
    snapshot = json.loads(canonical_json_bytes(plan.source_state))
    snapshot["last_event_hash"] = ""
    snapshot["journal_heads"] = dict(plan.source_heads)
    snapshot["focus"]["memo"]["summary"] = "snapshot divergence"
    (source / "operational-state.json").write_text(
        json.dumps(snapshot, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(OperationalError, match=r"operational-state\.json"):
        plan_v1_migration(source, device_id=_LOCAL)


def test_parity_failure_writes_no_prepared_or_activation_stamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "v1"
    target = tmp_path / "v2"
    _populate_v1(source)
    plan = plan_v1_migration(source, device_id=_LOCAL)
    mismatch = MigrationParityReport(
        equal=False,
        v1_state_sha256=plan.source_state_sha256,
        v2_state_sha256="0" * 64,
        diff=("focus/memo",),
    )
    monkeypatch.setattr(
        operation_migration,
        "verify_v1_parity",
        lambda *_args, **_kwargs: mismatch,
    )

    with pytest.raises(OperationalError, match="state mismatch"):
        apply_v1_migration(
            plan,
            target,
            authority=_authority(tmp_path / "authority"),
        )

    assert not target.exists()
    assert not list(tmp_path.glob("v2*/migration-v1.json"))
    assert not list(tmp_path.glob("v2*/operational-v2-activated.json"))


def test_prepared_generation_is_fsynced_before_publish_and_retry_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "v1"
    target = tmp_path / "v2"
    _populate_v1(source)
    authority = _authority(tmp_path / "authority")
    plan = plan_v1_migration(source, device_id=_LOCAL)
    real_fsync = operation_migration.os.fsync
    real_publish = operation_migration._renameat_exclusive
    fsynced: set[tuple[int, int]] = set()

    def tracked_fsync(descriptor: int) -> None:
        observed = operation_migration.os.fstat(descriptor)
        fsynced.add((observed.st_dev, observed.st_ino))
        real_fsync(descriptor)

    def crash_before_publish(
        parent_descriptor: int,
        source_name: str,
        target_name: str,
    ) -> None:
        observed = operation_migration.os.stat(
            source_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        assert (observed.st_dev, observed.st_ino) in fsynced
        assert next(tmp_path.glob(".v2.staging-*/migration-v1.json")).is_file()
        with pytest.raises(FileNotFoundError):
            operation_migration.os.stat(
                target_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        raise OSError("simulated crash before prepared-generation publish")

    monkeypatch.setattr(operation_migration.os, "fsync", tracked_fsync)
    monkeypatch.setattr(
        operation_migration,
        "_renameat_exclusive",
        crash_before_publish,
    )

    with pytest.raises(OSError, match="simulated crash"):
        apply_v1_migration(plan, target, authority=authority)

    assert not target.exists()
    assert not list(tmp_path.glob(".v2.staging-*"))

    monkeypatch.setattr(operation_migration, "_renameat_exclusive", real_publish)
    report = apply_v1_migration(plan, target, authority=authority)

    assert report.parity.equal is True
    assert (target / "migration-v1.json").is_file()
    assert not (target / "operational-v2-activated.json").exists()


def test_retry_recovers_after_crash_between_publish_and_parent_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "v1"
    target = tmp_path / "v2"
    _populate_v1(source)
    authority = _authority(tmp_path / "authority")
    plan = plan_v1_migration(source, device_id=_LOCAL)
    failpoint = operation_migration._migration_publish_failpoint

    def crash_after_publish(label: str) -> None:
        assert label == "after-rename-before-parent-fsync"
        raise OSError("simulated crash after prepared-generation publish")

    monkeypatch.setattr(
        operation_migration,
        "_migration_publish_failpoint",
        crash_after_publish,
    )

    with pytest.raises(OSError, match="simulated crash after"):
        apply_v1_migration(plan, target, authority=authority)

    assert target.is_dir()
    assert (target / "migration-v1.json").is_file()
    assert not list(tmp_path.glob(".v2.staging-*"))

    monkeypatch.setattr(
        operation_migration,
        "_migration_publish_failpoint",
        failpoint,
    )
    report = apply_v1_migration(plan, target, authority=authority)

    assert report.events_inserted == 0
    assert report.parity.equal is True
    assert not (target / "operational-v2-activated.json").exists()


def test_publish_rejects_parent_swap_before_exclusive_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "v1"
    requested_parent = tmp_path / "requested"
    requested_parent.mkdir()
    target = requested_parent / "v2"
    displaced_parent = tmp_path / "requested-displaced"
    _populate_v1(source)
    authority = _authority(tmp_path / "authority")
    plan = plan_v1_migration(source, device_id=_LOCAL)
    publish = operation_migration._renameat_exclusive

    def swap_parent_before_publish(
        parent_descriptor: int,
        source_name: str,
        target_name: str,
    ) -> None:
        requested_parent.rename(displaced_parent)
        requested_parent.mkdir(mode=0o700)
        publish(parent_descriptor, source_name, target_name)

    monkeypatch.setattr(
        operation_migration,
        "_renameat_exclusive",
        swap_parent_before_publish,
    )

    with pytest.raises(OSError, match="parent namespace identity changed"):
        apply_v1_migration(plan, target, authority=authority)

    assert not target.exists()
    assert (displaced_parent / "v2" / "migration-v1.json").is_file()
    assert not (target / "operational-v2-activated.json").exists()


def test_publish_rejects_parent_swap_after_rename_before_parent_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "v1"
    requested_parent = tmp_path / "requested"
    requested_parent.mkdir()
    target = requested_parent / "v2"
    displaced_parent = tmp_path / "requested-displaced"
    _populate_v1(source)
    authority = _authority(tmp_path / "authority")
    plan = plan_v1_migration(source, device_id=_LOCAL)

    def swap_parent_after_publish(label: str) -> None:
        assert label == "after-rename-before-parent-fsync"
        requested_parent.rename(displaced_parent)
        requested_parent.mkdir(mode=0o700)

    monkeypatch.setattr(
        operation_migration,
        "_migration_publish_failpoint",
        swap_parent_after_publish,
    )

    with pytest.raises(OSError, match="parent namespace identity changed"):
        apply_v1_migration(plan, target, authority=authority)

    assert not target.exists()
    assert (displaced_parent / "v2" / "migration-v1.json").is_file()
    assert not (target / "operational-v2-activated.json").exists()


def test_publish_rejects_boundary_swap_before_exclusive_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "v1"
    boundary = tmp_path / "boundary"
    requested_parent = boundary / "requested"
    requested_parent.mkdir(parents=True)
    target = requested_parent / "v2"
    displaced_boundary = tmp_path / "boundary-displaced"
    _populate_v1(source)
    authority = _authority(tmp_path / "authority")
    plan = plan_v1_migration(source, device_id=_LOCAL)
    publish = operation_migration._renameat_exclusive

    def swap_boundary_before_publish(
        parent_descriptor: int,
        source_name: str,
        target_name: str,
    ) -> None:
        boundary.rename(displaced_boundary)
        boundary.mkdir(mode=0o700)
        (boundary / "requested").mkdir(mode=0o700)
        publish(parent_descriptor, source_name, target_name)

    monkeypatch.setattr(
        operation_migration,
        "_renameat_exclusive",
        swap_boundary_before_publish,
    )

    with pytest.raises(OSError, match="parent namespace identity changed"):
        apply_v1_migration(plan, target, authority=authority)

    assert not target.exists()
    assert (displaced_boundary / "requested" / "v2" / "migration-v1.json").is_file()
    assert not (target / "operational-v2-activated.json").exists()


def test_publish_rejects_boundary_swap_after_rename_before_parent_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "v1"
    boundary = tmp_path / "boundary"
    requested_parent = boundary / "requested"
    requested_parent.mkdir(parents=True)
    target = requested_parent / "v2"
    displaced_boundary = tmp_path / "boundary-displaced"
    _populate_v1(source)
    authority = _authority(tmp_path / "authority")
    plan = plan_v1_migration(source, device_id=_LOCAL)

    def swap_boundary_after_publish(label: str) -> None:
        assert label == "after-rename-before-parent-fsync"
        boundary.rename(displaced_boundary)
        boundary.mkdir(mode=0o700)
        (boundary / "requested").mkdir(mode=0o700)

    monkeypatch.setattr(
        operation_migration,
        "_migration_publish_failpoint",
        swap_boundary_after_publish,
    )

    with pytest.raises(OSError, match="parent namespace identity changed"):
        apply_v1_migration(plan, target, authority=authority)

    assert not target.exists()
    assert (displaced_boundary / "requested" / "v2" / "migration-v1.json").is_file()
    assert not (target / "operational-v2-activated.json").exists()


def test_publish_rejects_replacement_generation_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "v1"
    target = tmp_path / "v2"
    _populate_v1(source)
    authority = _authority(tmp_path / "authority")
    plan = plan_v1_migration(source, device_id=_LOCAL)
    publish = operation_migration._renameat_exclusive

    def replace_after_publish(
        parent_descriptor: int,
        source_name: str,
        target_name: str,
    ) -> None:
        publish(parent_descriptor, source_name, target_name)
        operation_migration.os.rename(
            target_name,
            f".{target_name}.replaced",
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        operation_migration.os.mkdir(
            target_name,
            mode=0o700,
            dir_fd=parent_descriptor,
        )

    monkeypatch.setattr(
        operation_migration,
        "_renameat_exclusive",
        replace_after_publish,
    )

    with pytest.raises(OSError, match="identity changed"):
        apply_v1_migration(plan, target, authority=authority)

    assert target.is_dir()
    assert not (target / "migration-v1.json").exists()
    assert not (target / "operational-v2-activated.json").exists()


def test_verify_v1_parity_rejects_a_tampered_derived_view(tmp_path: Path) -> None:
    source = tmp_path / "v1"
    target = tmp_path / "v2"
    _populate_v1(source)
    authority = _authority(tmp_path / "authority")
    plan = plan_v1_migration(source, device_id=_LOCAL)
    apply_v1_migration(plan, target, authority=authority)
    database = target / "operational.db"

    import sqlite3

    with closing(sqlite3.connect(database)) as connection:
        connection.execute("DELETE FROM focus")
        connection.commit()

    parity = verify_v1_parity(plan, target, authority=authority)

    assert parity.equal is False
    assert "focus/memo" in parity.diff


def test_replay_rejects_a_tampered_prepared_stamp(tmp_path: Path) -> None:
    source = tmp_path / "v1"
    target = tmp_path / "v2"
    _populate_v1(source)
    authority = _authority(tmp_path / "authority")
    plan = plan_v1_migration(source, device_id=_LOCAL)
    apply_v1_migration(plan, target, authority=authority)
    marker = target / "migration-v1.json"
    stamp = json.loads(marker.read_bytes())
    stamp["signature"] = "forged-signature"
    marker.write_bytes(canonical_json_bytes(stamp))

    with pytest.raises(OperationalError):
        apply_v1_migration(plan, target, authority=authority)
