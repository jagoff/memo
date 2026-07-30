from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from memo.operational_key_store import (
    AuthorityPinStore,
    DeviceKeyStore,
    InMemoryAuthorityPinProvider,
)
from memo.operational_roster import VerificationRoster
from memo.operational_signing import OperationalSigner
from tools.memflow_absorption.control_record import (
    ControlRecordError,
    sign_control_record,
    verify_control_record,
)
from tools.memflow_absorption.schemas import (
    CutoverControlRecord,
    CutoverMode,
    CutoverState,
    DrainSnapshot,
    FenceMarker,
)


def _authority(tmp_path: Path) -> tuple[DeviceKeyStore, VerificationRoster]:
    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a")
    pins = AuthorityPinStore._for_test(
        tmp_path,
        provider=InMemoryAuthorityPinProvider(),
    )
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=key,
        root=tmp_path,
        pin_store=pins,
    )
    return keys, roster


def _record(
    keys: DeviceKeyStore,
    roster: VerificationRoster,
) -> CutoverControlRecord:
    unsigned = CutoverControlRecord(
        schema="memo.cutover_control_record.v1",
        control_oid="b" * 40,
        state=CutoverState.PREPARING,
        sequence=1,
        previous_control_oid="",
        attempt_id="attempt-123",
        roster_version=roster.version,
        signer_device_id="device-a",
        signer_key_id=roster.local_key_id,
        issued_at="2026-07-30T00:00:00Z",
        signature="",
    )
    return sign_control_record(
        unsigned,
        signer=OperationalSigner(keys, roster_version=roster.version),
    )


def test_shared_cutover_states_and_marker_bytes_are_canonical() -> None:
    assert tuple(state.value for state in CutoverState) == (
        "PREPARING",
        "READY",
        "QUIESCING",
        "QUIESCED",
        "STAGED",
        "ACTIVATION_READY",
        "EPOCH_COMMITTED",
        "ACTIVATED",
        "VERIFIED",
        "ABORTING",
        "ABORTED",
        "RETIRED",
    )
    assert tuple(mode.value for mode in CutoverMode) == (
        "active",
        "quiescing",
        "retired",
    )
    marker = FenceMarker(
        schema="memo.cutover_fence.v1",
        attempt_id="attempt-123",
        mode=CutoverMode.QUIESCING,
        epoch=7,
        expected_commit="a" * 40,
        runtime_digest="b" * 64,
        device_id="device-a",
        key_id="key-a",
        issued_at="2026-07-30T00:00:00Z",
        expires_at="2026-07-30T01:00:00Z",
        control_oid="c" * 40,
        control_sequence=2,
        previous_control_oid="d" * 40,
        signature="signed",
    )
    drain = DrainSnapshot(
        schema="memo.cutover_drain_snapshot.v1",
        captured_at="2026-07-30T00:01:00Z",
        requests=0,
        event_append=0,
        delivery=0,
        ack=0,
        cursor=0,
        sync=0,
        git_push=0,
        autonomous_loops=0,
        writable_handles=0,
        inflight_total=0,
        clean=True,
        last_fsync_at="2026-07-30T00:01:00Z",
    )

    assert b'"mode":"quiescing"' in marker.signed_bytes()
    assert b'"signature":""' in marker.signed_bytes()
    assert drain.to_dict()["clean"] is True


def test_control_record_requires_fresh_oid_and_valid_signature(tmp_path: Path) -> None:
    keys, roster = _authority(tmp_path)
    record = _record(keys, roster)

    verified = verify_control_record(
        expected_oid="b" * 40,
        roster=roster,
        record=record,
        fetched_oid="b" * 40,
    )

    assert verified.control_oid == "b" * 40
    assert verified.state is CutoverState.PREPARING
    assert verified.sequence == 1
    assert verified.signer_key_id == roster.local_key_id
    assert verified.canonical_payload
    assert verified.verified_at.endswith("Z")


def test_control_record_rejects_stale_fetch_tamper_and_bad_sequence(tmp_path: Path) -> None:
    keys, roster = _authority(tmp_path)
    record = _record(keys, roster)

    with pytest.raises(ControlRecordError, match="freshly fetched"):
        verify_control_record(
            expected_oid="b" * 40,
            roster=roster,
            record=record,
            fetched_oid="c" * 40,
        )
    with pytest.raises(ControlRecordError, match="signature"):
        verify_control_record(
            expected_oid="b" * 40,
            roster=roster,
            record=replace(record, state=CutoverState.READY),
            fetched_oid="b" * 40,
        )
    with pytest.raises(ControlRecordError, match="sequence"):
        verify_control_record(
            expected_oid="b" * 40,
            roster=roster,
            record=replace(record, sequence=0),
            fetched_oid="b" * 40,
        )


def test_control_record_requires_predecessor_after_first_sequence(tmp_path: Path) -> None:
    keys, roster = _authority(tmp_path)
    record = _record(keys, roster)

    with pytest.raises(ControlRecordError, match="predecessor"):
        sign_control_record(
            replace(record, sequence=2, signature=""),
            signer=OperationalSigner(keys, roster_version=roster.version),
        )
