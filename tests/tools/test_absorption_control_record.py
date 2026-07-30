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
from tools.memflow_absorption.schemas import CutoverControlRecord, CutoverState


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
