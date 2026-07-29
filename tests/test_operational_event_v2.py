from __future__ import annotations

from dataclasses import replace

import pytest

from memo.error_contract import OperationalError, OperationalErrorCode
from memo.identity import PrincipalIdentity
from memo.operational_event import (
    MigrationOrigin,
    OperationalEventV2,
    canonical_event_hash,
    validate_event,
)
from memo.operational_event_types import FOCUS_SET


def _event(**changes: object) -> OperationalEventV2:
    identity = PrincipalIdentity(
        principal_id="device-a:session-a",
        actor_id="agent-a",
        kind="agent",
        device_id="device-a",
        session_id="session-a",
        source_client="codex",
    )
    event = OperationalEventV2(
        schema="memo.operational_event.v2",
        schema_version=2,
        event_id="event-1",
        event_type=FOCUS_SET,
        actor=identity,
        target_id=None,
        project="demo",
        workspace="/tmp/demo",
        origin_device="device-a",
        origin_sequence=1,
        logical_clock="1",
        authority_epoch=0,
        control_oid="control-0",
        created_at="2026-07-29T12:00:00Z",
        expires_at=None,
        visibility="owner",
        idempotency_key="idem-1",
        caused_by=(),
        subject_uri="memo://focus/demo",
        trace_id="trace-1",
        payload={"b": 2, "a": 1},
        content_hash="content",
        previous_hash="",
        event_hash="",
        source_proof=None,
        roster_version=1,
        key_id="key-1",
        signature="sig",
    )
    event = replace(event, **changes)
    return replace(event, event_hash=canonical_event_hash(event))


def test_v2_event_hash_is_canonical_and_tamper_evident() -> None:
    first = _event(payload={"b": 2, "a": 1})
    same = _event(payload={"a": 1, "b": 2})
    assert first.event_hash == same.event_hash
    changed = replace(first, payload={"a": 9, "b": 2})
    assert canonical_event_hash(changed) != first.event_hash
    validate_event(first)


def test_v2_validation_rejects_schema_sequence_and_hash() -> None:
    with pytest.raises(OperationalError) as exc:
        validate_event(_event(schema="memo.operational_event.v9"))
    assert exc.value.code is OperationalErrorCode.UNKNOWN_SCHEMA
    with pytest.raises(OperationalError):
        validate_event(_event(origin_sequence=0))
    with pytest.raises(OperationalError):
        validate_event(replace(_event(), event_hash="bad"))


def test_migration_origin_uses_normative_migration_device_id() -> None:
    origin = MigrationOrigin(
        schema="memo.operational_migration_origin.v1",
        attempt_id="attempt-1",
        migration_device_id="migration-device",
        source_manifest_sha256="a" * 64,
        capability_manifest_sha256="b" * 64,
        attestor_device_id="device-a",
        attestor_key_id="key-1",
        roster_version=1,
        issued_at="2026-07-29T12:00:00Z",
        expires_at="2026-07-29T13:00:00Z",
        signature="sig",
    )
    assert origin.migration_device_id == "migration-device"
    assert not hasattr(origin, "device_id")
