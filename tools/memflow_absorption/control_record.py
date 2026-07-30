"""Signed, fresh-OID cutover control-record verification."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from memo.errors import SignatureError
from memo.operational_roster import VerificationRoster
from memo.operational_signing import OperationalSigner, OperationalVerifier
from tools.memflow_absorption.schemas import (
    CutoverControlRecord,
    VerifiedControlRecord,
)

CONTROL_RECORD_DOMAIN = "memo.cutover.control_record.v1"


class ControlRecordError(RuntimeError):
    """A cutover control record is stale, malformed, or unauthenticated."""


def _valid_oid(value: str, *, allow_empty: bool = False) -> bool:
    return (allow_empty and value == "") or (
        len(value) == 40 and all(character in "0123456789abcdef" for character in value)
    )


def sign_control_record(
    record: CutoverControlRecord,
    *,
    signer: OperationalSigner,
) -> CutoverControlRecord:
    if record.signature:
        raise ControlRecordError("control record must be unsigned before signing")
    if record.schema != "memo.cutover_control_record.v1":
        raise ControlRecordError("control record schema is invalid")
    if not _valid_oid(record.control_oid):
        raise ControlRecordError("control object id is invalid")
    if record.sequence < 1:
        raise ControlRecordError("control record sequence must be positive")
    if not _valid_oid(record.previous_control_oid, allow_empty=True):
        raise ControlRecordError("previous control object id is invalid")
    if (record.sequence == 1) != (record.previous_control_oid == ""):
        raise ControlRecordError("control record predecessor does not match sequence")
    if not record.attempt_id:
        raise ControlRecordError("control record attempt id is missing")
    envelope = signer.sign(
        domain=CONTROL_RECORD_DOMAIN,
        payload=record.canonical_payload,
        key_id=record.signer_key_id,
    )
    return replace(record, signature=envelope.signature)


def verify_control_record(
    *,
    expected_oid: str,
    roster: VerificationRoster,
    record: CutoverControlRecord,
    fetched_oid: str,
) -> VerifiedControlRecord:
    """Require an exact freshly fetched record before trusting its state."""

    if not _valid_oid(expected_oid) or not _valid_oid(fetched_oid):
        raise ControlRecordError("control object id is invalid")
    if fetched_oid != expected_oid or record.control_oid != expected_oid:
        raise ControlRecordError("control record was not freshly fetched at expected OID")
    if record.schema != "memo.cutover_control_record.v1":
        raise ControlRecordError("control record schema is invalid")
    if record.sequence < 1:
        raise ControlRecordError("control record sequence must be positive")
    if not _valid_oid(record.previous_control_oid, allow_empty=True):
        raise ControlRecordError("previous control object id is invalid")
    if (record.sequence == 1) != (record.previous_control_oid == ""):
        raise ControlRecordError("control record predecessor does not match sequence")
    if not record.attempt_id:
        raise ControlRecordError("control record attempt id is missing")
    try:
        OperationalVerifier().verify(
            domain=CONTROL_RECORD_DOMAIN,
            payload=record.canonical_payload,
            envelope=record.signature_envelope(),
            roster=roster,
        )
    except SignatureError as exc:
        raise ControlRecordError("control record signature is invalid") from exc
    return VerifiedControlRecord(
        control_oid=record.control_oid,
        canonical_payload=record.canonical_payload,
        state=record.state,
        sequence=record.sequence,
        previous_control_oid=record.previous_control_oid,
        roster_version=record.roster_version,
        verified_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        signer_device_id=record.signer_device_id,
        signer_key_id=record.signer_key_id,
    )


__all__ = [
    "CONTROL_RECORD_DOMAIN",
    "ControlRecordError",
    "sign_control_record",
    "verify_control_record",
]
