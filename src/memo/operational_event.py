"""Pure value objects and canonical encoding for the operational v2 ledger."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Literal

from memo.errors import OperationalError, OperationalErrorCode
from memo.identity import PrincipalIdentity
from memo.operational_event_types import validate_event_payload
from memo.operational_signing import SignatureEnvelope


def _json_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bytes):
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_signed_bytes(value: object) -> bytes:
    body = _json_value(value)
    if not isinstance(body, dict):
        raise TypeError("signed operational record must encode as an object")
    body["signature"] = ""
    return canonical_json_bytes(body)


@dataclass(frozen=True)
class SourceProof:
    source_system: str
    source_event_id: str
    source_schema: str
    source_origin: str
    source_sequence: int
    source_previous_hash: str
    source_event_hash: str
    source_content_hash: str
    source_actor: Mapping[str, object]
    source_subject_uri: str


@dataclass(frozen=True)
class MigrationOrigin:
    schema: Literal["memo.operational_migration_origin.v1"]
    attempt_id: str
    migration_device_id: str
    source_manifest_sha256: str
    capability_manifest_sha256: str
    attestor_device_id: str
    attestor_key_id: str
    roster_version: int
    issued_at: str
    expires_at: str
    signature: str


@dataclass(frozen=True)
class EpochMarkerAuthorization:
    schema: Literal["memo.operational_epoch_authorization.v1"]
    attempt_id: str
    device_id: str
    epoch: int
    control_oid: str
    artifact_digests: Mapping[str, str]
    roster_version: int
    key_id: str
    signature: SignatureEnvelope


@dataclass(frozen=True)
class ChainAnchor:
    schema: Literal["memo.operational_anchor.v1"]
    anchor_id: str
    origin_device: str
    ledger_epoch: int
    reducer_version: int
    kind: Literal["empty", "memo_v1", "compaction"]
    base_sequence: int
    base_event_hash: str
    final_sequence: int
    final_event_hash: str
    previous_anchor_hash: str
    source_manifest_sha256: str
    state_sha256: str
    checkpoint_id: str
    checkpoint_sha256: str
    checkpoint_size: int
    created_at: str
    anchor_hash: str
    roster_version: int
    signer_role: Literal["origin", "migration_attestor"]
    attested_origin: str
    key_id: str
    signature: str


@dataclass(frozen=True)
class OperationalCommand:
    event_type: str
    actor: PrincipalIdentity
    target_id: str | None
    project: str
    workspace: str
    expires_at: str | None
    visibility: str
    idempotency_key: str
    caused_by: tuple[str, ...]
    subject_uri: str
    trace_id: str
    payload: Mapping[str, object]
    source_proof: SourceProof | None = None


@dataclass(frozen=True)
class OperationalEventV2:
    schema: Literal["memo.operational_event.v2"]
    schema_version: Literal[2]
    event_id: str
    event_type: str
    actor: PrincipalIdentity
    target_id: str | None
    project: str
    workspace: str
    origin_device: str
    origin_sequence: int
    logical_clock: str
    authority_epoch: int
    control_oid: str
    created_at: str
    expires_at: str | None
    visibility: str
    idempotency_key: str
    caused_by: tuple[str, ...]
    subject_uri: str
    trace_id: str
    payload: Mapping[str, object]
    content_hash: str
    previous_hash: str
    event_hash: str
    source_proof: SourceProof | None
    roster_version: int
    key_id: str
    signature: str


@dataclass(frozen=True)
class CommandResult:
    event: OperationalEventV2
    replayed: bool
    result: Mapping[str, object]


@dataclass(frozen=True)
class OriginPosition:
    origin_device: str
    sequence: int
    event_hash: str
    anchor_hash: str


@dataclass(frozen=True)
class OriginBundle:
    anchor: ChainAnchor
    checkpoint: bytes
    events: tuple[OperationalEventV2, ...]
    head_sequence: int
    head_hash: str


@dataclass(frozen=True)
class StateCheckpoint:
    schema: Literal["memo.operational_checkpoint.v1"]
    checkpoint_id: str
    reducer_version: int
    origin_device: str
    through_sequence: int
    through_event_hash: str
    state_bytes: bytes
    state_sha256: str
    created_at: str


@dataclass(frozen=True)
class SessionCheckpoint:
    session_id: str
    principal_id: str
    project: str
    workspace: str
    status: Literal["active", "recoverable", "terminated"]
    branch: str | None
    head: str | None
    summary: str
    checkpointed_at: str
    source_event_id: str


@dataclass(frozen=True)
class MigrationPreparedStamp:
    schema: Literal["memo.operational_migration_prepared.v1"]
    source_manifest_sha256: str
    target_generation_sha256: str
    parity_report_sha256: str
    attestor_key_id: str
    signature: str


@dataclass(frozen=True)
class LedgerImportReport:
    manifest_sha256: str
    origins_seen: tuple[str, ...]
    events_inserted: int
    events_replayed: int
    quarantined: tuple[str, ...]
    final_positions: tuple[OriginPosition, ...]


@dataclass(frozen=True)
class VerificationReport:
    ok: bool
    checked_origins: tuple[str, ...]
    checked_events: int
    state_sha256: str
    errors: tuple[str, ...]


def canonical_event_hash(event: OperationalEventV2) -> str:
    body = asdict(event)
    body["event_hash"] = ""
    body["signature"] = ""
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def canonical_anchor_hash(anchor: ChainAnchor) -> str:
    body = asdict(anchor)
    body["anchor_hash"] = ""
    body["signature"] = ""
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def validate_anchor(anchor: ChainAnchor) -> None:
    if anchor.schema != "memo.operational_anchor.v1":
        raise OperationalError(
            OperationalErrorCode.UNKNOWN_SCHEMA,
            f"unsupported operational anchor schema: {anchor.schema}",
            retryable=False,
        )
    if anchor.kind in {"empty", "compaction"}:
        if anchor.signer_role != "origin":
            raise OperationalError(
                OperationalErrorCode.ANCHOR_CONFLICT,
                f"{anchor.kind} anchor requires an origin signer",
                retryable=False,
            )
    elif anchor.kind == "memo_v1":
        if (
            anchor.signer_role != "migration_attestor"
            or anchor.attested_origin != anchor.origin_device
        ):
            raise OperationalError(
                OperationalErrorCode.ANCHOR_CONFLICT,
                "memo_v1 anchor requires a migration attestor for its legacy origin",
                retryable=False,
            )
    else:
        raise OperationalError(
            OperationalErrorCode.ANCHOR_CONFLICT,
            f"unsupported operational anchor kind: {anchor.kind}",
            retryable=False,
        )
    if anchor.base_sequence < 0 or anchor.final_sequence < anchor.base_sequence:
        raise OperationalError(
            OperationalErrorCode.ANCHOR_CONFLICT,
            "anchor sequence range is invalid",
            retryable=False,
        )
    if canonical_anchor_hash(anchor) != anchor.anchor_hash:
        raise OperationalError(
            OperationalErrorCode.ANCHOR_CONFLICT,
            "anchor hash mismatch",
            retryable=False,
        )


def validate_event(event: OperationalEventV2) -> None:
    if event.schema != "memo.operational_event.v2" or event.schema_version != 2:
        raise OperationalError(
            OperationalErrorCode.UNKNOWN_SCHEMA,
            f"unsupported operational schema: {event.schema}/{event.schema_version}",
            retryable=False,
        )
    if event.origin_sequence < 1 or not event.idempotency_key:
        raise OperationalError(
            OperationalErrorCode.INVALID_EVENT,
            "sequence must be positive and idempotency_key must be non-empty",
            retryable=False,
        )
    validate_event_payload(event.event_type, event.payload)
    if canonical_event_hash(event) != event.event_hash:
        raise OperationalError(
            OperationalErrorCode.INVALID_EVENT,
            "event hash mismatch",
            retryable=False,
        )


__all__ = [
    "ChainAnchor",
    "CommandResult",
    "EpochMarkerAuthorization",
    "LedgerImportReport",
    "MigrationOrigin",
    "MigrationPreparedStamp",
    "OperationalCommand",
    "OperationalEventV2",
    "OriginBundle",
    "OriginPosition",
    "SessionCheckpoint",
    "SourceProof",
    "StateCheckpoint",
    "VerificationReport",
    "canonical_anchor_hash",
    "canonical_event_hash",
    "canonical_json_bytes",
    "canonical_signed_bytes",
    "validate_anchor",
    "validate_event",
]
