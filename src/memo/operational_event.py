"""Pure value objects and canonical encoding for the operational v2 ledger."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Literal

from memo.errors import OperationalError, OperationalErrorCode
from memo.identity import PrincipalIdentity
from memo.operational_event_types import validate_event_payload
from memo.operational_signing import OperationalVerifier, SignatureEnvelope

if TYPE_CHECKING:
    from memo.operational_roster import VerificationRoster

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
EMPTY_REDUCER_STATE_BYTES = b"{}"


def _json_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bytes):
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical operational mappings require string keys")
        return {key: _json_value(item) for key, item in value.items()}
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


def _parse_time(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise OperationalError(
            OperationalErrorCode.INVALID_EVENT,
            f"{field} must be an ISO-8601 timestamp",
            retryable=False,
        ) from exc
    if parsed.tzinfo is None:
        raise OperationalError(
            OperationalErrorCode.INVALID_EVENT,
            f"{field} must include a timezone",
            retryable=False,
        )
    return parsed.astimezone(UTC)


def validate_migration_origin(
    origin: MigrationOrigin,
    *,
    roster: VerificationRoster,
    verifier: OperationalVerifier,
    at_time: str | None = None,
) -> None:
    if origin.schema != "memo.operational_migration_origin.v1":
        raise OperationalError(
            OperationalErrorCode.UNKNOWN_SCHEMA,
            f"unsupported migration origin schema: {origin.schema}",
            retryable=False,
        )
    if (
        not origin.attempt_id
        or not origin.migration_device_id
        or not origin.attestor_device_id
        or not _SHA256_RE.fullmatch(origin.source_manifest_sha256)
        or not _SHA256_RE.fullmatch(origin.capability_manifest_sha256)
    ):
        raise OperationalError(
            OperationalErrorCode.INVALID_EVENT,
            "migration origin identity or manifest digest is invalid",
            retryable=False,
        )
    issued = _parse_time(origin.issued_at, "issued_at")
    expires = _parse_time(origin.expires_at, "expires_at")
    observed = _parse_time(at_time, "at_time") if at_time else datetime.now(UTC)
    if expires <= issued or observed < issued or observed >= expires:
        raise OperationalError(
            OperationalErrorCode.EXPIRED,
            "migration origin is outside its validity window",
            retryable=False,
        )
    envelope = SignatureEnvelope(
        algorithm="ed25519",
        key_id=origin.attestor_key_id,
        roster_version=origin.roster_version,
        signature=origin.signature,
    )
    verifier.verify(
        domain="memo.operational.migration_origin.v1",
        payload=canonical_signed_bytes(origin),
        envelope=envelope,
        roster=roster,
    )


def validate_anchor(
    anchor: ChainAnchor,
    *,
    checkpoint: bytes | None = None,
    roster: VerificationRoster | None = None,
    verifier: OperationalVerifier | None = None,
) -> None:
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
    if checkpoint is not None and (
        anchor.checkpoint_size != len(checkpoint)
        or anchor.checkpoint_sha256 != hashlib.sha256(checkpoint).hexdigest()
    ):
        raise OperationalError(
            OperationalErrorCode.ANCHOR_CONFLICT,
            "anchor checkpoint bytes do not match authority metadata",
            retryable=False,
        )
    if anchor.kind in {"empty", "memo_v1"} and (
        anchor.checkpoint_size != len(EMPTY_REDUCER_STATE_BYTES)
        or anchor.checkpoint_sha256
        != hashlib.sha256(EMPTY_REDUCER_STATE_BYTES).hexdigest()
        or (
            checkpoint is not None
            and checkpoint != EMPTY_REDUCER_STATE_BYTES
        )
    ):
        raise OperationalError(
            OperationalErrorCode.ANCHOR_CONFLICT,
            f"{anchor.kind} anchor requires the canonical empty checkpoint",
            retryable=False,
        )
    if anchor.kind == "empty" and (
        anchor.base_sequence != 0
        or anchor.final_sequence != 0
        or anchor.base_event_hash
        or anchor.final_event_hash
        or anchor.source_manifest_sha256
    ):
        raise OperationalError(
            OperationalErrorCode.ANCHOR_CONFLICT,
            "empty anchor cannot authorize prior events or a source manifest",
            retryable=False,
        )
    if anchor.kind == "memo_v1" and not _SHA256_RE.fullmatch(
        anchor.source_manifest_sha256
    ):
        raise OperationalError(
            OperationalErrorCode.ANCHOR_CONFLICT,
            "memo_v1 anchor requires a source manifest digest",
            retryable=False,
        )
    if canonical_anchor_hash(anchor) != anchor.anchor_hash:
        raise OperationalError(
            OperationalErrorCode.ANCHOR_CONFLICT,
            "anchor hash mismatch",
            retryable=False,
        )
    if (roster is None) != (verifier is None):
        raise TypeError("roster and verifier must be supplied together")
    if roster is not None and verifier is not None:
        verifier.verify(
            domain="memo.operational.anchor.v1",
            payload=canonical_signed_bytes(anchor),
            envelope=SignatureEnvelope(
                algorithm="ed25519",
                key_id=anchor.key_id,
                roster_version=anchor.roster_version,
                signature=anchor.signature,
            ),
            roster=roster,
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
    "EMPTY_REDUCER_STATE_BYTES",
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
    "validate_migration_origin",
]
