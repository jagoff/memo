"""Pure value objects and canonical encoding for the operational v2 ledger."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
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
        body = {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
        if isinstance(value, SourceProof) and value.authentication is None:
            body.pop("authentication", None)
        if isinstance(value, MigrationOrigin):
            if not value.source_proof_root_sha256:
                body.pop("source_proof_root_sha256", None)
            if value.source_proof_count == 0:
                body.pop("source_proof_count", None)
        if isinstance(value, OperationalEventV2):
            if value.migration_origin is None:
                body.pop("migration_origin", None)
            if not value.migration_origin_sha256:
                body.pop("migration_origin_sha256", None)
        return body
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bytes):
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical operational mappings require string keys")
        body = {key: _json_value(item) for key, item in value.items()}
        schema = body.get("schema")
        if schema == "memo.operational_event.v2":
            if body.get("migration_origin") is None:
                body.pop("migration_origin", None)
            if not body.get("migration_origin_sha256"):
                body.pop("migration_origin_sha256", None)
        elif schema == "memo.operational_migration_origin.v1":
            if not body.get("source_proof_root_sha256"):
                body.pop("source_proof_root_sha256", None)
            if body.get("source_proof_count") == 0:
                body.pop("source_proof_count", None)
        elif set(body) == {
            "source_system",
            "source_event_id",
            "source_schema",
            "source_origin",
            "source_sequence",
            "source_previous_hash",
            "source_event_hash",
            "source_content_hash",
            "source_actor",
            "source_subject_uri",
            "authentication",
        }:
            if body.get("authentication") is None:
                body.pop("authentication", None)
        return body
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
class SourceProofAuthentication:
    schema: Literal["memo.operational_source_inclusion.v1"]
    source_manifest_sha256: str
    leaf_index: int
    leaf_count: int
    merkle_path: tuple[str, ...]


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
    authentication: SourceProofAuthentication | None = None


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
    source_proof_root_sha256: str = ""
    source_proof_count: int = 0


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
    migration_origin: MigrationOrigin | None = None
    migration_origin_sha256: str = ""


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


@dataclass(frozen=True)
class LedgerRecoveryReport:
    recovered_transactions: tuple[str, ...] = ()
    discarded_transactions: tuple[str, ...] = ()
    repaired_tails: tuple[str, ...] = ()
    repaired_heads: tuple[str, ...] = ()
    recovered_compactions: tuple[str, ...] = ()
    published_targets: int = 0


def operational_wire_dict(value: object) -> dict[str, object]:
    """Return the canonical wire projection for one operational record."""
    projected = _json_value(value)
    if not isinstance(projected, dict):
        raise TypeError("operational record must encode as an object")
    return projected


def canonical_event_hash(event: OperationalEventV2) -> str:
    body = operational_wire_dict(event)
    body["event_hash"] = ""
    body["signature"] = ""
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def canonical_anchor_hash(anchor: ChainAnchor) -> str:
    body = operational_wire_dict(anchor)
    body["anchor_hash"] = ""
    body["signature"] = ""
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _source_proof_original_fields(proof: SourceProof) -> dict[str, object]:
    return {
        "source_system": proof.source_system,
        "source_event_id": proof.source_event_id,
        "source_schema": proof.source_schema,
        "source_origin": proof.source_origin,
        "source_sequence": proof.source_sequence,
        "source_previous_hash": proof.source_previous_hash,
        "source_event_hash": proof.source_event_hash,
        "source_content_hash": proof.source_content_hash,
        "source_actor": proof.source_actor,
        "source_subject_uri": proof.source_subject_uri,
    }


def source_proof_leaf_sha256(proof: SourceProof) -> str:
    if not isinstance(proof, SourceProof):
        raise TypeError("source proof is required")
    return hashlib.sha256(
        b"memo-source-proof-leaf-v1\0" + canonical_json_bytes(_source_proof_original_fields(proof))
    ).hexdigest()


def _source_proof_node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"memo-source-proof-node-v1\0" + left + right).digest()


def authenticate_source_proofs(
    proofs: tuple[SourceProof, ...],
    *,
    source_manifest_sha256: str,
) -> tuple[str, tuple[SourceProof, ...]]:
    """Build the exact reusable inclusion tree used by migration producers."""
    if not proofs:
        raise OperationalError(
            OperationalErrorCode.INVALID_EVENT,
            "source proof tree must contain at least one leaf",
            retryable=False,
        )
    if not _SHA256_RE.fullmatch(source_manifest_sha256):
        raise OperationalError(
            OperationalErrorCode.INVALID_EVENT,
            "source proof manifest digest is invalid",
            retryable=False,
        )
    leaves = [bytes.fromhex(source_proof_leaf_sha256(proof)) for proof in proofs]
    paths: list[list[str]] = [[] for _ in proofs]
    positions = list(range(len(proofs)))
    width = len(leaves)
    level = leaves
    while width > 1:
        for leaf_index, position in enumerate(positions):
            sibling_index = position ^ 1
            sibling = level[position] if sibling_index >= width else level[sibling_index]
            paths[leaf_index].append(sibling.hex())
            positions[leaf_index] //= 2
        next_level: list[bytes] = []
        for index in range(0, width, 2):
            left = level[index]
            right = level[index + 1] if index + 1 < width else left
            next_level.append(_source_proof_node(left, right))
        level = next_level
        width = len(level)
    root = level[0].hex()
    authenticated = tuple(
        replace(
            proof,
            authentication=SourceProofAuthentication(
                schema="memo.operational_source_inclusion.v1",
                source_manifest_sha256=source_manifest_sha256,
                leaf_index=index,
                leaf_count=len(proofs),
                merkle_path=tuple(paths[index]),
            ),
        )
        for index, proof in enumerate(proofs)
    )
    return root, authenticated


def verify_source_proof_inclusion(
    proof: SourceProof,
    *,
    expected_root_sha256: str,
    expected_count: int,
    expected_manifest_sha256: str,
) -> None:
    authentication = proof.authentication
    if authentication is None or authentication.schema != "memo.operational_source_inclusion.v1":
        raise OperationalError(
            OperationalErrorCode.INVALID_EVENT,
            "source proof lacks authenticated inclusion",
            retryable=False,
        )
    if (
        not _SHA256_RE.fullmatch(expected_root_sha256)
        or not _SHA256_RE.fullmatch(expected_manifest_sha256)
        or authentication.source_manifest_sha256 != expected_manifest_sha256
        or authentication.leaf_count != expected_count
        or expected_count < 1
        or authentication.leaf_index < 0
        or authentication.leaf_index >= expected_count
    ):
        raise OperationalError(
            OperationalErrorCode.INVALID_EVENT,
            "source proof manifest, count, or inclusion index mismatch",
            retryable=False,
        )
    current = bytes.fromhex(source_proof_leaf_sha256(proof))
    index = authentication.leaf_index
    width = expected_count
    path_index = 0
    while width > 1:
        if path_index >= len(authentication.merkle_path):
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                "source proof inclusion path is incomplete",
                retryable=False,
            )
        sibling_text = authentication.merkle_path[path_index]
        if not _SHA256_RE.fullmatch(sibling_text):
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                "source proof inclusion path digest is invalid",
                retryable=False,
            )
        sibling = bytes.fromhex(sibling_text)
        if (index ^ 1) >= width and sibling != current:
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                "source proof odd-level duplicate is invalid",
                retryable=False,
            )
        current = (
            _source_proof_node(current, sibling)
            if index % 2 == 0
            else _source_proof_node(sibling, current)
        )
        index //= 2
        width = (width + 1) // 2
        path_index += 1
    if path_index != len(authentication.merkle_path) or current.hex() != expected_root_sha256:
        raise OperationalError(
            OperationalErrorCode.INVALID_EVENT,
            "source proof inclusion root mismatch",
            retryable=False,
        )


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
        or not _SHA256_RE.fullmatch(origin.source_proof_root_sha256)
        or origin.source_proof_count < 1
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
        algorithm=roster.key(origin.attestor_key_id).algorithm,
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
    checkpoint: bytes,
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
    if (
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
        or anchor.checkpoint_sha256 != hashlib.sha256(EMPTY_REDUCER_STATE_BYTES).hexdigest()
        or (checkpoint is not None and checkpoint != EMPTY_REDUCER_STATE_BYTES)
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
    if anchor.kind == "memo_v1" and not _SHA256_RE.fullmatch(anchor.source_manifest_sha256):
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
                algorithm=roster.key(anchor.key_id).algorithm,
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
    if event.source_proof is None:
        if event.migration_origin is not None or event.migration_origin_sha256:
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                "migration origin requires a source proof",
                retryable=False,
            )
    else:
        if event.migration_origin is None:
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                "source proof requires authenticated migration authority",
                retryable=False,
            )
        migration_digest = hashlib.sha256(canonical_json_bytes(event.migration_origin)).hexdigest()
        if (
            event.migration_origin_sha256 != migration_digest
            or event.origin_device != event.migration_origin.migration_device_id
        ):
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                "persisted migration origin digest or device mismatch",
                retryable=False,
            )
        verify_source_proof_inclusion(
            event.source_proof,
            expected_root_sha256=event.migration_origin.source_proof_root_sha256,
            expected_count=event.migration_origin.source_proof_count,
            expected_manifest_sha256=event.migration_origin.source_manifest_sha256,
        )
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
    "LedgerRecoveryReport",
    "MigrationOrigin",
    "MigrationPreparedStamp",
    "OperationalCommand",
    "OperationalEventV2",
    "OriginBundle",
    "OriginPosition",
    "SessionCheckpoint",
    "SourceProof",
    "SourceProofAuthentication",
    "StateCheckpoint",
    "VerificationReport",
    "authenticate_source_proofs",
    "canonical_anchor_hash",
    "canonical_event_hash",
    "canonical_json_bytes",
    "canonical_signed_bytes",
    "operational_wire_dict",
    "source_proof_leaf_sha256",
    "validate_anchor",
    "validate_event",
    "validate_migration_origin",
    "verify_source_proof_inclusion",
]
