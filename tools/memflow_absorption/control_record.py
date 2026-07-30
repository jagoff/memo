"""Signed, fresh-OID cutover control-record verification."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from typing import Literal

from memo.errors import SignatureError
from memo.operational_event import canonical_json_bytes
from memo.operational_roster import VerificationRoster
from memo.operational_signing import OperationalSigner, OperationalVerifier
from tools.memflow_absorption.schemas import (
    CapabilityManifest,
    ConsumerReplacementPlan,
    CutoverControlRecord,
    CutoverState,
    SynapseRetirementState,
    VerifiedControlRecord,
)

CONTROL_RECORD_DOMAIN = "memo.cutover.control_record.v1"


class ControlRecordError(RuntimeError):
    """A cutover control record is stale, malformed, or unauthenticated."""


class CutoverSafetyError(RuntimeError):
    """A Synapse retirement request would cross the committed safety fence."""


_SHA256_LENGTH = 64
_REQUIRED_SURFACES = frozenset(
    {
        "process",
        "port",
        "launchagent",
        "mcp_gateway_route",
        "shell_config_path",
        "state_root",
    }
)
_SYNAPSE_TRANSITIONS = {
    SynapseRetirementState.READY: SynapseRetirementState.QUIESCED,
    SynapseRetirementState.QUIESCED: SynapseRetirementState.STAGED,
    SynapseRetirementState.COMMITTED: SynapseRetirementState.VERIFIED,
}
_CUTOVER_STATE_BY_SYNAPSE_STATE = {
    SynapseRetirementState.PREPARING: CutoverState.PREPARING,
    SynapseRetirementState.READY: CutoverState.READY,
    SynapseRetirementState.QUIESCED: CutoverState.QUIESCED,
    SynapseRetirementState.STAGED: CutoverState.STAGED,
    SynapseRetirementState.COMMITTED: CutoverState.EPOCH_COMMITTED,
    SynapseRetirementState.VERIFIED: CutoverState.VERIFIED,
    SynapseRetirementState.ABORTED: CutoverState.ABORTED,
}


def _valid_oid(value: str, *, allow_empty: bool = False) -> bool:
    return (allow_empty and value == "") or (
        len(value) == 40 and all(character in "0123456789abcdef" for character in value)
    )


def _valid_sha256(value: str, *, allow_empty: bool = False) -> bool:
    return (allow_empty and value == "") or (
        len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_peer_votes(votes: tuple[str, ...]) -> None:
    if (
        len(votes) != 2
        or votes != tuple(sorted(set(votes)))
        or any(not _valid_sha256(vote) for vote in votes)
    ):
        raise CutoverSafetyError(
            "Synapse retirement requires two distinct peer vote SHA-256 values"
        )


def _validate_authority_digests(record: CutoverControlRecord) -> None:
    if not _valid_sha256(record.capability_manifest_sha256):
        raise ControlRecordError("Synapse capability-manifest digest is missing or invalid")
    if not _valid_sha256(record.consumer_plan_sha256):
        raise ControlRecordError("Synapse consumer-plan digest is missing or invalid")
    authority_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "capability_manifest_sha256": record.capability_manifest_sha256,
                "consumer_plan_sha256": record.consumer_plan_sha256,
            }
        )
    ).hexdigest()
    if authority_sha256 != record.synapse_authority_sha256:
        raise ControlRecordError("Synapse authority digests changed after preflight")


def _validate_aborted_synapse_fields(record: CutoverControlRecord) -> None:
    if record.retirement_epoch or record.independence_receipt_sha256:
        raise ControlRecordError("ABORTED Synapse control contains committed evidence")
    has_authority = bool(
        record.capability_manifest_sha256
        or record.consumer_plan_sha256
        or record.synapse_authority_sha256
    )
    if has_authority:
        _validate_authority_digests(record)
    if record.peer_vote_sha256:
        try:
            _validate_peer_votes(record.peer_vote_sha256)
        except CutoverSafetyError as exc:
            raise ControlRecordError(str(exc)) from exc
    has_staging = bool(
        record.synapse_manifest_sha256 or record.active_state_receipt_sha256
    )
    if has_staging and (
        not _valid_sha256(record.synapse_manifest_sha256)
        or not _valid_sha256(record.active_state_receipt_sha256)
    ):
        raise ControlRecordError("ABORTED Synapse control has incomplete staging evidence")
    if (record.peer_vote_sha256 or has_staging) and not has_authority:
        raise ControlRecordError("ABORTED Synapse control lacks preflight authority")
    if has_staging and not record.peer_vote_sha256:
        raise ControlRecordError("ABORTED Synapse control lacks peer votes")


def _validate_synapse_fields(record: CutoverControlRecord) -> None:
    state = record.synapse_state
    if record.state is not _CUTOVER_STATE_BY_SYNAPSE_STATE[state]:
        raise ControlRecordError("coordinated and Synapse cutover states disagree")
    if state is SynapseRetirementState.ABORTED:
        _validate_aborted_synapse_fields(record)
        return
    if state is SynapseRetirementState.PREPARING:
        if any(
            (
                record.synapse_manifest_sha256,
                record.capability_manifest_sha256,
                record.consumer_plan_sha256,
                record.synapse_authority_sha256,
                record.active_state_receipt_sha256,
                record.peer_vote_sha256,
                record.retirement_epoch,
                record.independence_receipt_sha256,
            )
        ):
            raise ControlRecordError("PREPARING Synapse control contains committed evidence")
        return
    _validate_authority_digests(record)
    if state in {
        SynapseRetirementState.QUIESCED,
        SynapseRetirementState.STAGED,
        SynapseRetirementState.COMMITTED,
        SynapseRetirementState.VERIFIED,
    }:
        try:
            _validate_peer_votes(record.peer_vote_sha256)
        except CutoverSafetyError as exc:
            raise ControlRecordError(str(exc)) from exc
    elif record.peer_vote_sha256:
        raise ControlRecordError("Synapse peer votes arrived before quiescence")
    if state in {
        SynapseRetirementState.STAGED,
        SynapseRetirementState.COMMITTED,
        SynapseRetirementState.VERIFIED,
    }:
        if not _valid_sha256(record.synapse_manifest_sha256):
            raise ControlRecordError("signed Synapse retirement manifest digest is missing")
        if not _valid_sha256(record.active_state_receipt_sha256):
            raise ControlRecordError("active-state migration receipt digest is missing")
    elif record.active_state_receipt_sha256 or record.synapse_manifest_sha256:
        raise ControlRecordError("staging evidence arrived before staging")
    if state in {
        SynapseRetirementState.COMMITTED,
        SynapseRetirementState.VERIFIED,
    }:
        if record.retirement_epoch < 1:
            raise ControlRecordError("Synapse retirement epoch is missing")
    elif record.retirement_epoch:
        raise ControlRecordError("Synapse retirement epoch was committed early")
    if state is SynapseRetirementState.VERIFIED:
        if not _valid_sha256(record.independence_receipt_sha256):
            raise ControlRecordError("Synapse independence receipt digest is missing")
    elif record.independence_receipt_sha256:
        raise ControlRecordError("Synapse independence receipt arrived before verification")


def _manifest_digest(manifest: CapabilityManifest) -> str:
    if (
        manifest.schema != "memo.cutover_capability_manifest.v1"
        or not manifest.frozen
        or manifest.blockers
        or not manifest.signature
    ):
        raise CutoverSafetyError("Synapse capability manifest is not frozen and signed")
    if (
        hashlib.sha256(manifest.operation_map_bytes()).hexdigest()
        != manifest.operation_map_sha256
        or hashlib.sha256(manifest.slo_baseline_bytes()).hexdigest()
        != manifest.slo_baseline_sha256
    ):
        raise CutoverSafetyError("Synapse capability manifest digest mismatch")
    if (
        len(manifest.machine_ids) != 2
        or manifest.machine_ids != tuple(sorted(set(manifest.machine_ids)))
    ):
        raise CutoverSafetyError("Synapse capability manifest lacks exact two-peer authority")
    return hashlib.sha256(manifest.signed_bytes()).hexdigest()


def _consumer_plan_digest(plan: ConsumerReplacementPlan) -> str:
    row_digest = hashlib.sha256(
        canonical_json_bytes([row.to_dict() for row in plan.rows])
    ).hexdigest()
    if row_digest != plan.digest:
        raise CutoverSafetyError("Synapse consumer-plan digest mismatch")
    if set(plan.covered_surfaces) != _REQUIRED_SURFACES:
        missing = sorted(_REQUIRED_SURFACES - set(plan.covered_surfaces))
        extra = sorted(set(plan.covered_surfaces) - _REQUIRED_SURFACES)
        detail = ",".join((*missing, *(f"unexpected:{item}" for item in extra)))
        raise CutoverSafetyError(f"Synapse preflight surface coverage is incomplete: {detail}")
    if any(
        not values
        or values != tuple(sorted(set(values)))
        or any(not value for value in values)
        for values in plan.covered_surfaces.values()
    ):
        raise CutoverSafetyError("Synapse preflight surface coverage is ambiguous")
    return hashlib.sha256(plan.authority_bytes()).hexdigest()


def prepare_synapse_retirement(
    control: CutoverControlRecord,
    manifest: CapabilityManifest,
    consumer_plan: ConsumerReplacementPlan,
) -> CutoverControlRecord:
    """Bind all signed preflight authority before quiescing either product."""

    if control.synapse_state in {
        SynapseRetirementState.COMMITTED,
        SynapseRetirementState.VERIFIED,
    }:
        raise CutoverSafetyError("synapse.cutover.retired")
    if control.synapse_state is not SynapseRetirementState.PREPARING:
        raise CutoverSafetyError("stale Synapse cutover attempt")
    _validate_synapse_fields(control)
    manifest_sha256 = _manifest_digest(manifest)
    consumer_plan_sha256 = _consumer_plan_digest(consumer_plan)
    authority_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "capability_manifest_sha256": manifest_sha256,
                "consumer_plan_sha256": consumer_plan_sha256,
            }
        )
    ).hexdigest()
    if control.signature == "":
        raise CutoverSafetyError("Synapse preflight requires a signed control record")
    return replace(
        control,
        state=CutoverState.READY,
        synapse_state=SynapseRetirementState.READY,
        capability_manifest_sha256=manifest_sha256,
        consumer_plan_sha256=consumer_plan_sha256,
        synapse_authority_sha256=authority_sha256,
        signature="",
    )


def advance_synapse_retirement(
    control: CutoverControlRecord,
    target: SynapseRetirementState,
    *,
    peer_vote_sha256: tuple[str, ...] = (),
    active_state_receipt_sha256: str = "",
    synapse_manifest_sha256: str = "",
    independence_receipt_sha256: str = "",
) -> CutoverControlRecord:
    """Advance one edge only; evidence digests can never be replaced."""

    if not control.signature:
        raise CutoverSafetyError("Synapse retirement transition requires a signed control record")
    _validate_synapse_fields(control)
    if target is SynapseRetirementState.ABORTED:
        if control.synapse_state in {
            SynapseRetirementState.COMMITTED,
            SynapseRetirementState.VERIFIED,
            SynapseRetirementState.ABORTED,
        }:
            raise CutoverSafetyError("Synapse retirement cannot abort after commit")
        return replace(
            control,
            state=CutoverState.ABORTED,
            synapse_state=target,
            signature="",
        )
    if _SYNAPSE_TRANSITIONS.get(control.synapse_state) is not target:
        raise CutoverSafetyError("stale or skipped Synapse retirement transition")
    if target is SynapseRetirementState.QUIESCED:
        _validate_peer_votes(peer_vote_sha256)
        advanced = replace(
            control,
            state=CutoverState.QUIESCED,
            synapse_state=target,
            peer_vote_sha256=peer_vote_sha256,
            signature="",
        )
    elif target is SynapseRetirementState.STAGED:
        if not _valid_sha256(active_state_receipt_sha256):
            raise CutoverSafetyError("active-state migration receipt digest is missing")
        if not _valid_sha256(synapse_manifest_sha256):
            raise CutoverSafetyError("signed Synapse retirement manifest digest is missing")
        advanced = replace(
            control,
            state=CutoverState.STAGED,
            synapse_state=target,
            active_state_receipt_sha256=active_state_receipt_sha256,
            synapse_manifest_sha256=synapse_manifest_sha256,
            signature="",
        )
    else:
        assert target is SynapseRetirementState.VERIFIED
        if not _valid_sha256(independence_receipt_sha256):
            raise CutoverSafetyError("Synapse independence receipt digest is missing")
        advanced = replace(
            control,
            state=CutoverState.VERIFIED,
            synapse_state=target,
            independence_receipt_sha256=independence_receipt_sha256,
            signature="",
        )
    _validate_synapse_fields(advanced)
    return advanced


def _verified_from_record(record: CutoverControlRecord) -> VerifiedControlRecord:
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
        attempt_id=record.attempt_id,
        synapse_state=record.synapse_state,
        capability_manifest_sha256=record.capability_manifest_sha256,
        synapse_manifest_sha256=record.synapse_manifest_sha256,
        consumer_plan_sha256=record.consumer_plan_sha256,
        synapse_authority_sha256=record.synapse_authority_sha256,
        active_state_receipt_sha256=record.active_state_receipt_sha256,
        peer_vote_sha256=record.peer_vote_sha256,
        retirement_epoch=record.retirement_epoch,
        independence_receipt_sha256=record.independence_receipt_sha256,
    )


def commit_synapse_activation(
    control: CutoverControlRecord,
    epoch: int,
) -> VerifiedControlRecord:
    """Create the immutable retired fence view without activating any service."""

    if control.retirement_epoch:
        raise CutoverSafetyError("second Synapse activation epoch is forbidden")
    if epoch < 1:
        raise CutoverSafetyError("Synapse activation epoch must be positive")
    if control.synapse_state is not SynapseRetirementState.STAGED:
        raise CutoverSafetyError("stale or skipped Synapse retirement transition")
    if not control.signature:
        raise CutoverSafetyError("Synapse activation requires a signed control record")
    _validate_synapse_fields(control)
    committed = replace(
        control,
        state=CutoverState.EPOCH_COMMITTED,
        synapse_state=SynapseRetirementState.COMMITTED,
        retirement_epoch=epoch,
        signature="",
    )
    _validate_synapse_fields(committed)
    return _verified_from_record(committed)


def validate_synapse_request(
    control: VerifiedControlRecord,
    epoch: int,
    *,
    kind: Literal["status", "startup", "write", "fallback"] = "write",
) -> None:
    """Keep status readable while rejecting stale or retired runtime activity."""

    if kind == "status":
        return
    if control.retirement_epoch and epoch != control.retirement_epoch:
        raise CutoverSafetyError("stale activation epoch")
    if control.synapse_state in {
        SynapseRetirementState.COMMITTED,
        SynapseRetirementState.VERIFIED,
    }:
        raise CutoverSafetyError("synapse.cutover.retired")


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
    _validate_synapse_fields(record)
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
    _validate_synapse_fields(record)
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
        attempt_id=record.attempt_id,
        synapse_state=record.synapse_state,
        capability_manifest_sha256=record.capability_manifest_sha256,
        synapse_manifest_sha256=record.synapse_manifest_sha256,
        consumer_plan_sha256=record.consumer_plan_sha256,
        synapse_authority_sha256=record.synapse_authority_sha256,
        active_state_receipt_sha256=record.active_state_receipt_sha256,
        peer_vote_sha256=record.peer_vote_sha256,
        retirement_epoch=record.retirement_epoch,
        independence_receipt_sha256=record.independence_receipt_sha256,
    )


__all__ = [
    "CONTROL_RECORD_DOMAIN",
    "ControlRecordError",
    "CutoverSafetyError",
    "advance_synapse_retirement",
    "commit_synapse_activation",
    "prepare_synapse_retirement",
    "sign_control_record",
    "validate_synapse_request",
    "verify_control_record",
]
