"""Signed, fresh-OID cutover control-record verification."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from memo.errors import SignatureError
from memo.operational_event import canonical_json_bytes
from memo.operational_roster import VerificationRoster
from memo.operational_signing import OperationalSigner, OperationalVerifier
from tools.memflow_absorption.consumer_migration import (
    ConsumerMigrationError,
    build_consumer_replacement_plan,
)
from tools.memflow_absorption.inventory import (
    InventoryError,
    verify_consumer_inventory,
    verify_synapse_retirement_manifest,
)
from tools.memflow_absorption.manifest import ManifestError, verify_capability_manifest
from tools.memflow_absorption.schemas import (
    CapabilityManifest,
    ConsumerInventory,
    ConsumerReplacementPlan,
    CutoverControlRecord,
    CutoverState,
    IndependenceReceipt,
    IndependenceScanReceipt,
    SynapsePeerVote,
    SynapseRetirementManifest,
    SynapseRetirementState,
    VerifiedControlRecord,
)

CONTROL_RECORD_DOMAIN = "memo.cutover.control_record.v1"
SYNAPSE_PEER_VOTE_DOMAIN = "memo.cutover.vote.v1"


class ControlRecordError(RuntimeError):
    """A cutover control record is stale, malformed, or unauthenticated."""


class CutoverSafetyError(RuntimeError):
    """A Synapse retirement request would cross the committed safety fence."""


def control_record_from_dict(value: Mapping[str, Any]) -> CutoverControlRecord:
    """Parse the exact signed control-record schema without coercion."""

    expected = {
        "schema",
        "control_oid",
        "state",
        "sequence",
        "previous_control_oid",
        "attempt_id",
        "roster_version",
        "signer_device_id",
        "signer_key_id",
        "issued_at",
        "signature",
        "synapse_state",
        "capability_manifest_sha256",
        "synapse_manifest_sha256",
        "consumer_plan_sha256",
        "synapse_authority_sha256",
        "active_state_receipt_sha256",
        "peer_vote_sha256",
        "peer_device_ids",
        "retirement_epoch",
        "independence_receipt_sha256",
    }
    string_fields = expected - {
        "sequence",
        "roster_version",
        "peer_vote_sha256",
        "peer_device_ids",
        "retirement_epoch",
    }
    if (
        set(value) != expected
        or any(not isinstance(value.get(field), str) for field in string_fields)
        or any(
            isinstance(value.get(field), bool) or not isinstance(value.get(field), int)
            for field in ("sequence", "roster_version", "retirement_epoch")
        )
        or any(
            not isinstance(value.get(field), list)
            or any(not isinstance(item, str) for item in value[field])
            for field in ("peer_vote_sha256", "peer_device_ids")
        )
    ):
        raise ControlRecordError("control record fields are invalid")
    try:
        state = CutoverState(cast(str, value["state"]))
        synapse_state = SynapseRetirementState(cast(str, value["synapse_state"]))
    except ValueError as exc:
        raise ControlRecordError("control record state is invalid") from exc
    return CutoverControlRecord(
        schema=cast(Any, value["schema"]),
        control_oid=cast(str, value["control_oid"]),
        state=state,
        sequence=cast(int, value["sequence"]),
        previous_control_oid=cast(str, value["previous_control_oid"]),
        attempt_id=cast(str, value["attempt_id"]),
        roster_version=cast(int, value["roster_version"]),
        signer_device_id=cast(str, value["signer_device_id"]),
        signer_key_id=cast(str, value["signer_key_id"]),
        issued_at=cast(str, value["issued_at"]),
        signature=cast(str, value["signature"]),
        synapse_state=synapse_state,
        capability_manifest_sha256=cast(str, value["capability_manifest_sha256"]),
        synapse_manifest_sha256=cast(str, value["synapse_manifest_sha256"]),
        consumer_plan_sha256=cast(str, value["consumer_plan_sha256"]),
        synapse_authority_sha256=cast(str, value["synapse_authority_sha256"]),
        active_state_receipt_sha256=cast(str, value["active_state_receipt_sha256"]),
        peer_vote_sha256=tuple(value["peer_vote_sha256"]),
        peer_device_ids=tuple(value["peer_device_ids"]),
        retirement_epoch=cast(int, value["retirement_epoch"]),
        independence_receipt_sha256=cast(str, value["independence_receipt_sha256"]),
    )


class ControlRecordCAS(Protocol):
    def read(self) -> tuple[str, CutoverControlRecord]: ...

    def compare_and_swap(
        self,
        expected_oid: str,
        replacement: CutoverControlRecord,
    ) -> bool: ...


class InMemoryControlRecordCAS:
    """Process-local CAS adapter for offline orchestration and tests."""

    def __init__(self, record: CutoverControlRecord) -> None:
        self._record = record
        self._lock = threading.Lock()

    def read(self) -> tuple[str, CutoverControlRecord]:
        with self._lock:
            return self._record.control_oid, self._record

    def compare_and_swap(
        self,
        expected_oid: str,
        replacement: CutoverControlRecord,
    ) -> bool:
        with self._lock:
            if self._record.control_oid != expected_oid:
                return False
            self._record = replacement
            return True


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
_LEGAL_SIGNED_TRANSITIONS = {
    (SynapseRetirementState.PREPARING, SynapseRetirementState.READY),
    (SynapseRetirementState.READY, SynapseRetirementState.QUIESCED),
    (SynapseRetirementState.QUIESCED, SynapseRetirementState.STAGED),
    (SynapseRetirementState.STAGED, SynapseRetirementState.COMMITTED),
    (SynapseRetirementState.COMMITTED, SynapseRetirementState.VERIFIED),
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
        len(value) == _SHA256_LENGTH and all(character in "0123456789abcdef" for character in value)
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
                "peer_device_ids": list(record.peer_device_ids),
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
        if len(record.peer_device_ids) != 2 or record.peer_device_ids != tuple(
            sorted(set(record.peer_device_ids))
        ):
            raise ControlRecordError("ABORTED Synapse control lacks bound peer devices")
    elif record.peer_device_ids:
        raise ControlRecordError("ABORTED Synapse control has orphan peer devices")
    if record.peer_vote_sha256:
        try:
            _validate_peer_votes(record.peer_vote_sha256)
        except CutoverSafetyError as exc:
            raise ControlRecordError(str(exc)) from exc
    has_staging = bool(record.synapse_manifest_sha256 or record.active_state_receipt_sha256)
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
                record.peer_device_ids,
                record.retirement_epoch,
                record.independence_receipt_sha256,
            )
        ):
            raise ControlRecordError("PREPARING Synapse control contains committed evidence")
        return
    _validate_authority_digests(record)
    if len(record.peer_device_ids) != 2 or record.peer_device_ids != tuple(
        sorted(set(record.peer_device_ids))
    ):
        raise ControlRecordError("Synapse authority requires two bound peer devices")
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


def _manifest_digest(
    manifest: CapabilityManifest,
    *,
    roster: VerificationRoster,
) -> str:
    if (
        manifest.schema != "memo.cutover_capability_manifest.v1"
        or not manifest.frozen
        or manifest.blockers
        or not manifest.signature
    ):
        raise CutoverSafetyError("Synapse capability manifest is not frozen and signed")
    try:
        verify_capability_manifest(manifest, roster=roster)
    except ManifestError as exc:
        raise CutoverSafetyError("Synapse capability manifest signature is invalid") from exc
    if len(manifest.machine_ids) != 2 or manifest.machine_ids != tuple(
        sorted(set(manifest.machine_ids))
    ):
        raise CutoverSafetyError("Synapse capability manifest lacks exact two-peer authority")
    return hashlib.sha256(manifest.signed_bytes()).hexdigest()


def _consumer_plan_digest(
    plan: ConsumerReplacementPlan,
    *,
    inventory: ConsumerInventory,
    manifest: CapabilityManifest,
    roster: VerificationRoster,
    memo_bin: Path,
) -> str:
    try:
        verify_consumer_inventory(inventory, roster=roster)
        verify_capability_manifest(manifest, roster=roster)
    except (InventoryError, ManifestError) as exc:
        raise CutoverSafetyError("Synapse consumer-plan authority is invalid") from exc
    try:
        expected = build_consumer_replacement_plan(
            inventory,
            manifest,
            roster=roster,
            memo_bin=memo_bin,
        )
    except ConsumerMigrationError as exc:
        raise CutoverSafetyError(
            "Synapse consumer plan cannot be deterministically rebuilt"
        ) from exc
    if plan.authority_bytes() != expected.authority_bytes():
        raise CutoverSafetyError(
            "Synapse consumer plan differs from deterministic verified authority"
        )
    if set(plan.covered_surfaces) != _REQUIRED_SURFACES:
        missing = sorted(_REQUIRED_SURFACES - set(plan.covered_surfaces))
        extra = sorted(set(plan.covered_surfaces) - _REQUIRED_SURFACES)
        detail = ",".join((*missing, *(f"unexpected:{item}" for item in extra)))
        raise CutoverSafetyError(f"Synapse preflight surface coverage is incomplete: {detail}")
    if any(
        not values or values != tuple(sorted(set(values))) or any(not value for value in values)
        for values in plan.covered_surfaces.values()
    ):
        raise CutoverSafetyError("Synapse preflight surface coverage is ambiguous")
    return hashlib.sha256(plan.authority_bytes()).hexdigest()


def _require_fresh_control(
    cas: ControlRecordCAS,
    control: VerifiedControlRecord,
    *,
    roster: VerificationRoster,
) -> None:
    fetched_oid, fetched = cas.read()
    if fetched_oid != control.control_oid:
        raise CutoverSafetyError("control record CAS OID is stale")
    try:
        verified = verify_control_record(
            expected_oid=control.control_oid,
            roster=roster,
            record=fetched,
            fetched_oid=fetched_oid,
        )
    except ControlRecordError as exc:
        raise CutoverSafetyError("control record CAS authority is invalid") from exc
    if (
        verified.canonical_payload != control.canonical_payload
        or verified.sequence != control.sequence
        or verified.previous_control_oid != control.previous_control_oid
    ):
        raise CutoverSafetyError("control record was not freshly fetched")


def _next_control_record(
    control: VerifiedControlRecord,
    *,
    next_control_oid: str,
    state: CutoverState,
    synapse_state: SynapseRetirementState,
    capability_manifest_sha256: str | None = None,
    consumer_plan_sha256: str | None = None,
    synapse_authority_sha256: str | None = None,
    peer_device_ids: tuple[str, ...] | None = None,
    peer_vote_sha256: tuple[str, ...] | None = None,
    active_state_receipt_sha256: str | None = None,
    synapse_manifest_sha256: str | None = None,
    retirement_epoch: int | None = None,
    independence_receipt_sha256: str | None = None,
) -> CutoverControlRecord:
    return CutoverControlRecord(
        schema="memo.cutover_control_record.v1",
        control_oid=next_control_oid,
        state=state,
        sequence=control.sequence + 1,
        previous_control_oid=control.control_oid,
        attempt_id=control.attempt_id,
        roster_version=control.roster_version,
        signer_device_id=control.signer_device_id,
        signer_key_id=control.signer_key_id,
        issued_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        signature="",
        synapse_state=synapse_state,
        capability_manifest_sha256=(
            control.capability_manifest_sha256
            if capability_manifest_sha256 is None
            else capability_manifest_sha256
        ),
        synapse_manifest_sha256=(
            control.synapse_manifest_sha256
            if synapse_manifest_sha256 is None
            else synapse_manifest_sha256
        ),
        consumer_plan_sha256=(
            control.consumer_plan_sha256 if consumer_plan_sha256 is None else consumer_plan_sha256
        ),
        synapse_authority_sha256=(
            control.synapse_authority_sha256
            if synapse_authority_sha256 is None
            else synapse_authority_sha256
        ),
        active_state_receipt_sha256=(
            control.active_state_receipt_sha256
            if active_state_receipt_sha256 is None
            else active_state_receipt_sha256
        ),
        peer_vote_sha256=(
            control.peer_vote_sha256 if peer_vote_sha256 is None else peer_vote_sha256
        ),
        peer_device_ids=(control.peer_device_ids if peer_device_ids is None else peer_device_ids),
        retirement_epoch=(
            control.retirement_epoch if retirement_epoch is None else retirement_epoch
        ),
        independence_receipt_sha256=(
            control.independence_receipt_sha256
            if independence_receipt_sha256 is None
            else independence_receipt_sha256
        ),
    )


def _verify_peer_votes(
    control: VerifiedControlRecord,
    votes: tuple[SynapsePeerVote, ...],
    *,
    roster: VerificationRoster,
) -> tuple[str, ...]:
    if len(votes) != 2:
        raise CutoverSafetyError("Synapse retirement requires exactly two peer votes")
    devices: list[str] = []
    key_ids: list[str] = []
    digests: list[str] = []
    for vote in votes:
        if (
            vote.schema != "memo.synapse_peer_vote.v1"
            or vote.attempt_id != control.attempt_id
            or vote.control_oid != control.control_oid
            or vote.authority_sha256 != control.synapse_authority_sha256
            or vote.target_state != "QUIESCED"
        ):
            raise CutoverSafetyError("Synapse peer vote is not bound to cutover authority")
        try:
            key = roster.key(vote.signer_key_id)
            if key.device_id != vote.signer_device_id:
                raise SignatureError("peer vote device does not own signer key")
            OperationalVerifier().verify(
                domain=SYNAPSE_PEER_VOTE_DOMAIN,
                payload=vote.signed_bytes(),
                envelope=vote.signature_envelope(),
                roster=roster,
            )
        except (KeyError, SignatureError) as exc:
            raise CutoverSafetyError("Synapse peer vote signature is invalid") from exc
        devices.append(vote.signer_device_id)
        key_ids.append(vote.signer_key_id)
        digests.append(hashlib.sha256(vote.signed_bytes()).hexdigest())
    if (
        tuple(sorted(devices)) != control.peer_device_ids
        or len(set(devices)) != 2
        or len(set(key_ids)) != 2
    ):
        raise CutoverSafetyError("Synapse peer votes do not cover both authority devices")
    return tuple(sorted(digests))


def _commit_signed_transition(
    cas: ControlRecordCAS,
    predecessor: VerifiedControlRecord,
    replacement: CutoverControlRecord,
    *,
    signer: OperationalSigner,
    roster: VerificationRoster,
) -> VerifiedControlRecord:
    signed = sign_control_record(replacement, signer=signer, predecessor=predecessor)
    try:
        verified_signed = verify_control_record(
            expected_oid=signed.control_oid,
            roster=roster,
            record=signed,
            fetched_oid=signed.control_oid,
        )
    except ControlRecordError as exc:
        raise CutoverSafetyError(
            "replacement control record failed roster verification before CAS"
        ) from exc
    if not cas.compare_and_swap(predecessor.control_oid, signed):
        raise CutoverSafetyError("control record CAS compare-and-swap failed")
    fetched_oid, fetched = cas.read()
    try:
        committed = verify_control_record(
            expected_oid=signed.control_oid,
            roster=roster,
            record=fetched,
            fetched_oid=fetched_oid,
        )
    except ControlRecordError as exc:
        raise CutoverSafetyError("committed control record failed verification") from exc
    if committed.canonical_payload != verified_signed.canonical_payload:
        raise CutoverSafetyError("committed control record differs from verified replacement")
    return committed


def prepare_synapse_retirement(
    cas: ControlRecordCAS,
    control: VerifiedControlRecord,
    manifest: CapabilityManifest,
    inventory: ConsumerInventory,
    consumer_plan: ConsumerReplacementPlan,
    *,
    roster: VerificationRoster,
    signer: OperationalSigner,
    next_control_oid: str,
    memo_bin: Path,
) -> VerifiedControlRecord:
    """Bind all signed preflight authority before quiescing either product."""

    _require_fresh_control(cas, control, roster=roster)
    if control.synapse_state in {
        SynapseRetirementState.COMMITTED,
        SynapseRetirementState.VERIFIED,
    }:
        raise CutoverSafetyError("synapse.cutover.retired")
    if control.synapse_state is not SynapseRetirementState.PREPARING:
        raise CutoverSafetyError("stale Synapse cutover attempt")
    manifest_sha256 = _manifest_digest(manifest, roster=roster)
    consumer_plan_sha256 = _consumer_plan_digest(
        consumer_plan,
        inventory=inventory,
        manifest=manifest,
        roster=roster,
        memo_bin=memo_bin,
    )
    peer_device_ids = tuple(sorted(manifest.machine_ids))
    authority_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "capability_manifest_sha256": manifest_sha256,
                "consumer_plan_sha256": consumer_plan_sha256,
                "peer_device_ids": list(peer_device_ids),
            }
        )
    ).hexdigest()
    replacement = _next_control_record(
        control,
        next_control_oid=next_control_oid,
        state=CutoverState.READY,
        synapse_state=SynapseRetirementState.READY,
        capability_manifest_sha256=manifest_sha256,
        consumer_plan_sha256=consumer_plan_sha256,
        synapse_authority_sha256=authority_sha256,
        peer_device_ids=peer_device_ids,
    )
    return _commit_signed_transition(
        cas,
        control,
        replacement,
        signer=signer,
        roster=roster,
    )


def advance_synapse_retirement(
    cas: ControlRecordCAS,
    control: VerifiedControlRecord,
    target: SynapseRetirementState,
    *,
    roster: VerificationRoster,
    signer: OperationalSigner,
    next_control_oid: str,
    peer_votes: tuple[SynapsePeerVote, ...] = (),
    active_state_receipt_sha256: str = "",
    synapse_manifest: SynapseRetirementManifest | None = None,
    independence_receipt: IndependenceReceipt | None = None,
    independence_inventory: ConsumerInventory | None = None,
    independence_manifest: SynapseRetirementManifest | None = None,
    post_stop_scan: IndependenceScanReceipt | None = None,
    post_reboot_scan: IndependenceScanReceipt | None = None,
) -> VerifiedControlRecord:
    """Advance one signed CAS edge from an exactly freshly verified record."""

    _require_fresh_control(cas, control, roster=roster)
    if target is SynapseRetirementState.ABORTED:
        if control.synapse_state in {
            SynapseRetirementState.COMMITTED,
            SynapseRetirementState.VERIFIED,
            SynapseRetirementState.ABORTED,
        }:
            raise CutoverSafetyError("Synapse retirement cannot abort after commit")
        replacement = _next_control_record(
            control,
            next_control_oid=next_control_oid,
            state=CutoverState.ABORTED,
            synapse_state=target,
        )
        return _commit_signed_transition(
            cas,
            control,
            replacement,
            signer=signer,
            roster=roster,
        )
    if _SYNAPSE_TRANSITIONS.get(control.synapse_state) is not target:
        raise CutoverSafetyError("stale or skipped Synapse retirement transition")
    if target is SynapseRetirementState.QUIESCED:
        vote_digests = _verify_peer_votes(control, peer_votes, roster=roster)
        replacement = _next_control_record(
            control,
            next_control_oid=next_control_oid,
            state=CutoverState.QUIESCED,
            synapse_state=target,
            peer_vote_sha256=vote_digests,
        )
    elif target is SynapseRetirementState.STAGED:
        if not _valid_sha256(active_state_receipt_sha256):
            raise CutoverSafetyError("active-state migration receipt digest is missing")
        if synapse_manifest is None:
            raise CutoverSafetyError("signed Synapse retirement manifest is missing")
        try:
            verify_synapse_retirement_manifest(synapse_manifest, roster=roster)
        except InventoryError as exc:
            raise CutoverSafetyError(
                "signed Synapse retirement manifest signature is invalid"
            ) from exc
        synapse_manifest_sha256 = hashlib.sha256(synapse_manifest.signed_bytes()).hexdigest()
        replacement = _next_control_record(
            control,
            next_control_oid=next_control_oid,
            state=CutoverState.STAGED,
            synapse_state=target,
            active_state_receipt_sha256=active_state_receipt_sha256,
            synapse_manifest_sha256=synapse_manifest_sha256,
        )
    else:
        assert target is SynapseRetirementState.VERIFIED
        evidence = (
            independence_receipt,
            independence_inventory,
            independence_manifest,
            post_stop_scan,
            post_reboot_scan,
        )
        if any(item is None for item in evidence):
            raise CutoverSafetyError("verified Synapse independence evidence is incomplete")
        assert independence_receipt is not None
        assert independence_inventory is not None
        assert independence_manifest is not None
        assert post_stop_scan is not None
        assert post_reboot_scan is not None
        from tools.memflow_absorption.safety import verify_independence_receipt

        verify_independence_receipt(
            independence_receipt,
            control,
            independence_inventory,
            independence_manifest,
            post_stop_scan,
            post_reboot_scan,
            roster=roster,
        )
        independence_receipt_sha256 = hashlib.sha256(
            independence_receipt.signed_bytes()
        ).hexdigest()
        replacement = _next_control_record(
            control,
            next_control_oid=next_control_oid,
            state=CutoverState.VERIFIED,
            synapse_state=target,
            independence_receipt_sha256=independence_receipt_sha256,
        )
    return _commit_signed_transition(
        cas,
        control,
        replacement,
        signer=signer,
        roster=roster,
    )


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
        peer_device_ids=record.peer_device_ids,
        retirement_epoch=record.retirement_epoch,
        independence_receipt_sha256=record.independence_receipt_sha256,
    )


def commit_synapse_activation(
    cas: ControlRecordCAS,
    control: VerifiedControlRecord,
    epoch: int,
    *,
    roster: VerificationRoster,
    signer: OperationalSigner,
    next_control_oid: str,
) -> VerifiedControlRecord:
    """Atomically commit the sole STAGED -> COMMITTED activation epoch."""

    _require_fresh_control(cas, control, roster=roster)
    if control.retirement_epoch:
        raise CutoverSafetyError("second Synapse activation epoch is forbidden")
    if epoch < 1:
        raise CutoverSafetyError("Synapse activation epoch must be positive")
    if control.synapse_state is not SynapseRetirementState.STAGED:
        raise CutoverSafetyError("stale or skipped Synapse retirement transition")
    replacement = _next_control_record(
        control,
        next_control_oid=next_control_oid,
        state=CutoverState.EPOCH_COMMITTED,
        synapse_state=SynapseRetirementState.COMMITTED,
        retirement_epoch=epoch,
    )
    return _commit_signed_transition(
        cas,
        control,
        replacement,
        signer=signer,
        roster=roster,
    )


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
    predecessor: VerifiedControlRecord | None = None,
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
    if predecessor is None:
        if record.sequence != 1 or record.previous_control_oid != "":
            raise ControlRecordError("initial control record predecessor is invalid")
        if record.synapse_state is not SynapseRetirementState.PREPARING:
            raise ControlRecordError("initial control record must be PREPARING")
    else:
        if (
            record.sequence != predecessor.sequence + 1
            or record.previous_control_oid != predecessor.control_oid
            or record.control_oid == predecessor.control_oid
        ):
            raise ControlRecordError("control record predecessor does not match exact sequence")
        transition = (predecessor.synapse_state, record.synapse_state)
        if not (
            transition in _LEGAL_SIGNED_TRANSITIONS
            or (
                record.synapse_state is SynapseRetirementState.ABORTED
                and predecessor.synapse_state
                in {
                    SynapseRetirementState.PREPARING,
                    SynapseRetirementState.READY,
                    SynapseRetirementState.QUIESCED,
                    SynapseRetirementState.STAGED,
                }
            )
        ):
            raise ControlRecordError("control record transition is illegal")
        if (
            record.attempt_id != predecessor.attempt_id
            or record.roster_version != predecessor.roster_version
        ):
            raise ControlRecordError("control record predecessor authority changed")
        if predecessor.synapse_state is not SynapseRetirementState.PREPARING and (
            record.capability_manifest_sha256 != predecessor.capability_manifest_sha256
            or record.consumer_plan_sha256 != predecessor.consumer_plan_sha256
            or record.synapse_authority_sha256 != predecessor.synapse_authority_sha256
            or record.peer_device_ids != predecessor.peer_device_ids
        ):
            raise ControlRecordError("control record authority changed after preflight")
        if (
            predecessor.synapse_state
            in {
                SynapseRetirementState.QUIESCED,
                SynapseRetirementState.STAGED,
                SynapseRetirementState.COMMITTED,
                SynapseRetirementState.VERIFIED,
            }
            and record.peer_vote_sha256 != predecessor.peer_vote_sha256
        ):
            raise ControlRecordError("control record peer votes changed after quiescence")
        if predecessor.synapse_state in {
            SynapseRetirementState.STAGED,
            SynapseRetirementState.COMMITTED,
            SynapseRetirementState.VERIFIED,
        } and (
            record.synapse_manifest_sha256 != predecessor.synapse_manifest_sha256
            or record.active_state_receipt_sha256 != predecessor.active_state_receipt_sha256
        ):
            raise ControlRecordError("control record staging evidence changed after staging")
        if (
            predecessor.synapse_state
            in {
                SynapseRetirementState.COMMITTED,
                SynapseRetirementState.VERIFIED,
            }
            and record.retirement_epoch != predecessor.retirement_epoch
        ):
            raise ControlRecordError("control record retirement epoch changed after commit")
        if (
            predecessor.independence_receipt_sha256
            and record.independence_receipt_sha256 != predecessor.independence_receipt_sha256
        ):
            raise ControlRecordError(
                "control record independence receipt changed after verification"
            )
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
        key = roster.key(record.signer_key_id)
        if record.roster_version != roster.version or key.device_id != record.signer_device_id:
            raise SignatureError("control record signer does not match roster authority")
        OperationalVerifier().verify(
            domain=CONTROL_RECORD_DOMAIN,
            payload=record.canonical_payload,
            envelope=record.signature_envelope(),
            roster=roster,
        )
    except (KeyError, SignatureError) as exc:
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
        peer_device_ids=record.peer_device_ids,
        retirement_epoch=record.retirement_epoch,
        independence_receipt_sha256=record.independence_receipt_sha256,
    )


def fetch_verified_control(
    cas: ControlRecordCAS,
    *,
    expected_oid: str,
    roster: VerificationRoster,
) -> VerifiedControlRecord:
    fetched_oid, record = cas.read()
    return verify_control_record(
        expected_oid=expected_oid,
        roster=roster,
        record=record,
        fetched_oid=fetched_oid,
    )


def fetch_current_verified_control(
    cas: ControlRecordCAS,
    *,
    roster: VerificationRoster,
) -> VerifiedControlRecord:
    """Fetch and verify the authoritative CAS head without trusting cached state."""

    fetched_oid, record = cas.read()
    return verify_control_record(
        expected_oid=fetched_oid,
        roster=roster,
        record=record,
        fetched_oid=fetched_oid,
    )


__all__ = [
    "CONTROL_RECORD_DOMAIN",
    "SYNAPSE_PEER_VOTE_DOMAIN",
    "ControlRecordCAS",
    "ControlRecordError",
    "CutoverSafetyError",
    "InMemoryControlRecordCAS",
    "advance_synapse_retirement",
    "commit_synapse_activation",
    "control_record_from_dict",
    "fetch_current_verified_control",
    "fetch_verified_control",
    "prepare_synapse_retirement",
    "sign_control_record",
    "validate_synapse_request",
    "verify_control_record",
]
