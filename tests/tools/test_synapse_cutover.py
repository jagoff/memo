from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from memo.operational_event import canonical_json_bytes
from memo.operational_key_store import (
    AuthorityPinStore,
    DeviceKeyStore,
    InMemoryAuthorityPinProvider,
)
from memo.operational_roster import VerificationRoster
from memo.operational_signing import OperationalSigner, SignatureEnvelope
from tools.memflow_absorption.__main__ import main
from tools.memflow_absorption.consumer_migration import (
    build_consumer_replacement_plan,
)
from tools.memflow_absorption.control_record import (
    CONTROL_RECORD_DOMAIN,
    SYNAPSE_PEER_VOTE_DOMAIN,
    ControlRecordError,
    CutoverSafetyError,
    InMemoryControlRecordCAS,
    advance_synapse_retirement,
    commit_synapse_activation,
    fetch_verified_control,
    prepare_synapse_retirement,
    sign_control_record,
    validate_synapse_request,
)
from tools.memflow_absorption.inventory import (
    CONSUMER_INVENTORY_DOMAIN,
    SYNAPSE_RETIREMENT_DOMAIN,
)
from tools.memflow_absorption.manifest import CAPABILITY_MANIFEST_DOMAIN
from tools.memflow_absorption.runtime_gate import (
    before_fallback,
    before_listener_start,
    before_worker_start,
    before_write,
)
from tools.memflow_absorption.safety import (
    SYNAPSE_INDEPENDENCE_RECEIPT_DOMAIN,
    SYNAPSE_INDEPENDENCE_SCAN_DOMAIN,
    independence_receipt_from_dict,
    verify_independence_receipt,
    verify_synapse_retired,
)
from tools.memflow_absorption.schemas import (
    CapabilityManifest,
    ConsumerInventory,
    ConsumerInventoryRow,
    ConsumerReplacement,
    ConsumerReplacementPlan,
    CutoverControlRecord,
    CutoverState,
    IndependenceObservation,
    IndependenceReceipt,
    IndependenceScanReceipt,
    SynapseOperation,
    SynapsePeerVote,
    SynapseRetirementManifest,
    SynapseRetirementState,
    VerifiedControlRecord,
)

SURFACES = (
    "launchagent",
    "mcp_gateway_route",
    "port",
    "process",
    "shell_config_path",
    "state_root",
)


@pytest.fixture
def authority(
    tmp_path: Path,
) -> tuple[DeviceKeyStore, VerificationRoster, OperationalSigner]:
    keys = DeviceKeyStore.in_memory()
    mac_a = keys.generate(device_id="mac-a")
    pins = AuthorityPinStore._for_test(
        tmp_path / "authority",
        provider=InMemoryAuthorityPinProvider(),
    )
    roster = VerificationRoster.bootstrap(
        device_id="mac-a",
        key=mac_a,
        root=tmp_path / "authority",
        pin_store=pins,
    )
    mac_b = keys.generate(device_id="mac-b", roles=("origin",), enrollment_sequence=2)
    roster = roster.with_keys(
        version=2,
        peers=("mac-a", "mac-b"),
        keys=(mac_a, mac_b),
        signer=OperationalSigner(keys, roster_version=1),
        root=tmp_path / "authority",
        pin_store=pins,
    )
    return keys, roster, OperationalSigner(keys, roster_version=roster.version)


def _signed_control(authority) -> CutoverControlRecord:
    _keys, roster, signer = authority
    return sign_control_record(
        CutoverControlRecord(
            schema="memo.cutover_control_record.v1",
            control_oid="b" * 40,
            state=CutoverState.PREPARING,
            sequence=1,
            previous_control_oid="",
            attempt_id="attempt-123",
            roster_version=roster.version,
            signer_device_id=roster.local_device_id,
            signer_key_id=roster.local_key_id,
            issued_at="2026-07-30T00:00:00Z",
            signature="",
        ),
        signer=signer,
    )


def _signed_manifest(authority) -> CapabilityManifest:
    _keys, roster, signer = authority
    unsigned = CapabilityManifest(
        schema="memo.cutover_capability_manifest.v1",
        frozen_at="2026-07-30T00:00:00Z",
        window_started_at="2026-07-29T00:00:00Z",
        window_ended_at="2026-07-30T00:00:00Z",
        machine_ids=("mac-a", "mac-b"),
        source_receipt_sha256={"synapse": "a" * 64},
        capabilities=(),
        operation_mappings=(),
        slo_baselines=(),
        operation_map_sha256=hashlib.sha256(
            canonical_json_bytes(
                {
                    "mappings": [],
                    "registry_authority_sha256": "",
                    "fixture_authority_sha256": "",
                }
            )
        ).hexdigest(),
        slo_baseline_sha256=hashlib.sha256(canonical_json_bytes([])).hexdigest(),
        blockers=(),
        frozen=True,
        signer_device_id=roster.local_device_id,
        signer_key_id=roster.local_key_id,
        roster_version=roster.version,
        signature="",
    )
    envelope = signer.sign(
        domain=CAPABILITY_MANIFEST_DOMAIN,
        payload=unsigned.signed_bytes(),
        key_id=roster.local_key_id,
    )
    return replace(unsigned, signature=envelope.signature)


def _observations() -> dict[str, tuple[str, ...]]:
    return {surface: (f"scan:{surface}",) for surface in SURFACES}


def _signed_inventory(
    authority,
    *,
    scan_digests: tuple[str, ...] = (),
) -> ConsumerInventory:
    _keys, roster, signer = authority
    source_digest = (
        hashlib.sha256(canonical_json_bytes(list(scan_digests))).hexdigest()
        if scan_digests
        else "e" * 64
    )
    unsigned = ConsumerInventory(
        schema="memo.cutover_consumer_inventory.v1",
        rows=(),
        blockers=(),
        source_scan_sha256=source_digest,
        signer_device_id=roster.local_device_id,
        signer_key_id=roster.local_key_id,
        roster_version=roster.version,
        signature="",
        covered_surfaces=SURFACES if scan_digests else (),
        surface_observations=_observations(),
        scan_receipt_sha256=scan_digests,
    )
    envelope = signer.sign(
        domain=CONSUMER_INVENTORY_DOMAIN,
        payload=unsigned.signed_bytes(),
        key_id=roster.local_key_id,
    )
    return replace(unsigned, signature=envelope.signature)


def _real_plan(
    authority, tmp_path: Path
) -> tuple[
    CapabilityManifest,
    ConsumerInventory,
    ConsumerReplacementPlan,
]:
    memo_bin = tmp_path / "runtime" / "memo"
    memo_bin.parent.mkdir(exist_ok=True)
    memo_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    memo_bin.chmod(0o755)
    manifest = _signed_manifest(authority)
    inventory = _signed_inventory(authority)
    plan = build_consumer_replacement_plan(
        inventory,
        manifest,
        roster=authority[1],
        memo_bin=memo_bin,
    )
    return manifest, inventory, plan


def _initial(authority) -> tuple[InMemoryControlRecordCAS, VerifiedControlRecord]:
    record = _signed_control(authority)
    cas = InMemoryControlRecordCAS(record)
    return cas, fetch_verified_control(
        cas,
        expected_oid=record.control_oid,
        roster=authority[1],
    )


def _vote(
    authority,
    control: VerifiedControlRecord,
    device_id: str,
) -> SynapsePeerVote:
    keys, roster, signer = authority
    key_id = next(key.key_id for key in roster.keys if key.device_id == device_id)
    unsigned = SynapsePeerVote(
        schema="memo.synapse_peer_vote.v1",
        attempt_id=control.attempt_id,
        control_oid=control.control_oid,
        authority_sha256=control.synapse_authority_sha256,
        target_state="QUIESCED",
        signer_device_id=device_id,
        signer_key_id=key_id,
        roster_version=roster.version,
        signature="",
    )
    envelope = signer.sign(
        domain=SYNAPSE_PEER_VOTE_DOMAIN,
        payload=unsigned.signed_bytes(),
        key_id=key_id,
    )
    assert keys.algorithm_for_key_id(key_id) == "ed25519"
    return replace(unsigned, signature=envelope.signature)


def _retirement_manifest(authority) -> SynapseRetirementManifest:
    _keys, roster, signer = authority
    unsigned = SynapseRetirementManifest(
        schema="memo.synapse_retirement.v2",
        source_commit="c" * 40,
        files=("synapse/server.py",),
        symbols=("MemflowGateway",),
        tests=("tests/test_server.py",),
        goldens=("tests/goldens/ask.json",),
        active_reference_sha256="d" * 64,
        signer_key_id=roster.local_key_id,
        signature="",
        operations=(
            SynapseOperation(
                source_operation="synapse.chat.ask",
                source_files=("synapse/server.py",),
                source_symbols=("chat_ask",),
                consumers=("dashboard",),
                daemon_routes=("/chat",),
                exclusion_reason=None,
                fixture_paths=("tests/goldens/ask.json",),
            ),
        ),
    )
    envelope = signer.sign(
        domain=SYNAPSE_RETIREMENT_DOMAIN,
        payload=unsigned.signed_bytes(),
        key_id=roster.local_key_id,
    )
    return replace(unsigned, signature=envelope.signature)


def _committed(authority, tmp_path: Path):
    manifest, inventory, plan = _real_plan(authority, tmp_path)
    cas, initial = _initial(authority)
    ready = prepare_synapse_retirement(
        cas,
        initial,
        manifest,
        inventory,
        plan,
        roster=authority[1],
        signer=authority[2],
        next_control_oid="c" * 40,
        memo_bin=tmp_path / "runtime" / "memo",
    )
    votes = (_vote(authority, ready, "mac-a"), _vote(authority, ready, "mac-b"))
    quiesced = advance_synapse_retirement(
        cas,
        ready,
        SynapseRetirementState.QUIESCED,
        roster=authority[1],
        signer=authority[2],
        next_control_oid="d" * 40,
        peer_votes=votes,
    )
    retirement = _retirement_manifest(authority)
    staged = advance_synapse_retirement(
        cas,
        quiesced,
        SynapseRetirementState.STAGED,
        roster=authority[1],
        signer=authority[2],
        next_control_oid="e" * 40,
        active_state_receipt_sha256="3" * 64,
        synapse_manifest=retirement,
    )
    committed = commit_synapse_activation(
        cas,
        staged,
        epoch=8,
        roster=authority[1],
        signer=authority[2],
        next_control_oid="f" * 40,
    )
    return cas, committed, retirement


def _scan(
    authority,
    phase: str,
    *,
    active: bool = False,
    boot_id: str | None = None,
    captured_at: str | None = None,
) -> IndependenceScanReceipt:
    _keys, roster, signer = authority
    observations = tuple(
        IndependenceObservation(
            surface=surface,  # type: ignore[arg-type]
            identifier=f"scan:{surface}",
            active=active and surface == "process",
            references=("synapse",) if active and surface == "process" else (),
        )
        for surface in SURFACES
    )
    unsigned = IndependenceScanReceipt(
        schema="memo.synapse_independence_scan.v1",
        phase=phase,  # type: ignore[arg-type]
        boot_id=boot_id or ("boot-before" if phase == "post_stop" else "boot-after"),
        captured_at=captured_at
        or ("2026-07-30T01:00:00Z" if phase == "post_stop" else "2026-07-30T02:00:00Z"),
        source_scan_sha256=hashlib.sha256(
            canonical_json_bytes([row.to_dict() for row in observations])
        ).hexdigest(),
        observations=observations,
        signer_device_id=roster.local_device_id,
        signer_key_id=roster.local_key_id,
        roster_version=roster.version,
        signature="",
    )
    envelope = signer.sign(
        domain=SYNAPSE_INDEPENDENCE_SCAN_DOMAIN,
        payload=unsigned.signed_bytes(),
        key_id=roster.local_key_id,
    )
    return replace(unsigned, signature=envelope.signature)


def _signed_receipt(
    authority,
    control: VerifiedControlRecord,
    inventory: ConsumerInventory,
    retirement: SynapseRetirementManifest,
    stop: IndependenceScanReceipt,
    reboot: IndependenceScanReceipt,
) -> IndependenceReceipt:
    _keys, roster, signer = authority
    unsigned = IndependenceReceipt(
        schema="memo.synapse_independence_receipt.v1",
        attempt_id=control.attempt_id,
        control_oid=control.control_oid,
        retirement_epoch=control.retirement_epoch,
        synapse_manifest_sha256=hashlib.sha256(retirement.signed_bytes()).hexdigest(),
        consumer_inventory_sha256=hashlib.sha256(inventory.signed_bytes()).hexdigest(),
        scan_receipt_sha256=(
            hashlib.sha256(stop.signed_bytes()).hexdigest(),
            hashlib.sha256(reboot.signed_bytes()).hexdigest(),
        ),
        verified_at="2026-07-30T03:00:00Z",
        signer_device_id=roster.local_device_id,
        signer_key_id=roster.local_key_id,
        roster_version=roster.version,
        signature="",
    )
    envelope = signer.sign(
        domain=SYNAPSE_INDEPENDENCE_RECEIPT_DOMAIN,
        payload=unsigned.signed_bytes(),
        key_id=roster.local_key_id,
    )
    return replace(unsigned, signature=envelope.signature)


def test_real_consumer_builder_populates_verified_authority(authority, tmp_path) -> None:
    manifest, inventory, plan = _real_plan(authority, tmp_path)

    assert dict(plan.covered_surfaces) == _observations()
    assert plan.inventory_sha256 == hashlib.sha256(inventory.signed_bytes()).hexdigest()
    assert plan.capability_manifest_sha256 == hashlib.sha256(manifest.signed_bytes()).hexdigest()


@pytest.mark.parametrize("forgery", ["plan", "inventory", "manifest"])
def test_preflight_rejects_every_authority_forgery(
    authority,
    tmp_path: Path,
    forgery: str,
) -> None:
    manifest, inventory, plan = _real_plan(authority, tmp_path)
    if forgery == "plan":
        plan = replace(plan, inventory_sha256="0" * 64)
    elif forgery == "inventory":
        inventory = replace(inventory, surface_observations={"process": ("fake",)})
    else:
        manifest = replace(manifest, machine_ids=("mac-a", "mac-c"))
    cas, initial = _initial(authority)

    with pytest.raises(CutoverSafetyError):
        prepare_synapse_retirement(
            cas,
            initial,
            manifest,
            inventory,
            plan,
            roster=authority[1],
            signer=authority[2],
            next_control_oid="c" * 40,
            memo_bin=tmp_path / "runtime" / "memo",
        )


def test_preflight_rebuilds_plan_instead_of_trusting_self_consistent_rows(
    authority,
    tmp_path: Path,
) -> None:
    manifest, inventory, plan = _real_plan(authority, tmp_path)
    forged_row = ConsumerReplacement(
        old_label="com.synapse.unauthorized",
        new_label="com.memo.unauthorized",
        command=("/usr/local/bin/memo", "save"),
        owner="memo_native",
        restart_required=True,
        config_sha256="1" * 64,
        rollback_action="operator",
    )
    forged_rows = (forged_row,)
    forged_plan = replace(
        plan,
        rows=forged_rows,
        digest=hashlib.sha256(
            canonical_json_bytes([row.to_dict() for row in forged_rows])
        ).hexdigest(),
    )
    cas, initial = _initial(authority)

    with pytest.raises(CutoverSafetyError, match="deterministic verified authority"):
        prepare_synapse_retirement(
            cas,
            initial,
            manifest,
            inventory,
            forged_plan,
            roster=authority[1],
            signer=authority[2],
            next_control_oid="c" * 40,
            memo_bin=tmp_path / "runtime" / "memo",
        )


def test_invalid_replacement_signature_never_poison_cas(
    authority,
    tmp_path: Path,
) -> None:
    manifest, inventory, plan = _real_plan(authority, tmp_path)
    cas, initial = _initial(authority)
    before_oid, before_record = cas.read()

    class InvalidSigner:
        def sign(self, *, domain: str, payload: bytes, key_id: str) -> SignatureEnvelope:
            del domain, payload
            return SignatureEnvelope(
                algorithm="ed25519",
                key_id=key_id,
                roster_version=authority[1].version,
                signature="invalid",
            )

    with pytest.raises(CutoverSafetyError, match="before CAS"):
        prepare_synapse_retirement(
            cas,
            initial,
            manifest,
            inventory,
            plan,
            roster=authority[1],
            signer=InvalidSigner(),  # type: ignore[arg-type]
            next_control_oid="c" * 40,
            memo_bin=tmp_path / "runtime" / "memo",
        )
    assert cas.read() == (before_oid, before_record)


def test_control_verification_binds_claimed_signer_to_roster(authority) -> None:
    record = _signed_control(authority)
    unsigned = replace(record, signer_device_id="mac-b", signature="")
    envelope = authority[2].sign(
        domain="memo.cutover.control_record.v1",
        payload=unsigned.canonical_payload,
        key_id=authority[1].local_key_id,
    )
    forged = replace(unsigned, signature=envelope.signature)
    cas = InMemoryControlRecordCAS(forged)

    with pytest.raises(ControlRecordError, match="signature is invalid"):
        fetch_verified_control(
            cas,
            expected_oid=forged.control_oid,
            roster=authority[1],
        )


def test_advance_requires_fresh_cas_and_two_bound_signed_peer_votes(
    authority,
    tmp_path: Path,
) -> None:
    manifest, inventory, plan = _real_plan(authority, tmp_path)
    cas, initial = _initial(authority)
    ready = prepare_synapse_retirement(
        cas,
        initial,
        manifest,
        inventory,
        plan,
        roster=authority[1],
        signer=authority[2],
        next_control_oid="c" * 40,
        memo_bin=tmp_path / "runtime" / "memo",
    )
    mac_a = _vote(authority, ready, "mac-a")
    with pytest.raises(CutoverSafetyError, match="both authority devices"):
        advance_synapse_retirement(
            cas,
            ready,
            SynapseRetirementState.QUIESCED,
            roster=authority[1],
            signer=authority[2],
            next_control_oid="d" * 40,
            peer_votes=(mac_a, mac_a),
        )
    forged = replace(_vote(authority, ready, "mac-b"), control_oid="9" * 40)
    with pytest.raises(CutoverSafetyError, match="not bound"):
        advance_synapse_retirement(
            cas,
            ready,
            SynapseRetirementState.QUIESCED,
            roster=authority[1],
            signer=authority[2],
            next_control_oid="d" * 40,
            peer_votes=(mac_a, forged),
        )
    with pytest.raises(CutoverSafetyError, match="CAS OID is stale"):
        prepare_synapse_retirement(
            cas,
            initial,
            manifest,
            inventory,
            plan,
            roster=authority[1],
            signer=authority[2],
            next_control_oid="d" * 40,
            memo_bin=tmp_path / "runtime" / "memo",
        )


def test_peer_vote_rejects_device_impersonation_and_reused_roster_key(
    authority,
    tmp_path: Path,
) -> None:
    manifest, inventory, plan = _real_plan(authority, tmp_path)
    cas, initial = _initial(authority)
    ready = prepare_synapse_retirement(
        cas,
        initial,
        manifest,
        inventory,
        plan,
        roster=authority[1],
        signer=authority[2],
        next_control_oid="c" * 40,
        memo_bin=tmp_path / "runtime" / "memo",
    )
    mac_a = _vote(authority, ready, "mac-a")
    impersonated = replace(mac_a, signer_device_id="mac-b", signature="")
    envelope = authority[2].sign(
        domain=SYNAPSE_PEER_VOTE_DOMAIN,
        payload=impersonated.signed_bytes(),
        key_id=mac_a.signer_key_id,
    )
    impersonated = replace(impersonated, signature=envelope.signature)
    before = cas.read()

    with pytest.raises(CutoverSafetyError, match="signature is invalid"):
        advance_synapse_retirement(
            cas,
            ready,
            SynapseRetirementState.QUIESCED,
            roster=authority[1],
            signer=authority[2],
            next_control_oid="d" * 40,
            peer_votes=(mac_a, impersonated),
        )
    assert cas.read() == before


def test_signing_requires_exact_predecessor_sequence_and_legal_transition(
    authority,
) -> None:
    cas, initial = _initial(authority)
    _ = cas
    malformed = CutoverControlRecord(
        schema="memo.cutover_control_record.v1",
        control_oid="c" * 40,
        state=CutoverState.QUIESCED,
        sequence=initial.sequence + 2,
        previous_control_oid="9" * 40,
        attempt_id=initial.attempt_id,
        roster_version=initial.roster_version,
        signer_device_id=initial.signer_device_id,
        signer_key_id=initial.signer_key_id,
        issued_at="2026-07-30T01:00:00Z",
        signature="",
        synapse_state=SynapseRetirementState.QUIESCED,
    )
    with pytest.raises(ControlRecordError, match="exact sequence"):
        sign_control_record(malformed, signer=authority[2], predecessor=initial)


def test_commit_is_atomic_one_time_and_record_is_roster_verified(
    authority,
    tmp_path: Path,
) -> None:
    cas, committed, _retirement = _committed(authority, tmp_path)

    assert committed.synapse_state is SynapseRetirementState.COMMITTED
    refetched = fetch_verified_control(
        cas,
        expected_oid=committed.control_oid,
        roster=authority[1],
    )
    assert refetched.canonical_payload == committed.canonical_payload
    assert refetched.control_oid == committed.control_oid
    with pytest.raises(CutoverSafetyError):
        commit_synapse_activation(
            cas,
            committed,
            epoch=9,
            roster=authority[1],
            signer=authority[2],
            next_control_oid="1" * 40,
        )


def test_verified_transition_rejects_unverified_receipt_without_mutating_cas(
    authority,
    tmp_path: Path,
) -> None:
    cas, committed, retirement = _committed(authority, tmp_path)
    stop = _scan(authority, "post_stop")
    reboot = _scan(authority, "post_reboot")
    digests = (
        hashlib.sha256(stop.signed_bytes()).hexdigest(),
        hashlib.sha256(reboot.signed_bytes()).hexdigest(),
    )
    inventory = _signed_inventory(authority, scan_digests=digests)
    receipt = _signed_receipt(authority, committed, inventory, retirement, stop, reboot)
    before = cas.read()

    with pytest.raises(CutoverSafetyError, match="receipt signature"):
        advance_synapse_retirement(
            cas,
            committed,
            SynapseRetirementState.VERIFIED,
            roster=authority[1],
            signer=authority[2],
            next_control_oid="1" * 40,
            independence_receipt=replace(receipt, signature="invalid"),
            independence_inventory=inventory,
            independence_manifest=retirement,
            post_stop_scan=stop,
            post_reboot_scan=reboot,
        )
    assert cas.read() == before


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("peer_vote_sha256", ("4" * 64, "5" * 64), "peer votes changed"),
        ("synapse_manifest_sha256", "4" * 64, "staging evidence changed"),
        ("active_state_receipt_sha256", "4" * 64, "staging evidence changed"),
        ("retirement_epoch", 9, "retirement epoch changed"),
    ],
)
def test_signing_freezes_all_committed_authority(
    authority,
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    cas, committed, _retirement = _committed(authority, tmp_path)
    committed_record = cas.read()[1]
    successor = replace(
        committed_record,
        control_oid="1" * 40,
        state=CutoverState.VERIFIED,
        sequence=committed.sequence + 1,
        previous_control_oid=committed.control_oid,
        issued_at="2026-07-30T03:00:00Z",
        signature="",
        synapse_state=SynapseRetirementState.VERIFIED,
        independence_receipt_sha256="6" * 64,
        **{field: value},
    )

    with pytest.raises(ControlRecordError, match=message):
        sign_control_record(successor, signer=authority[2], predecessor=committed)


@pytest.mark.parametrize("failure", ["same_boot", "missing_surface", "active", "signature"])
def test_final_scan_receipts_fail_closed(
    authority,
    tmp_path: Path,
    failure: str,
) -> None:
    _cas, committed, retirement = _committed(authority, tmp_path)
    stop = _scan(authority, "post_stop")
    reboot = _scan(authority, "post_reboot", active=failure == "active")
    if failure == "same_boot":
        reboot = replace(reboot, boot_id=stop.boot_id)
    elif failure == "missing_surface":
        reboot = replace(reboot, observations=reboot.observations[:-1])
    elif failure == "signature":
        reboot = replace(reboot, signature=stop.signature)
    digests = (
        hashlib.sha256(stop.signed_bytes()).hexdigest(),
        hashlib.sha256(reboot.signed_bytes()).hexdigest(),
    )
    inventory = _signed_inventory(authority, scan_digests=digests)

    with pytest.raises(CutoverSafetyError):
        verify_synapse_retired(
            committed,
            inventory,
            retirement,
            stop,
            reboot,
            roster=authority[1],
            signer=authority[2],
            signer_key_id=authority[1].local_key_id,
        )


def test_signed_independence_receipt_parses_verifies_and_rejects_tamper(
    authority,
    tmp_path: Path,
) -> None:
    _cas, committed, retirement = _committed(authority, tmp_path)
    stop = _scan(authority, "post_stop")
    reboot = _scan(authority, "post_reboot")
    digests = (
        hashlib.sha256(stop.signed_bytes()).hexdigest(),
        hashlib.sha256(reboot.signed_bytes()).hexdigest(),
    )
    inventory = _signed_inventory(authority, scan_digests=digests)
    receipt = verify_synapse_retired(
        committed,
        inventory,
        retirement,
        stop,
        reboot,
        roster=authority[1],
        signer=authority[2],
        signer_key_id=authority[1].local_key_id,
    )

    parsed = independence_receipt_from_dict(receipt.to_dict())
    verify_independence_receipt(
        parsed,
        committed,
        inventory,
        retirement,
        stop,
        reboot,
        roster=authority[1],
    )
    with pytest.raises(CutoverSafetyError):
        verify_independence_receipt(
            replace(parsed, retirement_epoch=9),
            committed,
            inventory,
            retirement,
            stop,
            reboot,
            roster=authority[1],
        )


@pytest.mark.parametrize("failure", ["same_boot", "capture_order", "inventory_binding"])
def test_public_receipt_verifier_repeats_cross_invariants(
    authority,
    tmp_path: Path,
    failure: str,
) -> None:
    _cas, committed, retirement = _committed(authority, tmp_path)
    stop = _scan(
        authority,
        "post_stop",
        captured_at="2026-07-30T02:00:00Z" if failure == "capture_order" else None,
    )
    reboot = _scan(
        authority,
        "post_reboot",
        boot_id=stop.boot_id if failure == "same_boot" else None,
        captured_at="2026-07-30T01:00:00Z" if failure == "capture_order" else None,
    )
    scan_digests = (
        hashlib.sha256(stop.signed_bytes()).hexdigest(),
        hashlib.sha256(reboot.signed_bytes()).hexdigest(),
    )
    inventory_digests = ("8" * 64, "9" * 64) if failure == "inventory_binding" else scan_digests
    inventory = _signed_inventory(authority, scan_digests=inventory_digests)
    receipt = _signed_receipt(authority, committed, inventory, retirement, stop, reboot)

    with pytest.raises(CutoverSafetyError):
        verify_independence_receipt(
            receipt,
            committed,
            inventory,
            retirement,
            stop,
            reboot,
            roster=authority[1],
        )


def test_final_verification_rejects_manifest_and_inventory_signature_forgery(
    authority,
    tmp_path: Path,
) -> None:
    _cas, committed, retirement = _committed(authority, tmp_path)
    stop = _scan(authority, "post_stop")
    reboot = _scan(authority, "post_reboot")
    digests = (
        hashlib.sha256(stop.signed_bytes()).hexdigest(),
        hashlib.sha256(reboot.signed_bytes()).hexdigest(),
    )
    inventory = _signed_inventory(authority, scan_digests=digests)
    for bad_inventory, bad_manifest in (
        (replace(inventory, signer_device_id="mac-b"), retirement),
        (inventory, replace(retirement, active_reference_sha256="0" * 64)),
    ):
        with pytest.raises(CutoverSafetyError, match="signature is invalid"):
            verify_synapse_retired(
                committed,
                bad_inventory,
                bad_manifest,
                stop,
                reboot,
                roster=authority[1],
                signer=authority[2],
                signer_key_id=authority[1].local_key_id,
            )


@pytest.mark.parametrize(
    "adapter",
    [before_listener_start, before_worker_start, before_write, before_fallback],
)
def test_runtime_adapters_fence_before_callback(
    authority,
    tmp_path: Path,
    adapter,
) -> None:
    cas, committed, _retirement = _committed(authority, tmp_path)
    called = False

    def callback() -> None:
        nonlocal called
        called = True

    with pytest.raises(CutoverSafetyError, match=r"synapse\.cutover\.retired"):
        adapter(cas, authority[1], 8, callback)
    assert called is False
    validate_synapse_request(committed, 7, kind="status")


def test_runtime_gate_verifies_current_cas_head_before_every_admission(
    authority,
    tmp_path: Path,
) -> None:
    cas, committed, _retirement = _committed(authority, tmp_path)
    oid, record = cas.read()
    assert cas.compare_and_swap(oid, replace(record, signature="invalid"))
    called = False

    def callback() -> None:
        nonlocal called
        called = True

    with pytest.raises(ControlRecordError, match="signature is invalid"):
        before_write(cas, authority[1], committed.retirement_epoch, callback)
    assert called is False


def test_cli_requires_derived_plan_and_two_signed_scan_receipts(
    authority,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "tools.memflow_absorption.__main__._verification_roster",
        lambda _path: authority[1],
    )
    manifest, preflight_inventory, plan = _real_plan(authority, tmp_path)
    _initial_cas, initial = _initial(authority)
    initial_record = _signed_control(authority)
    plan_dict = {
        "rows": [row.to_dict() for row in plan.rows],
        "digest": plan.digest,
        "covered_surfaces": {
            key: list(values) for key, values in sorted(plan.covered_surfaces.items())
        },
        "inventory_sha256": plan.inventory_sha256,
        "capability_manifest_sha256": plan.capability_manifest_sha256,
    }
    preflight_paths = {
        "control": tmp_path / "preflight-control.json",
        "manifest": tmp_path / "capability.json",
        "inventory": tmp_path / "preflight-inventory.json",
        "plan": tmp_path / "plan.json",
    }
    preflight_paths["control"].write_bytes(canonical_json_bytes(initial_record.to_dict()))
    preflight_paths["manifest"].write_bytes(canonical_json_bytes(manifest.to_dict()))
    preflight_paths["inventory"].write_bytes(canonical_json_bytes(preflight_inventory.to_dict()))
    preflight_paths["plan"].write_bytes(canonical_json_bytes(plan_dict))
    preflight_command = [
        "synapse-preflight",
        "--control-record",
        str(preflight_paths["control"]),
        "--capability-manifest",
        str(preflight_paths["manifest"]),
        "--consumer-inventory",
        str(preflight_paths["inventory"]),
        "--consumer-plan",
        str(preflight_paths["plan"]),
        "--memo-bin",
        str(tmp_path / "runtime" / "memo"),
        "--roster-root",
        str(tmp_path),
    ]
    assert initial.synapse_state is SynapseRetirementState.PREPARING
    assert main(preflight_command) == 0
    assert json.loads(capsys.readouterr().out)["preflight_passed"] is True
    malformed_inventory = preflight_inventory.to_dict()
    malformed_row = ConsumerInventoryRow(
        kind="source",
        location="/tmp/source",
        references=("memflow",),
    ).to_dict()
    malformed_row["kind"] = "rogue"
    malformed_inventory["rows"] = [malformed_row]
    malformed_inventory["signature"] = ""
    malformed_envelope = authority[2].sign(
        domain=CONSUMER_INVENTORY_DOMAIN,
        payload=canonical_json_bytes(malformed_inventory),
        key_id=authority[1].local_key_id,
    )
    malformed_inventory["signature"] = malformed_envelope.signature
    malformed_plan = dict(plan_dict)
    unsigned_malformed_inventory = dict(malformed_inventory)
    unsigned_malformed_inventory["signature"] = ""
    malformed_plan["inventory_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned_malformed_inventory)
    ).hexdigest()
    preflight_paths["inventory"].write_bytes(canonical_json_bytes(malformed_inventory))
    preflight_paths["plan"].write_bytes(canonical_json_bytes(malformed_plan))
    with pytest.raises(SystemExit, match="typed authority verification failed"):
        main(preflight_command)
    preflight_paths["inventory"].write_bytes(canonical_json_bytes(preflight_inventory.to_dict()))
    preflight_paths["plan"].write_bytes(canonical_json_bytes(plan_dict))

    for field, value in (("control_oid", "bad-oid"), ("sequence", 2)):
        malformed_control = initial_record.to_dict()
        malformed_control[field] = value
        malformed_control["signature"] = ""
        envelope = authority[2].sign(
            domain=CONTROL_RECORD_DOMAIN,
            payload=canonical_json_bytes(malformed_control),
            key_id=authority[1].local_key_id,
        )
        malformed_control["signature"] = envelope.signature
        preflight_paths["control"].write_bytes(canonical_json_bytes(malformed_control))
        with pytest.raises(SystemExit, match="typed authority verification failed"):
            main(preflight_command)
    preflight_paths["control"].write_bytes(canonical_json_bytes(initial_record.to_dict()))

    malformed_manifest = manifest.to_dict()
    malformed_manifest["operation_mappings"] = [{"source_operation": "forged"}]
    malformed_manifest["operation_map_sha256"] = hashlib.sha256(
        canonical_json_bytes(
            {
                "mappings": malformed_manifest["operation_mappings"],
                "registry_authority_sha256": "",
                "fixture_authority_sha256": "",
            }
        )
    ).hexdigest()
    malformed_manifest["signature"] = ""
    manifest_envelope = authority[2].sign(
        domain=CAPABILITY_MANIFEST_DOMAIN,
        payload=canonical_json_bytes(malformed_manifest),
        key_id=authority[1].local_key_id,
    )
    malformed_manifest["signature"] = manifest_envelope.signature
    preflight_paths["manifest"].write_bytes(canonical_json_bytes(malformed_manifest))
    with pytest.raises(SystemExit, match="typed authority verification failed"):
        main(preflight_command)
    preflight_paths["manifest"].write_bytes(canonical_json_bytes(manifest.to_dict()))

    forged_row = ConsumerReplacement(
        old_label="com.synapse.self-signed",
        new_label="com.memo.self-signed",
        command=(str(tmp_path / "runtime" / "memo"), "save"),
        owner="memo_native",
        restart_required=True,
        config_sha256="7" * 64,
        rollback_action="operator",
    )
    forged_plan = dict(plan_dict)
    forged_plan["rows"] = [forged_row.to_dict()]
    forged_plan["digest"] = hashlib.sha256(canonical_json_bytes(forged_plan["rows"])).hexdigest()
    preflight_paths["plan"].write_bytes(canonical_json_bytes(forged_plan))
    with pytest.raises(SystemExit, match="deterministic verified authority"):
        main(preflight_command)

    cas, committed, retirement = _committed(authority, tmp_path)
    stop = _scan(authority, "post_stop")
    reboot = _scan(authority, "post_reboot")
    scan_digests = (
        hashlib.sha256(stop.signed_bytes()).hexdigest(),
        hashlib.sha256(reboot.signed_bytes()).hexdigest(),
    )
    inventory = _signed_inventory(authority, scan_digests=scan_digests)
    receipt = verify_synapse_retired(
        committed,
        inventory,
        retirement,
        stop,
        reboot,
        roster=authority[1],
        signer=authority[2],
        signer_key_id=authority[1].local_key_id,
    )
    committed_record = cas.read()[1]
    verify_paths = {
        "control": tmp_path / "committed.json",
        "manifest": tmp_path / "retirement.json",
        "inventory": tmp_path / "final-inventory.json",
        "stop": tmp_path / "post-stop.json",
        "reboot": tmp_path / "post-reboot.json",
        "receipt": tmp_path / "independence.json",
    }
    for key, payload in (
        ("control", committed_record.to_dict()),
        ("manifest", retirement.to_dict()),
        ("inventory", inventory.to_dict()),
        ("stop", stop.to_dict()),
        ("reboot", reboot.to_dict()),
        ("receipt", receipt.to_dict()),
    ):
        verify_paths[key].write_bytes(canonical_json_bytes(payload))
    with pytest.raises(SystemExit, match="receipt is not committed"):
        main(
            [
                "synapse-verify",
                "--control-record",
                str(verify_paths["control"]),
                "--inventory",
                str(verify_paths["inventory"]),
                "--retirement-manifest",
                str(verify_paths["manifest"]),
                "--post-stop-scan",
                str(verify_paths["stop"]),
                "--post-reboot-scan",
                str(verify_paths["reboot"]),
                "--independence-receipt",
                str(verify_paths["receipt"]),
                "--roster-root",
                str(tmp_path),
            ]
        )
    verified = advance_synapse_retirement(
        cas,
        committed,
        SynapseRetirementState.VERIFIED,
        roster=authority[1],
        signer=authority[2],
        next_control_oid="1" * 40,
        independence_receipt=receipt,
        independence_inventory=inventory,
        independence_manifest=retirement,
        post_stop_scan=stop,
        post_reboot_scan=reboot,
    )
    verify_paths["control"].write_bytes(canonical_json_bytes(cas.read()[1].to_dict()))
    assert (
        verified.independence_receipt_sha256 == hashlib.sha256(receipt.signed_bytes()).hexdigest()
    )
    assert (
        main(
            [
                "synapse-verify",
                "--control-record",
                str(verify_paths["control"]),
                "--inventory",
                str(verify_paths["inventory"]),
                "--retirement-manifest",
                str(verify_paths["manifest"]),
                "--post-stop-scan",
                str(verify_paths["stop"]),
                "--post-reboot-scan",
                str(verify_paths["reboot"]),
                "--independence-receipt",
                str(verify_paths["receipt"]),
                "--roster-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["independent"] is True
    committed_record = replace(
        committed_record,
        control_oid=verified.control_oid,
        state=CutoverState.VERIFIED,
        sequence=committed.sequence + 1,
        previous_control_oid=committed.control_oid,
        issued_at="2026-07-30T04:00:00Z",
        signature="",
        synapse_state=SynapseRetirementState.VERIFIED,
        independence_receipt_sha256="9" * 64,
    )
    wrong_digest_control = sign_control_record(
        committed_record,
        signer=authority[2],
        predecessor=committed,
    )
    verify_paths["control"].write_bytes(canonical_json_bytes(wrong_digest_control.to_dict()))
    with pytest.raises(SystemExit, match="does not commit"):
        main(
            [
                "synapse-verify",
                "--control-record",
                str(verify_paths["control"]),
                "--inventory",
                str(verify_paths["inventory"]),
                "--retirement-manifest",
                str(verify_paths["manifest"]),
                "--post-stop-scan",
                str(verify_paths["stop"]),
                "--post-reboot-scan",
                str(verify_paths["reboot"]),
                "--independence-receipt",
                str(verify_paths["receipt"]),
                "--roster-root",
                str(tmp_path),
            ]
        )
