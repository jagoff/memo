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
from memo.operational_signing import OperationalSigner
from tools.memflow_absorption.__main__ import main
from tools.memflow_absorption.control_record import (
    ControlRecordError,
    CutoverSafetyError,
    advance_synapse_retirement,
    commit_synapse_activation,
    prepare_synapse_retirement,
    sign_control_record,
    validate_synapse_request,
)
from tools.memflow_absorption.inventory import (
    CONSUMER_INVENTORY_DOMAIN,
    SYNAPSE_RETIREMENT_DOMAIN,
)
from tools.memflow_absorption.manifest import CAPABILITY_MANIFEST_DOMAIN
from tools.memflow_absorption.safety import verify_synapse_retired
from tools.memflow_absorption.schemas import (
    CapabilityManifest,
    ConsumerInventory,
    ConsumerInventoryRow,
    ConsumerReplacementPlan,
    CutoverControlRecord,
    CutoverState,
    SynapseRetirementManifest,
    SynapseRetirementState,
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
    return keys, roster, OperationalSigner(keys, roster_version=roster.version)


def _signed_control(
    authority: tuple[DeviceKeyStore, VerificationRoster, OperationalSigner],
) -> CutoverControlRecord:
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
            signer_device_id="device-a",
            signer_key_id=roster.local_key_id,
            issued_at="2026-07-30T00:00:00Z",
            signature="",
        ),
        signer=signer,
    )


def _signed_capability_manifest(
    authority: tuple[DeviceKeyStore, VerificationRoster, OperationalSigner],
) -> CapabilityManifest:
    _keys, roster, signer = authority
    operation_map_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "mappings": [],
                "registry_authority_sha256": "",
                "fixture_authority_sha256": "",
            }
        )
    ).hexdigest()
    slo_baseline_sha256 = hashlib.sha256(canonical_json_bytes([])).hexdigest()
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
        operation_map_sha256=operation_map_sha256,
        slo_baseline_sha256=slo_baseline_sha256,
        blockers=(),
        frozen=True,
        signer_device_id="device-a",
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


def _consumer_plan() -> ConsumerReplacementPlan:
    return ConsumerReplacementPlan(
        rows=(),
        digest=hashlib.sha256(canonical_json_bytes([])).hexdigest(),
        covered_surfaces={surface: (f"synapse:{surface}",) for surface in SURFACES},
    )


def _signed_retirement_manifest(
    authority: tuple[DeviceKeyStore, VerificationRoster, OperationalSigner],
) -> SynapseRetirementManifest:
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
    )
    envelope = signer.sign(
        domain=SYNAPSE_RETIREMENT_DOMAIN,
        payload=unsigned.signed_bytes(),
        key_id=roster.local_key_id,
    )
    return replace(unsigned, signature=envelope.signature)


def _signed_inventory(
    authority: tuple[DeviceKeyStore, VerificationRoster, OperationalSigner],
    *,
    rows: tuple[ConsumerInventoryRow, ...] = (),
    phases: tuple[str, ...] = ("post_stop", "post_reboot"),
) -> ConsumerInventory:
    _keys, roster, signer = authority
    unsigned = ConsumerInventory(
        schema="memo.cutover_consumer_inventory.v1",
        rows=rows,
        blockers=(),
        source_scan_sha256="e" * 64,
        signer_device_id="device-a",
        signer_key_id=roster.local_key_id,
        roster_version=roster.version,
        signature="",
        verification_phases=phases,  # type: ignore[arg-type]
        covered_surfaces=SURFACES,
    )
    envelope = signer.sign(
        domain=CONSUMER_INVENTORY_DOMAIN,
        payload=unsigned.signed_bytes(),
        key_id=roster.local_key_id,
    )
    return replace(unsigned, signature=envelope.signature)


def _sign_transition(
    record: CutoverControlRecord,
    authority: tuple[DeviceKeyStore, VerificationRoster, OperationalSigner],
) -> CutoverControlRecord:
    return sign_control_record(record, signer=authority[2])


def _staged_control(
    authority: tuple[DeviceKeyStore, VerificationRoster, OperationalSigner],
) -> tuple[CutoverControlRecord, SynapseRetirementManifest]:
    ready = prepare_synapse_retirement(
        _signed_control(authority),
        _signed_capability_manifest(authority),
        _consumer_plan(),
    )
    quiesced = advance_synapse_retirement(
        _sign_transition(ready, authority),
        SynapseRetirementState.QUIESCED,
        peer_vote_sha256=("1" * 64, "2" * 64),
    )
    retirement_manifest = _signed_retirement_manifest(authority)
    staged = advance_synapse_retirement(
        _sign_transition(quiesced, authority),
        SynapseRetirementState.STAGED,
        active_state_receipt_sha256="3" * 64,
        synapse_manifest_sha256=hashlib.sha256(
            retirement_manifest.signed_bytes()
        ).hexdigest(),
    )
    return _sign_transition(staged, authority), retirement_manifest


def test_retired_synapse_refuses_startup_before_listener(authority) -> None:
    staged, _manifest = _staged_control(authority)
    committed = replace(
        staged,
        state=CutoverState.EPOCH_COMMITTED,
        synapse_state=SynapseRetirementState.COMMITTED,
        retirement_epoch=8,
    )

    with pytest.raises(CutoverSafetyError, match=r"synapse\.cutover\.retired"):
        prepare_synapse_retirement(
            committed,
            _signed_capability_manifest(authority),
            _consumer_plan(),
        )


def test_stale_synapse_epoch_cannot_write_after_memo_commit(authority) -> None:
    staged, _manifest = _staged_control(authority)
    committed = commit_synapse_activation(staged, epoch=8)

    with pytest.raises(CutoverSafetyError, match="stale activation epoch"):
        validate_synapse_request(committed, epoch=7)
    with pytest.raises(CutoverSafetyError, match=r"synapse\.cutover\.retired"):
        validate_synapse_request(committed, epoch=8, kind="startup")
    validate_synapse_request(committed, epoch=7, kind="status")


def test_state_machine_rejects_offline_peer_skips_digest_change_and_second_epoch(
    authority,
) -> None:
    ready = prepare_synapse_retirement(
        _signed_control(authority),
        _signed_capability_manifest(authority),
        _consumer_plan(),
    )
    signed_ready = _sign_transition(ready, authority)
    with pytest.raises(CutoverSafetyError, match="two distinct peer"):
        advance_synapse_retirement(
            signed_ready,
            SynapseRetirementState.QUIESCED,
            peer_vote_sha256=("1" * 64,),
        )
    with pytest.raises(CutoverSafetyError, match="stale or skipped"):
        advance_synapse_retirement(
            signed_ready,
            SynapseRetirementState.STAGED,
            active_state_receipt_sha256="3" * 64,
            synapse_manifest_sha256="4" * 64,
        )
    with pytest.raises(ControlRecordError, match="authority digests changed"):
        _sign_transition(replace(ready, consumer_plan_sha256="f" * 64), authority)

    staged, _manifest = _staged_control(authority)
    with pytest.raises(CutoverSafetyError, match="second Synapse activation epoch"):
        commit_synapse_activation(replace(staged, retirement_epoch=7), epoch=8)


def test_abort_is_the_only_precommit_failure_branch(authority) -> None:
    ready = prepare_synapse_retirement(
        _signed_control(authority),
        _signed_capability_manifest(authority),
        _consumer_plan(),
    )
    aborted = advance_synapse_retirement(
        _sign_transition(ready, authority),
        SynapseRetirementState.ABORTED,
    )
    signed_aborted = _sign_transition(aborted, authority)

    assert signed_aborted.state is CutoverState.ABORTED
    with pytest.raises(CutoverSafetyError, match="stale or skipped"):
        advance_synapse_retirement(
            signed_aborted,
            SynapseRetirementState.QUIESCED,
            peer_vote_sha256=("1" * 64, "2" * 64),
        )


@pytest.mark.parametrize("kind", ["process", "launchd"])
def test_final_independence_rejects_resurrected_runtime(authority, kind: str) -> None:
    staged, retirement_manifest = _staged_control(authority)
    committed = commit_synapse_activation(staged, epoch=8)
    row = ConsumerInventoryRow(
        kind=kind,  # type: ignore[arg-type]
        location=f"live:{kind}",
        references=("synapse",),
        active=True,
    )

    with pytest.raises(CutoverSafetyError, match="active reference resurrected"):
        verify_synapse_retired(
            committed,
            _signed_inventory(authority, rows=(row,)),
            retirement_manifest,
        )


def test_final_independence_requires_both_scans_and_returns_bound_receipt(
    authority,
) -> None:
    staged, retirement_manifest = _staged_control(authority)
    committed = commit_synapse_activation(staged, epoch=8)
    with pytest.raises(CutoverSafetyError, match="post-stop and post-reboot"):
        verify_synapse_retired(
            committed,
            _signed_inventory(authority, phases=("post_stop",)),
            retirement_manifest,
        )

    receipt = verify_synapse_retired(
        committed,
        _signed_inventory(authority),
        retirement_manifest,
    )

    assert receipt.attempt_id == "attempt-123"
    assert receipt.retirement_epoch == 8
    assert receipt.verification_phases == ("post_stop", "post_reboot")
    assert len(receipt.sha256) == 64


def test_cli_preflight_verifies_signatures_and_never_applies(
    authority,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tools.memflow_absorption.__main__._verification_roster",
        lambda _path: authority[1],
    )
    control = _signed_control(authority)
    manifest = _signed_capability_manifest(authority)
    plan = _consumer_plan()
    paths = {
        "control": tmp_path / "control.json",
        "manifest": tmp_path / "manifest.json",
        "plan": tmp_path / "plan.json",
    }
    paths["control"].write_bytes(canonical_json_bytes(control.to_dict()))
    paths["manifest"].write_bytes(canonical_json_bytes(manifest.to_dict()))
    paths["plan"].write_bytes(
        canonical_json_bytes(
            {
                "rows": [],
                "digest": plan.digest,
                "covered_surfaces": {
                    key: list(values)
                    for key, values in sorted(plan.covered_surfaces.items())
                },
            }
        )
    )
    command = [
        "synapse-preflight",
        "--control-record",
        str(paths["control"]),
        "--capability-manifest",
        str(paths["manifest"]),
        "--consumer-plan",
        str(paths["plan"]),
        "--roster-root",
        str(tmp_path),
    ]

    assert main(command) == 0
    assert json.loads(capsys.readouterr().out)["preflight_passed"] is True
    with pytest.raises(SystemExit, match="inspection-only"):
        main([*command, "--apply"])

    tampered = manifest.to_dict()
    tampered["machine_ids"] = ["mac-a", "mac-c"]
    paths["manifest"].write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(SystemExit, match="signature is invalid"):
        main(command)


def test_cli_verify_fails_closed_on_loaded_launchagent(
    authority,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tools.memflow_absorption.__main__._verification_roster",
        lambda _path: authority[1],
    )
    staged, retirement_manifest = _staged_control(authority)
    committed = sign_control_record(
        replace(
            staged,
            state=CutoverState.EPOCH_COMMITTED,
            synapse_state=SynapseRetirementState.COMMITTED,
            retirement_epoch=8,
            signature="",
        ),
        signer=authority[2],
    )
    loaded = ConsumerInventoryRow(
        kind="launchd",
        location="com.synapse.gateway:/Library/LaunchAgents/com.synapse.gateway.plist",
        references=("synapse",),
        active=True,
    )
    inventory = _signed_inventory(authority, rows=(loaded,))
    control_path = tmp_path / "committed.json"
    inventory_path = tmp_path / "inventory.json"
    manifest_path = tmp_path / "retirement.json"
    control_path.write_bytes(canonical_json_bytes(committed.to_dict()))
    inventory_path.write_bytes(canonical_json_bytes(inventory.to_dict()))
    manifest_path.write_bytes(canonical_json_bytes(retirement_manifest.to_dict()))

    with pytest.raises(SystemExit, match="active reference resurrected"):
        main(
            [
                "synapse-verify",
                "--control-record",
                str(control_path),
                "--inventory",
                str(inventory_path),
                "--retirement-manifest",
                str(manifest_path),
                "--roster-root",
                str(tmp_path),
            ]
        )
