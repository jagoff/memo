from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memo.operational_event import canonical_json_bytes
from memo.operational_key_store import (
    AuthorityPinStore,
    DeviceKeyStore,
    InMemoryAuthorityPinProvider,
)
from memo.operational_roster import VerificationRoster
from memo.operational_signing import OperationalSigner
from tools.memflow_absorption import __main__ as absorption_cli
from tools.memflow_absorption.inventory import (
    SYNAPSE_RETIREMENT_DOMAIN,
    InventoryError,
    build_independence_receipt,
)
from tools.memflow_absorption.safety import (
    CutoverSafetyError,
    assert_retirement_cleanup_authority,
)
from tools.memflow_absorption.schemas import (
    CutoverState,
    SynapseOperation,
    SynapseRetirementManifest,
    SynapseRetirementState,
    VerifiedControlRecord,
)


def _manifest() -> SynapseRetirementManifest:
    return SynapseRetirementManifest(
        schema="memo.synapse_retirement.v2",
        source_commit="a" * 40,
        files=("src/synapse/runtime.py",),
        symbols=("runtime_loop",),
        tests=("tests/test_runtime.py",),
        goldens=("tests/goldens/runtime.json",),
        active_reference_sha256="b" * 64,
        signer_key_id="origin-key",
        signature="signed",
    )


def _signed_manifest(
    authority_root: Path,
) -> tuple[SynapseRetirementManifest, VerificationRoster]:
    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=key,
        root=authority_root,
        pin_store=AuthorityPinStore._for_test(
            authority_root,
            provider=InMemoryAuthorityPinProvider(),
        ),
    )
    unsigned = replace(
        _manifest(),
        signer_key_id=roster.local_key_id,
        signature="",
        operations=(
            SynapseOperation(
                source_operation="synapse.runtime.loop",
                source_files=("src/synapse/runtime.py",),
                source_symbols=("runtime_loop",),
                consumers=("com.synapse.runtime",),
                daemon_routes=("runtime",),
                exclusion_reason="self_audit",
                fixture_paths=(),
            ),
        ),
    )
    envelope = OperationalSigner(
        keys,
        roster_version=roster.version,
    ).sign(
        domain=SYNAPSE_RETIREMENT_DOMAIN,
        payload=unsigned.signed_bytes(),
        key_id=roster.local_key_id,
    )
    return replace(unsigned, signature=envelope.signature), roster


def _verified_control(
    *,
    manifest_sha256: str,
    independence_sha256: str,
) -> VerifiedControlRecord:
    return VerifiedControlRecord(
        control_oid="c" * 40,
        canonical_payload=b"verified-control",
        state=CutoverState.VERIFIED,
        sequence=6,
        previous_control_oid="d" * 40,
        roster_version=1,
        verified_at="2026-07-30T21:00:00Z",
        signer_device_id="device-a",
        signer_key_id="origin-key",
        attempt_id="attempt-123",
        synapse_state=SynapseRetirementState.VERIFIED,
        synapse_manifest_sha256=manifest_sha256,
        consumer_plan_sha256="3" * 64,
        retirement_epoch=1,
        independence_receipt_sha256=independence_sha256,
    )


def test_retirement_audit_rejects_unlisted_reference(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        "command = 'synapse-mcp'",
        encoding="utf-8",
    )

    with pytest.raises(InventoryError, match="unlisted active reference"):
        build_independence_receipt((tmp_path,), manifest=_manifest())


def test_retirement_audit_accepts_clean_tree(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'memo'\n",
        encoding="utf-8",
    )

    receipt = build_independence_receipt((tmp_path,), manifest=_manifest())

    assert receipt.status == "verified"
    assert receipt.file_count == 1
    assert len(receipt.source_scan_sha256) == 64


def test_retirement_audit_cli_report_combines_terminal_authority_and_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(_manifest().to_dict()))
    installed = tmp_path / "installed"
    installed.mkdir()
    (installed / "pyproject.toml").write_text(
        "[project]\nname = 'memo'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        absorption_cli,
        "_synapse_verify",
        lambda _args: {
            "independent": True,
            "retirement_epoch": 7,
            "synapse_manifest_sha256": "1" * 64,
            "consumer_inventory_sha256": "2" * 64,
            "independence_receipt_sha256": "3" * 64,
        },
    )

    result = absorption_cli._retirement_audit(
        SimpleNamespace(
            retirement_manifest=manifest_path,
            scan_root=[installed],
            archive_root=[],
        )
    )

    assert result["command"] == "retirement-audit"
    assert result["status"] == "verified"
    assert result["independent"] is True
    assert {
        "source",
        "runtime",
        "configuration",
        "process",
        "port",
        "launchagent",
        "mcp_gateway_route",
        "wrapper",
        "state_root",
        "package_metadata",
    } <= set(result["covered_surfaces"])


def test_retirement_audit_allows_only_manifest_listed_archived_provenance(
    tmp_path: Path,
) -> None:
    installed = tmp_path / "installed"
    archive = tmp_path / "archive"
    installed.mkdir()
    archived_source = archive / "src" / "synapse" / "runtime.py"
    archived_source.parent.mkdir(parents=True)
    archived_source.write_text("def synapse_runtime(): ...\n", encoding="utf-8")
    manifest, roster = _signed_manifest(tmp_path / "authority")

    with pytest.raises(InventoryError, match="roster-verified"):
        build_independence_receipt(
            (installed,),
            manifest=manifest,
            archived_roots=(archive,),
        )

    receipt = build_independence_receipt(
        (installed,),
        manifest=manifest,
        archived_roots=(archive,),
        roster=roster,
    )

    assert receipt.status == "verified"
    assert receipt.archived_provenance == ("src/synapse/runtime.py",)

    (archive / "config.toml").write_text("SYNAPSE_TOKEN='live'", encoding="utf-8")
    with pytest.raises(InventoryError, match="unlisted active reference"):
        build_independence_receipt(
            (installed,),
            manifest=manifest,
            archived_roots=(archive,),
            roster=roster,
        )


def test_retirement_audit_rejects_symlink_blind_spot(tmp_path: Path) -> None:
    installed = tmp_path / "installed"
    installed.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("synapse runtime", encoding="utf-8")
    (installed / "runtime").symlink_to(outside)

    with pytest.raises(InventoryError, match="symlink"):
        build_independence_receipt((installed,), manifest=_manifest())


def test_cleanup_authority_rejects_non_verified_control(tmp_path: Path) -> None:
    manifest_sha256 = "1" * 64
    independence_sha256 = "2" * 64
    control = replace(
        _verified_control(
            manifest_sha256=manifest_sha256,
            independence_sha256=independence_sha256,
        ),
        state=CutoverState.ACTIVATED,
        synapse_state=SynapseRetirementState.COMMITTED,
    )

    with pytest.raises(CutoverSafetyError, match="VERIFIED"):
        assert_retirement_cleanup_authority(
            control,
            expected_digests=_expected_digests(
                manifest_sha256=manifest_sha256,
                independence_sha256=independence_sha256,
            ),
            observed_digests=_expected_digests(
                manifest_sha256=manifest_sha256,
                independence_sha256=independence_sha256,
            ),
            cleanup_paths=(tmp_path / "synapse-state",),
        )


def _expected_digests(
    *,
    manifest_sha256: str = "1" * 64,
    independence_sha256: str = "2" * 64,
) -> dict[str, str]:
    return {
        "control_record": hashlib.sha256(b"control-record").hexdigest(),
        "retirement_manifest": manifest_sha256,
        "consumer_replacement_receipt": "3" * 64,
        "bounded_data_receipt": "4" * 64,
        "independence_receipt": independence_sha256,
    }


def test_cleanup_authority_requires_all_exact_artifact_digests(
    tmp_path: Path,
) -> None:
    expected = _expected_digests()
    control = _verified_control(
        manifest_sha256=expected["retirement_manifest"],
        independence_sha256=expected["independence_receipt"],
    )
    observed = dict(expected)
    observed.pop("bounded_data_receipt")

    with pytest.raises(CutoverSafetyError, match="exact artifact digests"):
        assert_retirement_cleanup_authority(
            control,
            expected_digests=expected,
            observed_digests=observed,
            cleanup_paths=(tmp_path / "synapse-state",),
        )

    observed = dict(expected)
    observed["bounded_data_receipt"] = "5" * 64
    with pytest.raises(CutoverSafetyError, match="digest mismatch"):
        assert_retirement_cleanup_authority(
            control,
            expected_digests=expected,
            observed_digests=observed,
            cleanup_paths=(tmp_path / "synapse-state",),
        )


@pytest.mark.parametrize("unsafe_path", [Path("/"), Path("$STATE/synapse")])
def test_cleanup_authority_rejects_broad_or_unresolved_paths(
    tmp_path: Path,
    unsafe_path: Path,
) -> None:
    expected = _expected_digests()
    control = _verified_control(
        manifest_sha256=expected["retirement_manifest"],
        independence_sha256=expected["independence_receipt"],
    )

    with pytest.raises(CutoverSafetyError, match="cleanup path"):
        assert_retirement_cleanup_authority(
            control,
            expected_digests=expected,
            observed_digests=expected,
            cleanup_paths=(unsafe_path,),
        )


def test_cleanup_remains_blocked_without_runtime_gate_and_signed_path_plan(
    tmp_path: Path,
) -> None:
    expected = _expected_digests()
    control = _verified_control(
        manifest_sha256=expected["retirement_manifest"],
        independence_sha256=expected["independence_receipt"],
    )
    cleanup_target = tmp_path / "synapse-state"
    cleanup_target.mkdir()
    marker = cleanup_target / "must-survive"
    marker.write_text("not deleted", encoding="utf-8")

    with pytest.raises(CutoverSafetyError, match="runtime gate evidence"):
        assert_retirement_cleanup_authority(
            control,
            expected_digests=expected,
            observed_digests=expected,
            cleanup_paths=(cleanup_target,),
        )

    assert marker.read_text(encoding="utf-8") == "not deleted"


def test_cleanup_cli_is_refusal_only_even_with_exact_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = {
        "control": {"artifact": "control"},
        "manifest": {"artifact": "manifest"},
        "plan": {"artifact": "plan"},
        "data": {"artifact": "data"},
        "independence": {"artifact": "independence"},
    }
    paths: dict[str, Path] = {}
    for name, value in objects.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_bytes(canonical_json_bytes(value))

    manifest = _manifest()
    plan_bytes = canonical_json_bytes(objects["plan"])
    independence_bytes = canonical_json_bytes(objects["independence"])
    expected = {
        "control_record": hashlib.sha256(
            canonical_json_bytes(objects["control"])
        ).hexdigest(),
        "retirement_manifest": hashlib.sha256(
            manifest.signed_bytes()
        ).hexdigest(),
        "consumer_replacement_receipt": hashlib.sha256(plan_bytes).hexdigest(),
        "bounded_data_receipt": hashlib.sha256(
            canonical_json_bytes(objects["data"])
        ).hexdigest(),
        "independence_receipt": hashlib.sha256(independence_bytes).hexdigest(),
    }
    control = _verified_control(
        manifest_sha256=expected["retirement_manifest"],
        independence_sha256=expected["independence_receipt"],
    )
    control = replace(
        control,
        consumer_plan_sha256=expected["consumer_replacement_receipt"],
    )
    monkeypatch.setattr(absorption_cli, "_verification_roster", lambda _path: object())
    monkeypatch.setattr(
        absorption_cli,
        "control_record_from_dict",
        lambda _value: SimpleNamespace(control_oid=control.control_oid),
    )
    monkeypatch.setattr(
        absorption_cli,
        "verify_control_record",
        lambda **_kwargs: control,
    )
    monkeypatch.setattr(
        absorption_cli,
        "synapse_retirement_manifest_from_dict",
        lambda _value: manifest,
    )
    monkeypatch.setattr(
        absorption_cli,
        "verify_synapse_retirement_manifest",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        absorption_cli,
        "consumer_replacement_plan_from_dict",
        lambda _value: SimpleNamespace(authority_bytes=lambda: plan_bytes),
    )
    monkeypatch.setattr(
        absorption_cli,
        "independence_receipt_from_dict",
        lambda _value: SimpleNamespace(signed_bytes=lambda: independence_bytes),
    )
    cleanup_target = tmp_path / "state-to-keep"
    cleanup_target.mkdir()
    marker = cleanup_target / "marker"
    marker.write_text("preserved", encoding="utf-8")
    arguments: dict[str, Any] = {
        "control_record": paths["control"],
        "retirement_manifest": paths["manifest"],
        "consumer_replacement_receipt": paths["plan"],
        "bounded_data_receipt": paths["data"],
        "independence_receipt": paths["independence"],
        "roster_root": tmp_path,
        "cleanup_path": [cleanup_target],
    }
    arguments.update({f"{key}_sha256": value for key, value in expected.items()})

    with pytest.raises(SystemExit, match="runtime gate evidence"):
        absorption_cli._retirement_cleanup(SimpleNamespace(**arguments))

    assert marker.read_text(encoding="utf-8") == "preserved"
