from __future__ import annotations

import hashlib
import os
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
from tools.memflow_absorption import safety as safety_module
from tools.memflow_absorption.inventory import (
    SYNAPSE_RETIREMENT_DOMAIN,
    InventoryError,
    build_independence_receipt,
)
from tools.memflow_absorption.safety import (
    CutoverSafetyError,
    assert_retirement_cleanup_authority,
    load_verification_roster_readonly,
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
    manifest_sha256 = hashlib.sha256(_manifest().signed_bytes()).hexdigest()
    monkeypatch.setattr(
        absorption_cli,
        "_synapse_verify",
        lambda _args: {
            "independent": True,
            "retirement_epoch": 7,
            "synapse_manifest_sha256": manifest_sha256,
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
    assert result["synapse_manifest_sha256"] == manifest_sha256
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


def test_retirement_audit_rejects_manifest_changed_after_authority_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(_manifest().to_dict()))
    installed = tmp_path / "installed"
    installed.mkdir()
    monkeypatch.setattr(
        absorption_cli,
        "_synapse_verify",
        lambda _args: {
            "independent": True,
            "retirement_epoch": 7,
            "synapse_manifest_sha256": "0" * 64,
            "consumer_inventory_sha256": "2" * 64,
            "independence_receipt_sha256": "3" * 64,
        },
    )

    with pytest.raises(SystemExit, match="changed after verification"):
        absorption_cli._retirement_audit(
            SimpleNamespace(
                retirement_manifest=manifest_path,
                scan_root=[installed],
                archive_root=[],
            )
        )


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


def test_retirement_audit_propagates_walk_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def denied_listdir(_descriptor: int) -> list[str]:
        raise PermissionError("denied")

    monkeypatch.setattr(os, "listdir", denied_listdir)

    with pytest.raises(InventoryError, match="traversal failed"):
        build_independence_receipt((tmp_path,), manifest=_manifest())


def test_retirement_audit_scans_ascii_references_inside_non_utf8_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "runtime.bin").write_bytes(b"\xff\x00SYNAPSE_TOKEN=active\xfe")

    with pytest.raises(InventoryError, match="unlisted active reference"):
        build_independence_receipt((tmp_path,), manifest=_manifest())


@pytest.mark.parametrize(
    "failure",
    [FileNotFoundError("vanished"), PermissionError("denied")],
)
def test_retirement_audit_rejects_entry_descriptor_stat_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: OSError,
) -> None:
    target = tmp_path / "listed.txt"
    target.write_text("clean", encoding="utf-8")
    original_stat = os.stat

    def failing_stat(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes] | int,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        if path == target.name and dir_fd is not None and follow_symlinks is False:
            raise failure
        return original_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(os, "stat", failing_stat)

    with pytest.raises(InventoryError, match="cannot classify inventory entry"):
        build_independence_receipt((tmp_path,), manifest=_manifest())


def test_retirement_audit_rejects_special_files(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "runtime.pipe")

    with pytest.raises(InventoryError, match="unsupported inventory entry type"):
        build_independence_receipt((tmp_path,), manifest=_manifest())


@pytest.mark.parametrize("replacement_kind", ["symlink", "directory"])
def test_retirement_audit_rejects_directory_swap_before_descent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    scan_root = tmp_path / "installed"
    child = scan_root / "runtime"
    child.mkdir(parents=True)
    (child / "clean.txt").write_text("clean", encoding="utf-8")
    parked = tmp_path / "parked-runtime"
    outside = tmp_path / "outside-runtime"
    outside.mkdir()
    (outside / "active.txt").write_text("SYNAPSE_TOKEN=active", encoding="utf-8")
    original_open = os.open
    swapped = False

    def swapping_open(
        path: str | bytes,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if not swapped and path == child.name and dir_fd is not None and flags & os.O_DIRECTORY:
            child.rename(parked)
            if replacement_kind == "symlink":
                child.symlink_to(outside, target_is_directory=True)
            else:
                child.mkdir()
            swapped = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swapping_open)

    with pytest.raises(InventoryError, match=r"cannot descend safely|changed identity"):
        build_independence_receipt((scan_root,), manifest=_manifest())

    assert swapped is True


@pytest.mark.parametrize("change_kind", ["membership", "identity"])
def test_retirement_audit_rejects_directory_change_after_listing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change_kind: str,
) -> None:
    scan_root = tmp_path / "installed"
    scan_root.mkdir()
    target = scan_root / "a-clean.txt"
    target.write_text("clean", encoding="utf-8")
    (scan_root / "z-trigger.txt").write_text("clean", encoding="utf-8")
    parked = tmp_path / "parked-clean.txt"
    original_read = os.read
    changed = False
    data_reads = 0

    def changing_read(descriptor: int, length: int) -> bytes:
        nonlocal changed, data_reads
        data = original_read(descriptor, length)
        if data:
            data_reads += 1
        should_change = change_kind == "membership" or data_reads == 2
        if data and not changed and should_change:
            if change_kind == "membership":
                (scan_root / "late.txt").write_text(
                    "SYNAPSE_TOKEN=active",
                    encoding="utf-8",
                )
            else:
                target.rename(parked)
                target.write_text("clean", encoding="utf-8")
            changed = True
        return data

    monkeypatch.setattr(os, "read", changing_read)

    with pytest.raises(InventoryError, match=r"membership changed|changed identity"):
        build_independence_receipt((scan_root,), manifest=_manifest())

    assert changed is True


def test_retirement_audit_revalidates_closed_child_after_sibling_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scan_root = tmp_path / "installed"
    first = scan_root / "a"
    sibling = scan_root / "z"
    first.mkdir(parents=True)
    sibling.mkdir()
    (first / "clean.txt").write_text("clean", encoding="utf-8")
    (sibling / "trigger.txt").write_text("trigger", encoding="utf-8")
    original_read = os.read
    changed = False

    def changing_read(descriptor: int, length: int) -> bytes:
        nonlocal changed
        data = original_read(descriptor, length)
        if data == b"trigger" and not changed:
            (first / "late.txt").write_text(
                "SYNAPSE_TOKEN=active",
                encoding="utf-8",
            )
            changed = True
        return data

    monkeypatch.setattr(os, "read", changing_read)

    with pytest.raises(InventoryError, match="inventory directory membership changed"):
        build_independence_receipt((scan_root,), manifest=_manifest())

    assert changed is True


def test_retirement_audit_rejects_file_swap_before_descriptor_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scan_root = tmp_path / "installed"
    scan_root.mkdir()
    target = scan_root / "runtime.txt"
    target.write_text("SYNAPSE_TOKEN=active", encoding="utf-8")
    parked = tmp_path / "parked-runtime.txt"
    original_open = os.open
    swapped = False

    def swapping_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if (
            not swapped
            and path == target.name
            and dir_fd is not None
            and not flags & os.O_DIRECTORY
        ):
            target.rename(parked)
            target.write_text("clean", encoding="utf-8")
            swapped = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swapping_open)

    with pytest.raises(InventoryError, match="inventory file changed identity"):
        build_independence_receipt((scan_root,), manifest=_manifest())

    assert swapped is True


def test_retirement_audit_rejects_file_metadata_change_while_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "runtime.txt"
    target.write_text("clean", encoding="utf-8")
    original_read = os.read
    changed = False

    def changing_read(descriptor: int, length: int) -> bytes:
        nonlocal changed
        data = original_read(descriptor, length)
        if data and not changed:
            target.write_text("SYNAPSE_TOKEN=active", encoding="utf-8")
            changed = True
        return data

    monkeypatch.setattr(os, "read", changing_read)

    with pytest.raises(InventoryError, match="inventory file changed while reading"):
        build_independence_receipt((tmp_path,), manifest=_manifest())

    assert changed is True


def test_retirement_audit_rejects_root_ancestor_swap_during_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ancestor = tmp_path / "authority"
    scan_root = ancestor / "installed"
    scan_root.mkdir(parents=True)
    (scan_root / "clean.txt").write_text("clean", encoding="utf-8")
    replacement = tmp_path / "replacement-authority"
    replacement_root = replacement / "installed"
    replacement_root.mkdir(parents=True)
    (replacement_root / "clean.txt").write_text("clean", encoding="utf-8")
    parked = tmp_path / "parked-authority"
    original_read = os.read
    swapped = False

    def swapping_read(descriptor: int, length: int) -> bytes:
        nonlocal swapped
        data = original_read(descriptor, length)
        if data and not swapped:
            ancestor.rename(parked)
            replacement.rename(ancestor)
            swapped = True
        return data

    monkeypatch.setattr(os, "read", swapping_read)

    with pytest.raises(InventoryError, match="root component changed identity"):
        build_independence_receipt((scan_root,), manifest=_manifest())

    assert swapped is True


def test_roster_loader_is_strictly_read_only_and_rejects_pending_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a")
    provider = InMemoryAuthorityPinProvider()
    pins = AuthorityPinStore._for_test(tmp_path, provider=provider)
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=key,
        root=tmp_path,
        pin_store=pins,
    )
    before_files = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    before_pin = pins._snapshot_for_test()
    loaded = load_verification_roster_readonly(tmp_path, pin_store=pins)
    after_files = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    assert loaded == roster
    assert after_files == before_files
    assert pins._snapshot_for_test() == before_pin

    installation_id = next(iter(provider._installations.values()))

    class ReadOnlyProductionProvider:
        def _resolve_installation(self, _location_binding: str) -> str:
            return installation_id

        def _read_account(self, account: str) -> bytes | None:
            if account.startswith("binding:"):
                return installation_id.encode("ascii")
            if account == f"pin:{installation_id}":
                return provider._values[installation_id]
            return None

        def _read_pin(self, requested_installation_id: str) -> bytes | None:
            assert requested_installation_id == installation_id
            return provider._values[installation_id]

        def _write_account(self, _account: str, _value: bytes) -> None:
            raise AssertionError("read-only roster loader attempted a Keychain write")

        def _write_pin(self, _installation_id: str, _value: bytes) -> None:
            raise AssertionError("read-only roster loader attempted a pin write")

    monkeypatch.setattr(
        safety_module,
        "MacOSAuthorityPinProvider",
        ReadOnlyProductionProvider,
    )
    assert load_verification_roster_readonly(tmp_path) == roster
    assert {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == before_files

    second_key = keys.generate(
        device_id="device-b",
        roles=("origin",),
        enrollment_sequence=2,
    )

    def leave_pending(_root: Path, _record: bytes) -> None:
        raise RuntimeError("simulated crash after roster staging")

    monkeypatch.setattr(pins, "_finish_roster", leave_pending)
    with pytest.raises(RuntimeError, match="simulated crash"):
        roster.with_keys(
            version=2,
            peers=("device-a", "device-b"),
            keys=(key, second_key),
            signer=OperationalSigner(keys, roster_version=roster.version),
            root=tmp_path,
            pin_store=pins,
        )
    pending_files = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    pending_pin = pins._snapshot_for_test()

    with pytest.raises(CutoverSafetyError, match="pending recovery"):
        load_verification_roster_readonly(tmp_path, pin_store=pins)

    assert {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == pending_files
    assert pins._snapshot_for_test() == pending_pin

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "source.json").write_bytes(canonical_json_bytes({"source_commit": "a" * 40}))
    monkeypatch.setattr(
        absorption_cli,
        "discover_synapse_operations",
        lambda _snapshot: (
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
    monkeypatch.setattr(
        absorption_cli,
        "_verified_receipt_ids",
        lambda _args, _roster, _source_commit: ([], []),
    )
    monkeypatch.setattr(
        AuthorityPinStore,
        "for_root",
        classmethod(lambda _cls, _root: pins),
    )
    cli_files = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    cli_pin = pins._snapshot_for_test()

    with pytest.raises(SystemExit, match="read-only"):
        absorption_cli._synapse_manifest(
            SimpleNamespace(
                attempt_root=tmp_path / "state" / "memo" / "cutover" / "attempt-123",
                attempt_id="attempt-123",
                snapshot=snapshot,
                roster_root=tmp_path,
                usage_proof=[],
                exclusion=[],
                apply=False,
            )
        )

    assert {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == cli_files
    assert pins._snapshot_for_test() == cli_pin


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
            authority_root=tmp_path,
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
            authority_root=tmp_path,
            cleanup_paths=(tmp_path / "synapse-state",),
        )

    observed = dict(expected)
    observed["bounded_data_receipt"] = "5" * 64
    with pytest.raises(CutoverSafetyError, match="digest mismatch"):
        assert_retirement_cleanup_authority(
            control,
            expected_digests=expected,
            observed_digests=observed,
            authority_root=tmp_path,
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
    authority_root = tmp_path / "cleanup-authority"
    authority_root.mkdir()

    with pytest.raises(CutoverSafetyError, match="cleanup path"):
        assert_retirement_cleanup_authority(
            control,
            expected_digests=expected,
            observed_digests=expected,
            authority_root=authority_root,
            cleanup_paths=(unsafe_path,),
        )


def test_cleanup_authority_rejects_missing_special_and_repository_paths(
    tmp_path: Path,
) -> None:
    expected = _expected_digests()
    control = _verified_control(
        manifest_sha256=expected["retirement_manifest"],
        independence_sha256=expected["independence_receipt"],
    )
    authority_root = tmp_path / "cleanup-authority"
    authority_root.mkdir()

    with pytest.raises(CutoverSafetyError, match="does not exist"):
        assert_retirement_cleanup_authority(
            control,
            expected_digests=expected,
            observed_digests=expected,
            authority_root=authority_root,
            cleanup_paths=(authority_root / "missing",),
        )

    special = authority_root / "pipe"
    os.mkfifo(special)
    with pytest.raises(CutoverSafetyError, match="regular file or directory"):
        assert_retirement_cleanup_authority(
            control,
            expected_digests=expected,
            observed_digests=expected,
            authority_root=authority_root,
            cleanup_paths=(special,),
        )

    repository = tmp_path / "repo"
    (repository / ".git").mkdir(parents=True)
    repository_authority = repository / "cleanup-authority"
    repository_authority.mkdir()
    target = repository_authority / "state"
    target.mkdir()
    with pytest.raises(CutoverSafetyError, match="repository"):
        assert_retirement_cleanup_authority(
            control,
            expected_digests=expected,
            observed_digests=expected,
            authority_root=repository_authority,
            cleanup_paths=(target,),
        )


def test_cleanup_authority_rejects_broad_or_outside_authority_boundary(
    tmp_path: Path,
) -> None:
    expected = _expected_digests()
    control = _verified_control(
        manifest_sha256=expected["retirement_manifest"],
        independence_sha256=expected["independence_receipt"],
    )
    target = tmp_path / "target"
    target.mkdir()

    with pytest.raises(CutoverSafetyError, match="authority root is too broad"):
        assert_retirement_cleanup_authority(
            control,
            expected_digests=expected,
            observed_digests=expected,
            authority_root=Path("/Users"),
            cleanup_paths=(target,),
        )

    authority_root = tmp_path / "cleanup-authority"
    authority_root.mkdir()
    with pytest.raises(CutoverSafetyError, match="outside cleanup authority"):
        assert_retirement_cleanup_authority(
            control,
            expected_digests=expected,
            observed_digests=expected,
            authority_root=authority_root,
            cleanup_paths=(target,),
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
            authority_root=tmp_path,
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
        "control_record": hashlib.sha256(canonical_json_bytes(objects["control"])).hexdigest(),
        "retirement_manifest": hashlib.sha256(manifest.signed_bytes()).hexdigest(),
        "consumer_replacement_receipt": hashlib.sha256(plan_bytes).hexdigest(),
        "bounded_data_receipt": hashlib.sha256(canonical_json_bytes(objects["data"])).hexdigest(),
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
        "cleanup_authority_root": tmp_path,
        "cleanup_path": [cleanup_target],
    }
    arguments.update({f"{key}_sha256": value for key, value in expected.items()})

    with pytest.raises(SystemExit, match="runtime gate evidence"):
        absorption_cli._retirement_cleanup(SimpleNamespace(**arguments))

    assert marker.read_text(encoding="utf-8") == "preserved"
