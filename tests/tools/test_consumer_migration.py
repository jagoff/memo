"""Authority and filesystem regressions for staged consumer replacement."""

from __future__ import annotations

import hashlib
import os
import plistlib
from dataclasses import replace
from pathlib import Path

import pytest

from memo.operational_key_store import (
    AuthorityPinStore,
    DeviceKeyStore,
    InMemoryAuthorityPinProvider,
)
from memo.operational_roster import VerificationRoster
from memo.operational_signing import OperationalSigner
from tools.memflow_absorption.consumer_migration import (
    ConsumerMigrationError,
    build_consumer_replacement_plan,
    render_memo_launch_agents,
    verify_no_synapse_runtime_reference,
)
from tools.memflow_absorption.inventory import CONSUMER_INVENTORY_DOMAIN
from tools.memflow_absorption.manifest import CAPABILITY_MANIFEST_DOMAIN
from tools.memflow_absorption.schemas import (
    CapabilityManifest,
    CapabilityRow,
    ConsumerInventory,
    ConsumerInventoryRow,
    OperationMappingRow,
    OperationRoute,
)

_LABELS = (
    "com.synapse.whatsapp-ingest",
    "com.synapse.watcher",
    "com.synapse.memo-recall-daemon",
    "com.synapse.memo-nightly",
    "com.synapse.morning-digest",
    "com.synapse.dream-synthesis",
    "com.synapse.vault-ingest",
    "com.synapse.dashboard",
    "com.synapse.dashboard-relay",
)
_OPERATIONS = (
    "synapse.chat.ask",
    "synapse.cli.ops",
    "synapse.morning_digest.run",
    "synapse.watcher.event",
    "synapse.whatsapp_live.message",
)
_CLIS_BY_OPERATION = {
    "synapse.chat.ask": ("client-close-reconnect",),
    "synapse.cli.ops": (
        "memo dream run",
        "memo recall-daemon _serve",
        "memo reindex",
    ),
    "synapse.morning_digest.run": ("memo digest",),
    "synapse.watcher.event": ("memo watch",),
    "synapse.whatsapp_live.message": ("memo import whatsapp",),
}


@pytest.fixture
def authority(tmp_path: Path) -> tuple[DeviceKeyStore, VerificationRoster]:
    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a")
    root = tmp_path / "authority"
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=key,
        root=root,
        pin_store=AuthorityPinStore._for_test(
            root,
            provider=InMemoryAuthorityPinProvider(),
        ),
    )
    return keys, roster


def _mapping(operation: str) -> OperationMappingRow:
    routes = tuple(
        OperationRoute(
            route_id=f"{operation}-{index}",
            predicate={"mode": {"eq": "native"}},
            memo_methods=(f"memo.route.{index}",),
            memo_mcp=(f"memo_route_{index}",),
            memo_cli=(cli,),
            parameter_mapping={},
            defaults={"mode": "native"},
            result_mapping={"status": "status"},
            error_mapping={"error": "error"},
            transform_id="identity",
            fixture_sha256=("a" * 64,),
            atomic_group=None,
            fixture_paths=("eval/native.json",),
        )
        for index, cli in enumerate(_CLIS_BY_OPERATION[operation])
    )
    return OperationMappingRow(
        source_operation=operation,
        source_commit="a" * 40,
        source_tests=("tests/native.py",),
        evidence_ids=(f"usage:{operation}",),
        capability=operation,
        disposition="memo_native",
        routes=routes,
        parity_tests=("tests/tools/test_consumer_migration.py",),
        deletion_proof=(),
    )


def _resign_manifest(
    manifest: CapabilityManifest,
    keys: DeviceKeyStore,
    roster: VerificationRoster,
) -> CapabilityManifest:
    unsigned = replace(
        manifest,
        operation_map_sha256=hashlib.sha256(manifest.operation_map_bytes()).hexdigest(),
        slo_baseline_sha256=hashlib.sha256(manifest.slo_baseline_bytes()).hexdigest(),
        signer_device_id=roster.local_device_id,
        signer_key_id=roster.local_key_id,
        roster_version=roster.version,
        signature="",
    )
    envelope = OperationalSigner(keys, roster_version=roster.version).sign(
        domain=CAPABILITY_MANIFEST_DOMAIN,
        payload=unsigned.signed_bytes(),
        key_id=roster.local_key_id,
    )
    return replace(unsigned, signature=envelope.signature)


@pytest.fixture
def manifest(
    authority: tuple[DeviceKeyStore, VerificationRoster],
) -> CapabilityManifest:
    keys, roster = authority
    mappings = tuple(_mapping(operation) for operation in _OPERATIONS)
    capabilities = tuple(
        CapabilityRow(
            name=mapping.capability,
            sources=(mapping.source_operation,),
            consumers=("device-a",),
            window_started_at="2026-04-30T00:00:00Z",
            window_ended_at="2026-07-30T00:00:00Z",
            observed_calls=1,
            observed_daemon_events=1,
            machines=("device-a",),
            evidence_ids=mapping.evidence_ids,
            exclusion_counts={},
            evidence_complete=True,
            source_operations=(mapping.source_operation,),
            operation_mappings=(mapping,),
            slo_baseline_ids=(),
            dependencies=(),
            disposition="memo_native",
            memo_target=",".join(
                sorted(
                    method
                    for route in mapping.routes
                    for method in route.memo_methods
                )
            ),
            parity_tests=mapping.parity_tests,
            deletion_proof=(),
        )
        for mapping in mappings
    )
    unsigned = CapabilityManifest(
        schema="memo.cutover_capability_manifest.v1",
        frozen_at="2026-07-30T00:00:00Z",
        window_started_at="2026-04-30T00:00:00Z",
        window_ended_at="2026-07-30T00:00:00Z",
        machine_ids=("device-a",),
        source_receipt_sha256={},
        capabilities=capabilities,
        operation_mappings=mappings,
        slo_baselines=(),
        operation_map_sha256="",
        slo_baseline_sha256="",
        blockers=(),
        frozen=True,
        signer_device_id="",
        signer_key_id="",
        roster_version=0,
        signature="",
    )
    return _resign_manifest(unsigned, keys, roster)


def _launchd_row(label: str) -> ConsumerInventoryRow:
    common: dict[str, object] = {
        "kind": "launchd",
        "location": f"{label}:/operator/archive/{label}.plist",
        "references": ("synapse",),
        "label": label,
        "active": True,
        "run_at_load": True,
        "environment": (
            ("MEMO_DATA_DIR", "/operator/memo-data"),
            ("PATH", "/usr/bin:/bin"),
            ("UNRELATED_SECRET", "discard-me"),
        ),
        "environment_keys": ("MEMO_DATA_DIR", "PATH", "UNRELATED_SECRET"),
    }
    if label == "com.synapse.whatsapp-ingest":
        common.update(
            program_arguments=(
                "/operator/runtime/memo",
                "import",
                "whatsapp",
                "--include-chat",
                "chat-a",
                "--exclude-chat",
                "chat-b",
                "--retention-days",
                "90",
                "--json",
            ),
            watch_paths=("/operator/whatsapp/messages.db",),
            throttle_interval_seconds=300,
        )
    elif label == "com.synapse.watcher":
        common.update(
            program_arguments=("/operator/runtime/memo", "watch"),
            keep_alive={"SuccessfulExit": False},
        )
    elif label == "com.synapse.memo-recall-daemon":
        common.update(
            program_arguments=("/operator/runtime/memo", "recall-daemon", "_serve"),
            keep_alive=True,
        )
    elif label == "com.synapse.morning-digest":
        common.update(
            program_arguments=("/operator/runtime/memo", "digest"),
            run_at_load=False,
            start_calendar_interval=((("Hour", 3), ("Minute", 0)),),
        )
    elif label in {"com.synapse.memo-nightly", "com.synapse.dream-synthesis"}:
        common.update(
            program_arguments=("/operator/runtime/memo", "dream", "run"),
            run_at_load=False,
            start_calendar_interval=(
                (("Hour", 3), ("Minute", 0)),
                (("Hour", 15), ("Minute", 30)),
            ),
        )
    elif label == "com.synapse.vault-ingest":
        common.update(
            program_arguments=("/operator/runtime/memo", "reindex"),
            watch_paths=("/operator/vault",),
            throttle_interval_seconds=120,
        )
    else:
        common.update(program_arguments=("/operator/runtime/client",))
    return ConsumerInventoryRow(**common)  # type: ignore[arg-type]


def _resign_inventory(
    inventory: ConsumerInventory,
    keys: DeviceKeyStore,
    roster: VerificationRoster,
) -> ConsumerInventory:
    unsigned = replace(
        inventory,
        signer_device_id=roster.local_device_id,
        signer_key_id=roster.local_key_id,
        roster_version=roster.version,
        signature="",
    )
    envelope = OperationalSigner(keys, roster_version=roster.version).sign(
        domain=CONSUMER_INVENTORY_DOMAIN,
        payload=unsigned.signed_bytes(),
        key_id=roster.local_key_id,
    )
    return replace(unsigned, signature=envelope.signature)


@pytest.fixture
def inventory(
    authority: tuple[DeviceKeyStore, VerificationRoster],
) -> ConsumerInventory:
    keys, roster = authority
    unsigned = ConsumerInventory(
        schema="memo.cutover_consumer_inventory.v1",
        rows=(
            *(_launchd_row(label) for label in _LABELS),
            ConsumerInventoryRow(
                kind="launchd",
                location="com.synapse.unloaded:/operator/archive/unloaded.plist",
                references=("synapse",),
                label="com.synapse.unloaded",
                active=False,
            ),
        ),
        blockers=(),
        source_scan_sha256="a" * 64,
        surface_observations={
            surface: (f"signed-scan:{surface}",)
            for surface in (
                "process",
                "port",
                "launchagent",
                "mcp_gateway_route",
                "shell_config_path",
                "state_root",
            )
        },
    )
    return _resign_inventory(unsigned, keys, roster)


@pytest.fixture
def memo_bin(tmp_path: Path) -> Path:
    path = tmp_path / "isolated-runtime" / "memo"
    path.parent.mkdir()
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _plan(inventory, manifest, authority, memo_bin):
    return build_consumer_replacement_plan(
        inventory,
        manifest,
        roster=authority[1],
        memo_bin=memo_bin,
    )


def test_plan_maps_exact_labels_to_absolute_memo_commands(
    inventory, manifest, authority, memo_bin
):
    plan = _plan(inventory, manifest, authority, memo_bin)

    whatsapp = plan.by_old_label("com.synapse.whatsapp-ingest")
    assert whatsapp.command[:3] == (str(memo_bin.resolve()), "import", "whatsapp")
    assert whatsapp.command.count("--include-chat") == 1
    assert "--all-chats" not in whatsapp.command
    assert plan.by_old_label("com.synapse.watcher").command == (
        str(memo_bin.resolve()),
        "watch",
    )
    assert plan.by_old_label("com.synapse.dashboard").command == (
        "client-close-reconnect",
    )
    assert all(row.old_label != "com.synapse.unloaded" for row in plan.rows)


def test_unknown_or_substring_label_blocks(inventory, manifest, authority, memo_bin):
    keys, roster = authority
    row = replace(_launchd_row("com.synapse.watcher"), label="com.synapse.watcher-extra")
    changed = _resign_inventory(replace(inventory, rows=(row,)), keys, roster)

    with pytest.raises(ConsumerMigrationError, match="no Memo-owned replacement"):
        _plan(changed, manifest, authority, memo_bin)


def test_whatsapp_without_authoritative_scope_blocks(
    inventory, manifest, authority, memo_bin
):
    keys, roster = authority
    row = replace(
        _launchd_row("com.synapse.whatsapp-ingest"),
        program_arguments=("/old/memo", "import", "whatsapp", "--json"),
    )
    changed = _resign_inventory(replace(inventory, rows=(row,)), keys, roster)

    with pytest.raises(ConsumerMigrationError, match="authoritative chat scope"):
        _plan(changed, manifest, authority, memo_bin)


@pytest.mark.parametrize(
    ("option", "values", "message"),
    (
        ("--retention-days", ("90", "30"), "repeated"),
        ("--since", ("2026-01-01", "2026-02-01"), "repeated"),
        ("--notes-dir", ("/one", "/two"), "repeated"),
        ("--db", ("/one.db", "/two.db"), "repeated"),
        ("--retention-days", ("zero",), "retention"),
        ("--retention-days", ("0",), "retention"),
        ("--since", ("2026-02-30",), "since date"),
        ("--notes-dir", ("relative",), "path must be absolute"),
        ("--db", ("relative.db",), "path must be absolute"),
    ),
)
def test_whatsapp_single_value_options_are_typed_and_unique(
    inventory,
    manifest,
    authority,
    memo_bin,
    option,
    values,
    message,
):
    keys, roster = authority
    arguments = ["/old/memo", "import", "whatsapp", "--all-chats"]
    for value in values:
        arguments.extend((option, value))
    row = replace(
        _launchd_row("com.synapse.whatsapp-ingest"),
        program_arguments=tuple(arguments),
    )
    changed = _resign_inventory(replace(inventory, rows=(row,)), keys, roster)

    with pytest.raises(ConsumerMigrationError, match=message):
        _plan(changed, manifest, authority, memo_bin)


def test_unsigned_or_tampered_authority_blocks(inventory, manifest, authority, memo_bin):
    with pytest.raises(ConsumerMigrationError, match="authority is invalid"):
        _plan(replace(inventory, signature=""), manifest, authority, memo_bin)
    with pytest.raises(ConsumerMigrationError, match="authority is invalid"):
        _plan(
            inventory,
            replace(manifest, operation_map_sha256="0" * 64),
            authority,
            memo_bin,
        )


def test_missing_operation_capability_mapping_blocks(
    inventory, manifest, authority, memo_bin
):
    keys, roster = authority
    retained_mappings = tuple(
        row
        for row in manifest.operation_mappings
        if row.source_operation != "synapse.watcher.event"
    )
    retained_capabilities = tuple(
        row
        for row in manifest.capabilities
        if "synapse.watcher.event" not in row.source_operations
    )
    changed = _resign_manifest(
        replace(
            manifest,
            operation_mappings=retained_mappings,
            capabilities=retained_capabilities,
        ),
        keys,
        roster,
    )

    with pytest.raises(ConsumerMigrationError, match="operation mapping"):
        _plan(inventory, changed, authority, memo_bin)


def test_cli_route_and_capability_target_must_match_replacement_exactly(
    inventory, manifest, authority, memo_bin
):
    keys, roster = authority
    mapping = next(
        row
        for row in manifest.operation_mappings
        if row.source_operation == "synapse.watcher.event"
    )
    capability = next(
        row
        for row in manifest.capabilities
        if row.name == mapping.capability
    )
    mismatched_route = replace(mapping.routes[0], memo_cli=("memo native",))
    mismatched_mapping = replace(mapping, routes=(mismatched_route,))
    changed = _resign_manifest(
        replace(
            manifest,
            operation_mappings=tuple(
                mismatched_mapping if row is mapping else row
                for row in manifest.operation_mappings
            ),
            capabilities=tuple(
                replace(capability, operation_mappings=(mismatched_mapping,))
                if row is capability
                else row
                for row in manifest.capabilities
            ),
        ),
        keys,
        roster,
    )
    with pytest.raises(ConsumerMigrationError, match="exact admitted CLI route"):
        _plan(inventory, changed, authority, memo_bin)

    wrong_target_capability = replace(capability, memo_target="memo.unrelated")
    changed = _resign_manifest(
        replace(
            manifest,
            capabilities=tuple(
                wrong_target_capability if row is capability else row
                for row in manifest.capabilities
            ),
        ),
        keys,
        roster,
    )
    with pytest.raises(ConsumerMigrationError, match="target is not exactly admitted"):
        _plan(inventory, changed, authority, memo_bin)


def test_uncorrelated_live_process_blocks(inventory, manifest, authority, memo_bin):
    keys, roster = authority
    process = ConsumerInventoryRow(
        kind="process",
        location="pid:42:/operator/synapse",
        references=("synapse",),
        label="/operator/synapse runtime",
        program_arguments=("/operator/runtime/memo", "watch"),
    )
    changed = _resign_inventory(
        replace(inventory, rows=(*inventory.rows, process)),
        keys,
        roster,
    )

    with pytest.raises(ConsumerMigrationError, match="manual correlation"):
        _plan(changed, manifest, authority, memo_bin)

    inexact = replace(
        process,
        correlated_launchd_label="com.synapse.watcher",
        program_arguments=("/operator/runtime/other", "watch"),
    )
    changed = _resign_inventory(
        replace(inventory, rows=(*inventory.rows, inexact)),
        keys,
        roster,
    )
    with pytest.raises(ConsumerMigrationError, match="correlation is not exact"):
        _plan(changed, manifest, authority, memo_bin)

    correlated = replace(process, correlated_launchd_label="com.synapse.watcher")
    changed = _resign_inventory(
        replace(inventory, rows=(*inventory.rows, correlated)),
        keys,
        roster,
    )
    _plan(changed, manifest, authority, memo_bin)


def test_relative_or_project_runtime_binary_blocks(inventory, manifest, authority, tmp_path):
    with pytest.raises(ConsumerMigrationError, match="absolute"):
        _plan(inventory, manifest, authority, Path("memo"))

    project_bin = tmp_path / ".venv" / "bin" / "memo"
    project_bin.parent.mkdir(parents=True)
    project_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    project_bin.chmod(0o755)
    with pytest.raises(ConsumerMigrationError, match="stable isolated"):
        _plan(inventory, manifest, authority, project_bin)

    wrongly_named = tmp_path / "isolated-runtime" / "memo-wrapper"
    wrongly_named.parent.mkdir(exist_ok=True)
    wrongly_named.write_text("#!/bin/sh\n", encoding="utf-8")
    wrongly_named.chmod(0o755)
    with pytest.raises(ConsumerMigrationError, match="must be named memo"):
        _plan(inventory, manifest, authority, wrongly_named)


def test_relative_environment_path_blocks(inventory, manifest, authority, memo_bin):
    keys, roster = authority
    row = replace(
        _launchd_row("com.synapse.whatsapp-ingest"),
        environment=(("MEMO_DATA_DIR", "/operator/memo"), ("PATH", "relative:/bin")),
    )
    changed = _resign_inventory(replace(inventory, rows=(row,)), keys, roster)

    with pytest.raises(ConsumerMigrationError, match="authoritative PATH"):
        _plan(changed, manifest, authority, memo_bin)


@pytest.mark.parametrize("missing_key", ("MEMO_DATA_DIR", "PATH"))
def test_retained_environment_key_without_exact_value_blocks(
    inventory, manifest, authority, memo_bin, missing_key
):
    keys, roster = authority
    row = _launchd_row("com.synapse.whatsapp-ingest")
    changed_environment = tuple(
        item for item in row.environment if item[0] != missing_key
    )
    row = replace(row, environment=changed_environment)
    changed = _resign_inventory(replace(inventory, rows=(row,)), keys, roster)

    with pytest.raises(ConsumerMigrationError, match="exact authoritative environment values"):
        _plan(changed, manifest, authority, memo_bin)


def test_renderer_preserves_authoritative_schedules(
    inventory, manifest, authority, memo_bin, tmp_path
):
    plan = _plan(inventory, manifest, authority, memo_bin)
    paths = render_memo_launch_agents(plan, tmp_path / "operator-staging")
    rendered = {
        plistlib.loads(path.read_bytes())["Label"]: plistlib.loads(path.read_bytes())
        for path in paths
    }

    assert rendered["com.memo.watch"]["KeepAlive"] == {"SuccessfulExit": False}
    assert rendered["com.memo.recall-daemon"]["KeepAlive"] is True
    assert rendered["com.memo.dream-nightly"]["StartCalendarInterval"] == [
        {"Hour": 3, "Minute": 0},
        {"Hour": 15, "Minute": 30},
    ]
    assert rendered["com.memo.digest"]["StartCalendarInterval"] == {
        "Hour": 3,
        "Minute": 0,
    }
    assert rendered["com.memo.digest"]["KeepAlive"] is False
    assert rendered["com.memo.import-whatsapp"]["WatchPaths"] == [
        "/operator/whatsapp/messages.db"
    ]
    assert rendered["com.memo.import-whatsapp"]["ThrottleInterval"] == 300
    for row in rendered.values():
        environment = row["EnvironmentVariables"]
        assert environment["MEMO_DATA_DIR"] == "/operator/memo-data"
        assert environment["MEMO_NONINTERACTIVE"] == "1"
        assert "UNRELATED_SECRET" not in environment
        path_entries = environment["PATH"].split(":")
        assert path_entries[0] == str(memo_bin.resolve().parent)
        assert all(Path(entry).is_absolute() for entry in path_entries)
    assert all(Path(row["ProgramArguments"][0]).is_absolute() for row in rendered.values())


def test_invalid_keepalive_and_calendar_shapes_block(
    inventory, manifest, authority, memo_bin
):
    keys, roster = authority
    invalid_keep_alive = replace(
        _launchd_row("com.synapse.watcher"),
        keep_alive={"SuccessfulExit": "false"},
    )
    changed = _resign_inventory(
        replace(inventory, rows=(invalid_keep_alive,)),
        keys,
        roster,
    )
    with pytest.raises(ConsumerMigrationError, match="invalid KeepAlive"):
        _plan(changed, manifest, authority, memo_bin)

    invalid_calendar = replace(
        _launchd_row("com.synapse.memo-nightly"),
        start_calendar_interval=((("Hour", 24),),),
    )
    changed = _resign_inventory(
        replace(inventory, rows=(invalid_calendar,)),
        keys,
        roster,
    )
    with pytest.raises(ConsumerMigrationError, match="invalid calendar"):
        _plan(changed, manifest, authority, memo_bin)


@pytest.mark.parametrize(
    ("keep_alive", "valid"),
    (
        ({"PathState": {"/operator/ready": True}}, True),
        ({"PathState": {"operator/ready": True}}, False),
        ({"OtherJobEnabled": {"com.memo.indexer": True}}, True),
        ({"OtherJobEnabled": {"/Library/LaunchAgents/com.memo.indexer": True}}, False),
        ({"OtherJobEnabled": {"": True}}, False),
        ({"OtherJobEnabled": {"   ": True}}, False),
    ),
)
def test_keepalive_state_keys_validate_paths_and_job_labels(
    inventory, manifest, authority, memo_bin, keep_alive, valid
):
    keys, roster = authority
    row = replace(_launchd_row("com.synapse.watcher"), keep_alive=keep_alive)
    changed = _resign_inventory(replace(inventory, rows=(row,)), keys, roster)

    if valid:
        planned = _plan(changed, manifest, authority, memo_bin)
        assert planned.by_old_label("com.synapse.watcher").keep_alive == keep_alive
    else:
        with pytest.raises(ConsumerMigrationError, match="invalid KeepAlive"):
            _plan(changed, manifest, authority, memo_bin)


def test_periodic_job_without_schedule_blocks(inventory, manifest, authority, memo_bin):
    keys, roster = authority
    row = replace(
        _launchd_row("com.synapse.morning-digest"),
        start_calendar_interval=(),
        run_at_load=True,
    )
    changed = _resign_inventory(replace(inventory, rows=(row,)), keys, roster)

    with pytest.raises(ConsumerMigrationError, match="lacks authoritative launchd schedule"):
        _plan(changed, manifest, authority, memo_bin)


def test_renderer_rejects_production_library_or_ancestor(
    inventory, manifest, authority, memo_bin
):
    plan = _plan(inventory, manifest, authority, memo_bin)

    with pytest.raises(ConsumerMigrationError, match="production Library"):
        render_memo_launch_agents(plan, Path.home() / "Library")
    with pytest.raises(ConsumerMigrationError, match="production Library"):
        render_memo_launch_agents(plan, Path.home())
    for protected in (
        Path("/Library"),
        Path("/library"),
        Path("/System/Library"),
        Path("/Users/another-user/Library"),
        Path("/users/another-user/library"),
        Path("/Users/another-user"),
    ):
        with pytest.raises(ConsumerMigrationError, match="production Library"):
            render_memo_launch_agents(plan, protected)


def test_renderer_rejects_symlink_and_hardlink_outputs_without_mutation(
    inventory, manifest, authority, memo_bin, tmp_path
):
    plan = _plan(inventory, manifest, authority, memo_bin)
    real = tmp_path / "real-staging"
    linked = tmp_path / "linked-staging"
    real.mkdir()
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ConsumerMigrationError, match="unsafe"):
        render_memo_launch_agents(plan, linked)

    staging = tmp_path / "hardlink-staging"
    launch_agents = staging / "LaunchAgents"
    launch_agents.mkdir(parents=True)
    outside = tmp_path / "outside.plist"
    outside.write_text("operator-owned", encoding="utf-8")
    os.link(outside, launch_agents / "com.memo.watch.plist")
    with pytest.raises(ConsumerMigrationError, match="unsafe staged LaunchAgent"):
        render_memo_launch_agents(plan, staging)
    assert outside.read_text(encoding="utf-8") == "operator-owned"


def test_renderer_and_verifier_reject_stale_or_extra_outputs(
    inventory, manifest, authority, memo_bin, tmp_path
):
    plan = _plan(inventory, manifest, authority, memo_bin)
    stale_root = tmp_path / "stale-root"
    stale_root.mkdir()
    (stale_root / "unrelated.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(ConsumerMigrationError, match="stale output"):
        render_memo_launch_agents(plan, stale_root)

    staging = tmp_path / "operator-staging"
    render_memo_launch_agents(plan, staging)
    extra = staging / "LaunchAgents" / "extra.plist"
    extra.write_text("safe text", encoding="utf-8")
    with pytest.raises(ConsumerMigrationError, match="output set is not exact"):
        verify_no_synapse_runtime_reference(staging, plan)


def test_verifier_rejects_mutated_plist(inventory, manifest, authority, memo_bin, tmp_path):
    plan = _plan(inventory, manifest, authority, memo_bin)
    staging = tmp_path / "operator-staging"
    paths = render_memo_launch_agents(plan, staging)
    paths[0].write_text("legacy memflow runtime", encoding="utf-8")

    with pytest.raises(ConsumerMigrationError, match="retired runtime reference"):
        verify_no_synapse_runtime_reference(staging, plan)


def test_renderer_rejects_tampered_plan_digest(
    inventory, manifest, authority, memo_bin, tmp_path
):
    plan = _plan(inventory, manifest, authority, memo_bin)
    tampered = replace(plan, digest="0" * 64)

    with pytest.raises(ConsumerMigrationError, match="plan digest is invalid"):
        render_memo_launch_agents(tampered, tmp_path / "operator-staging")
