from __future__ import annotations

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
from tools.memflow_absorption.inventory import (
    InventoryError,
    build_consumer_inventory,
    build_synapse_retirement_manifest,
    verify_consumer_inventory,
    verify_synapse_retirement_manifest,
)
from tools.memflow_absorption.schemas import (
    LaunchdRecord,
    LaunchdSnapshot,
    ProcessRecord,
    ProcessSnapshot,
)


def _authority(tmp_path: Path) -> tuple[DeviceKeyStore, VerificationRoster]:
    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=key,
        root=tmp_path,
        pin_store=AuthorityPinStore._for_test(
            tmp_path,
            provider=InMemoryAuthorityPinProvider(),
        ),
    )
    return keys, roster


def _write_canonical_synapse_catalog(snapshot: Path) -> None:
    package = snapshot / "src" / "synapse"
    (package / "cli").mkdir(parents=True, exist_ok=True)
    (snapshot / "source.json").write_bytes(canonical_json_bytes({"source_commit": "a" * 40}))
    (package / "mcp_catalog.py").write_text(
        "CANONICAL_MCP_TOOLS = [McpToolManifest(tool_id='synapse.federate.query', mcp_name='synapse_federate_query')]\n",
        encoding="utf-8",
    )
    (package / "cli" / "parser.py").write_text(
        "def build_parser():\n    sub.add_parser('query')\n",
        encoding="utf-8",
    )
    for name, functions in {
        "runtime.py": "def runtime_loop(): pass\n",
        "watcher.py": "def _emit(): pass\n",
        "morning_digest.py": "def run_morning_digest(): pass\n",
        "whatsapp_live.py": "def last_messages(): pass\ndef last_messages_multi(): pass\n",
        "vault_archive.py": "def move_to_archive(): pass\n",
    }.items():
        (package / name).write_text(functions, encoding="utf-8")


def test_inventory_combines_source_process_and_launchd_without_following_symlinks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "consumers"
    root.mkdir()
    (root / "client.json").write_text(
        '{"command":"/Users/example/repos/memflow/.venv/bin/python","env":{"SYNAPSE_MEMFLOW_BIN":"x"}}',
        encoding="utf-8",
    )
    outside = tmp_path / "outside.txt"
    outside.write_text("memflow secret", encoding="utf-8")
    (root / "linked.txt").symlink_to(outside)
    processes = ProcessSnapshot(
        captured_at="2026-07-30T00:00:00Z",
        records=(
            ProcessRecord(
                pid=42,
                executable="/Users/example/repos/memflow/.venv/bin/python",
                argv=("memflow", "mcp-streamable"),
            ),
        ),
    )
    launchd = LaunchdSnapshot(
        captured_at="2026-07-30T00:00:00Z",
        records=(
            LaunchdRecord(
                label="com.synapse.dashboard",
                plist_path="/Users/example/Library/LaunchAgents/com.synapse.dashboard.plist",
                program_arguments=(
                    "/Users/example/repos/memflow/.venv/bin/python",
                    "-m",
                    "synapse.cli",
                ),
                environment_keys=("SYNAPSE_MEMFLOW_BIN", "MEMO_DATA_DIR", "PATH"),
                loaded=True,
                run_at_load=True,
                keep_alive={"SuccessfulExit": False},
                start_calendar_interval=(
                    (("Hour", 3), ("Minute", 0)),
                    (("Hour", 15), ("Minute", 30)),
                ),
                environment=(
                    ("MEMO_DATA_DIR", "/operator/memo"),
                    ("PATH", "/usr/bin:/bin"),
                ),
            ),
        ),
    )

    inventory = build_consumer_inventory((root,), processes, launchd)

    assert {row.kind for row in inventory.rows} == {"source", "process", "launchd"}
    assert any(row.location.endswith("client.json") for row in inventory.rows)
    assert all("outside.txt" not in row.location for row in inventory.rows)
    launchd_row = next(row for row in inventory.rows if row.kind == "launchd")
    assert launchd_row.label == "com.synapse.dashboard"
    assert launchd_row.active is True
    assert launchd_row.run_at_load is True
    assert launchd_row.keep_alive == {"SuccessfulExit": False}
    assert launchd_row.start_calendar_interval == (
        (("Hour", 3), ("Minute", 0)),
        (("Hour", 15), ("Minute", 30)),
    )
    assert launchd_row.environment == (
        ("MEMO_DATA_DIR", "/operator/memo"),
        ("PATH", "/usr/bin:/bin"),
    )
    assert launchd_row.environment_keys == (
        "SYNAPSE_MEMFLOW_BIN",
        "MEMO_DATA_DIR",
        "PATH",
    )
    assert inventory.blockers == ("symlink-skipped:linked.txt",)
    assert len(inventory.source_scan_sha256) == 64


def test_inventory_correlates_live_process_to_exact_loaded_launchd_command() -> None:
    command = ("/operator/synapse-runtime", "watch")
    inventory = build_consumer_inventory(
        (),
        ProcessSnapshot(
            captured_at="2026-07-30T00:00:00Z",
            records=(
                ProcessRecord(
                    pid=42,
                    executable=command[0],
                    argv=command[1:],
                ),
            ),
        ),
        LaunchdSnapshot(
            captured_at="2026-07-30T00:00:00Z",
            records=(
                LaunchdRecord(
                    label="com.synapse.watcher",
                    plist_path="/operator/archive/com.synapse.watcher.plist",
                    program_arguments=command,
                    environment_keys=(),
                    loaded=True,
                    run_at_load=True,
                    keep_alive=True,
                ),
            ),
        ),
    )

    process = next(row for row in inventory.rows if row.kind == "process")
    assert process.correlated_launchd_label == "com.synapse.watcher"
    assert process.program_arguments == command


def test_inventory_rejects_symlink_root(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(InventoryError, match="symlink"):
        build_consumer_inventory(
            (linked,),
            ProcessSnapshot(captured_at="", records=()),
            LaunchdSnapshot(captured_at="", records=()),
        )


def test_inventory_rejects_root_below_symlink_component(tmp_path: Path) -> None:
    real = tmp_path / "real"
    nested = real / "nested"
    nested.mkdir(parents=True)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(InventoryError, match="symlink component"):
        build_consumer_inventory(
            (linked / "nested",),
            ProcessSnapshot(captured_at="", records=()),
            LaunchdSnapshot(captured_at="", records=()),
        )


def test_clean_inventory_is_signed_and_tamper_evident(tmp_path: Path) -> None:
    root = tmp_path / "consumers"
    root.mkdir()
    (root / "client.json").write_text('{"backend":"memflow"}', encoding="utf-8")
    keys, roster = _authority(tmp_path / "authority")

    inventory = build_consumer_inventory(
        (root,),
        ProcessSnapshot(captured_at="", records=()),
        LaunchdSnapshot(captured_at="", records=()),
        signer=OperationalSigner(keys, roster_version=roster.version),
        signer_key_id=roster.local_key_id,
        roster=roster,
    )

    assert inventory.signature
    verify_consumer_inventory(inventory, roster=roster)
    with pytest.raises(InventoryError, match="signature"):
        verify_consumer_inventory(
            replace(inventory, source_scan_sha256="0" * 64),
            roster=roster,
        )


def test_synapse_retirement_manifest_enumerates_memflow_surface(tmp_path: Path) -> None:
    snapshot = tmp_path / "synapse"
    (snapshot / "src" / "synapse").mkdir(parents=True)
    (snapshot / "tests" / "goldens").mkdir(parents=True)
    _write_canonical_synapse_catalog(snapshot)
    (snapshot / "src" / "synapse" / "memflow_backend.py").write_text(
        "MEMFLOW_PROVIDER = 'legacy'\nclass MemflowBackend:\n    pass\n",
        encoding="utf-8",
    )
    (snapshot / "tests" / "test_memflow.py").write_text(
        "def test_memflow_backend():\n    pass\n",
        encoding="utf-8",
    )
    (snapshot / "tests" / "goldens" / "memflow.json").write_text(
        '{"backend":"memflow"}',
        encoding="utf-8",
    )

    manifest = build_synapse_retirement_manifest(snapshot)

    assert manifest.source_commit == "a" * 40
    assert "src/synapse/memflow_backend.py" in manifest.files
    assert "MemflowBackend" in manifest.symbols
    assert "MEMFLOW_PROVIDER" in manifest.symbols
    assert "tests/test_memflow.py" in manifest.tests
    assert "tests/goldens/memflow.json" in manifest.goldens
    assert len(manifest.active_reference_sha256) == 64
    assert manifest.operations


def test_synapse_retirement_manifest_can_be_signed(tmp_path: Path) -> None:
    snapshot = tmp_path / "synapse"
    snapshot.mkdir()
    _write_canonical_synapse_catalog(snapshot)
    (snapshot / "memflow.py").write_text("MEMFLOW = True\n", encoding="utf-8")
    keys, roster = _authority(tmp_path / "authority")

    manifest = build_synapse_retirement_manifest(
        snapshot,
        signer=OperationalSigner(keys, roster_version=roster.version),
        signer_key_id=roster.local_key_id,
        roster=roster,
    )

    assert manifest.signature
    verify_synapse_retirement_manifest(manifest, roster=roster)
    with pytest.raises(InventoryError, match="signature"):
        verify_synapse_retirement_manifest(
            replace(manifest, files=("other.py",)),
            roster=roster,
        )


def test_synapse_retirement_manifest_rejects_missing_canonical_catalog(tmp_path: Path) -> None:
    snapshot = tmp_path / "synapse"
    snapshot.mkdir()
    (snapshot / "source.json").write_bytes(canonical_json_bytes({"source_commit": "a" * 40}))

    with pytest.raises(InventoryError, match="canonical MCP catalog"):
        build_synapse_retirement_manifest(snapshot)
