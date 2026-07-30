"""Operator-staged Memo replacements for retired Synapse consumers."""

from __future__ import annotations

import plistlib

import pytest

from tools.memflow_absorption.consumer_migration import (
    ConsumerMigrationError,
    build_consumer_replacement_plan,
    render_memo_launch_agents,
    verify_no_synapse_runtime_reference,
)
from tools.memflow_absorption.schemas import (
    CapabilityManifest,
    ConsumerInventory,
    ConsumerInventoryRow,
)


@pytest.fixture
def manifest() -> CapabilityManifest:
    return CapabilityManifest(
        schema="memo.cutover_capability_manifest.v1",
        frozen_at="2026-07-30T00:00:00Z",
        window_started_at="2026-04-30T00:00:00Z",
        window_ended_at="2026-07-30T00:00:00Z",
        machine_ids=("device-a",),
        source_receipt_sha256={},
        capabilities=(),
        operation_mappings=(),
        slo_baselines=(),
        operation_map_sha256="",
        slo_baseline_sha256="",
        blockers=(),
        frozen=True,
        signer_device_id="device-a",
        signer_key_id="key-a",
        roster_version=1,
        signature="",
    )


@pytest.fixture
def inventory() -> ConsumerInventory:
    labels = (
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
    return ConsumerInventory(
        schema="memo.cutover_consumer_inventory.v1",
        rows=(
            *(
                ConsumerInventoryRow(
                    kind="launchd",
                    location=f"{label}:/operator/archive/{label}.plist",
                    references=("synapse",),
                    label=label,
                    active=True,
                )
                for label in labels
            ),
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
    )


def test_replacement_plan_maps_active_jobs_to_memo_owned_commands(inventory, manifest):
    plan = build_consumer_replacement_plan(inventory, manifest)

    assert plan.by_old_label("com.synapse.whatsapp-ingest").command[:2] == ("memo", "import")
    assert plan.by_old_label("com.synapse.watcher").command[:2] == ("memo", "watch")
    assert plan.by_old_label("com.synapse.memo-recall-daemon").command == (
        "memo",
        "recall-daemon",
        "_serve",
    )
    assert plan.by_old_label("com.synapse.dashboard").command == ("client-close-reconnect",)
    assert plan.by_old_label("com.synapse.memo-nightly").command == ("memo", "dream", "run")
    assert plan.by_old_label("com.synapse.dream-synthesis").command == (
        "memo",
        "dream",
        "run",
    )
    assert plan.by_old_label("com.synapse.vault-ingest").command == ("memo", "reindex")
    assert all(row.old_label != "com.synapse.unloaded" for row in plan.rows)
    assert len(plan.digest) == 64
    assert all(len(row.config_sha256) == 64 for row in plan.rows)


def test_rendered_plists_contain_no_synapse_or_memflow_paths(inventory, manifest, tmp_path):
    plan = build_consumer_replacement_plan(inventory, manifest)
    paths = render_memo_launch_agents(plan, tmp_path)

    assert paths
    assert all("synapse" not in path.read_text().lower() for path in paths)
    assert all("memflow" not in path.read_text().lower() for path in paths)
    assert all("MEMO_NONINTERACTIVE" in path.read_text() for path in paths)
    assert all("client.close-reconnect" not in path.read_text() for path in paths)
    rendered = {
        plistlib.loads(path.read_bytes())["Label"]: plistlib.loads(path.read_bytes())
        for path in paths
    }
    assert rendered["com.memo.watch"]["KeepAlive"] is True
    assert "KeepAlive" not in rendered["com.memo.digest"]
    assert "KeepAlive" not in rendered["com.memo.dream"]
    assert "KeepAlive" not in rendered["com.memo.reindex"]


def test_plan_ignores_non_launchd_reference_rows(inventory, manifest):
    inventory = ConsumerInventory(
        schema=inventory.schema,
        rows=(
            *inventory.rows,
            ConsumerInventoryRow(
                kind="source",
                location="/operator/snapshot/src/synapse/runtime.py",
                references=("synapse",),
                label="src/synapse/runtime.py",
            ),
            ConsumerInventoryRow(
                kind="process",
                location="pid:7:/operator/archive/synapse",
                references=("synapse",),
                label="/operator/archive/synapse runtime loop",
            ),
        ),
        blockers=(),
        source_scan_sha256=inventory.source_scan_sha256,
    )

    plan = build_consumer_replacement_plan(inventory, manifest)

    assert all(row.old_label.startswith("com.synapse.") for row in plan.rows)


def test_plan_rejects_unmapped_active_retired_job(manifest):
    inventory = ConsumerInventory(
        schema="memo.cutover_consumer_inventory.v1",
        rows=(
            ConsumerInventoryRow(
                kind="launchd",
                location="com.synapse.unknown:/operator/archive/unknown.plist",
                references=("synapse",),
                label="com.synapse.unknown",
            ),
        ),
        blockers=(),
        source_scan_sha256="a" * 64,
    )

    with pytest.raises(ConsumerMigrationError, match="no Memo-owned replacement"):
        build_consumer_replacement_plan(inventory, manifest)


def test_verifier_rejects_retired_reference_in_staged_output(inventory, manifest, tmp_path):
    plan = build_consumer_replacement_plan(inventory, manifest)
    (tmp_path / "bad.plist").write_text("legacy memflow runtime", encoding="utf-8")

    with pytest.raises(ConsumerMigrationError, match="retired runtime reference"):
        verify_no_synapse_runtime_reference(tmp_path, plan)


def test_renderer_rejects_tampered_plan_digest(inventory, manifest, tmp_path):
    plan = build_consumer_replacement_plan(inventory, manifest)
    tampered = type(plan)(rows=plan.rows, digest="0" * 64)

    with pytest.raises(ConsumerMigrationError, match="plan digest is invalid"):
        render_memo_launch_agents(tampered, tmp_path)


def test_renderer_rejects_existing_output_symlink(inventory, manifest, tmp_path):
    plan = build_consumer_replacement_plan(inventory, manifest)
    launch_agents = tmp_path / "LaunchAgents"
    launch_agents.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("operator-owned", encoding="utf-8")
    (launch_agents / "com.memo.watch.plist").symlink_to(outside)

    with pytest.raises(ConsumerMigrationError, match="must not be a symlink"):
        render_memo_launch_agents(plan, tmp_path)

    assert outside.read_text(encoding="utf-8") == "operator-owned"


def test_verifier_rejects_retired_runtime_filename(inventory, manifest, tmp_path):
    plan = build_consumer_replacement_plan(inventory, manifest)
    render_memo_launch_agents(plan, tmp_path)
    (tmp_path / "legacy-synapse.plist").write_text("memo watch", encoding="utf-8")

    with pytest.raises(ConsumerMigrationError, match="retired runtime path"):
        verify_no_synapse_runtime_reference(tmp_path, plan)
