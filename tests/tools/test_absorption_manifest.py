from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
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
from memo.operational_signing import OperationalSigner, OperationalVerifier
from tools.memflow_absorption.__main__ import _verified_receipt_ids
from tools.memflow_absorption.manifest import (
    CAPABILITY_MANIFEST_DOMAIN,
    ManifestError,
    build_capability_manifest,
    sign_audit_exclusions,
    sign_usage_proof,
    verify_audit_exclusions,
    verify_capability_manifest,
)
from tools.memflow_absorption.schemas import AuditExclusions, UsageProof
from tools.memflow_absorption.source_receipt import (
    SourceBucket,
    SourceReceiptV2,
    sign_source_receipt,
)

FROZEN_AT = "2026-07-30T00:00:00Z"
WINDOW_STARTED_AT = "2026-05-01T00:00:00Z"
SOURCE_COMMIT = "f" * 40
SANITIZED_FIXTURES = Path(__file__).parents[1] / "fixtures" / "memflow_absorption"


def _authority(tmp_path: Path) -> tuple[DeviceKeyStore, VerificationRoster]:
    keys = DeviceKeyStore.in_memory()
    mac_a = keys.generate(device_id="mac-a")
    pins = AuthorityPinStore._for_test(
        tmp_path,
        provider=InMemoryAuthorityPinProvider(),
    )
    roster = VerificationRoster.bootstrap(
        device_id="mac-a",
        key=mac_a,
        root=tmp_path,
        pin_store=pins,
    )
    mac_b = keys.generate(device_id="mac-b", roles=("origin",), enrollment_sequence=2)
    updated = roster.with_keys(
        version=2,
        peers=("mac-a", "mac-b"),
        keys=(mac_a, mac_b),
        signer=OperationalSigner(keys, roster_version=roster.version),
        root=tmp_path,
        pin_store=pins,
    )
    return keys, updated


def test_sanitized_route_fixture_digest_is_executable_authority() -> None:
    result = (SANITIZED_FIXTURES / "result.json").read_bytes()
    mappings = json.loads(
        (SANITIZED_FIXTURES / "mapping-candidates.json").read_text(encoding="utf-8")
    )

    assert canonical_json_bytes(json.loads(result)) + b"\n" == result
    assert mappings[0]["routes"][0]["fixture_sha256"] == [hashlib.sha256(result).hexdigest()]


def _route(operation: str, fixture_digest: str) -> dict[str, Any]:
    return {
        "route_id": f"{operation}-default",
        "predicate": {"mode": {"eq": "default"}},
        "memo_methods": [f"Memory.{operation}"],
        "memo_mcp": [f"memo_{operation}"],
        "memo_cli": [f"memo {operation.replace('_', ' ')}"],
        "parameter_mapping": {"value": "value"},
        "defaults": {"mode": "default"},
        "result_mapping": {"status": "status"},
        "error_mapping": {"invalid": "invalid"},
        "transform_id": f"{operation}-v1",
        "fixture_sha256": [fixture_digest],
        "atomic_group": None,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _fixture_tree(
    tmp_path: Path,
    *,
    ambiguous_task: bool = True,
    unknown_operation: bool = False,
    coverage_gap: bool = False,
    slo_samples: int = 100,
) -> tuple[Path, Path, Path, dict[str, list[dict[str, Any]]]]:
    memo_snapshot = tmp_path / "memo-snapshot"
    memflow_snapshot = tmp_path / "memflow-snapshot"
    usage_snapshot = tmp_path / "usage-snapshot"
    for root, digest in (
        (memo_snapshot, "1" * 64),
        (memflow_snapshot, "2" * 64),
        (usage_snapshot, "3" * 64),
    ):
        _write_json(root / "snapshot-receipt.json", {"sha256": digest})

    operations = [
        {
            "source_operation": "flow_continuity",
            "source_tests": ["tests/test_continuity.py::test_packet"],
            "capability": "continuity",
            "latency_sensitive": True,
        },
        {
            "source_operation": "flow_tasks",
            "source_tests": ["tests/test_tasks.py::test_create"],
            "capability": "tasks",
            "latency_sensitive": False,
        },
        {
            "source_operation": "flow_unused",
            "source_tests": ["tests/test_unused.py::test_absent"],
            "capability": "unused",
            "latency_sensitive": False,
        },
    ]
    _write_json(
        memflow_snapshot / "source.json",
        {"source_commit": SOURCE_COMMIT, "operations": operations},
    )
    fixture_digest = hashlib.sha256(canonical_json_bytes({"status": "ok"})).hexdigest()
    mappings = [
        {
            "source_operation": "flow_continuity",
            "source_commit": SOURCE_COMMIT,
            "source_tests": ["tests/test_continuity.py::test_packet"],
            "evidence_ids": ["e-cont-a", "e-cont-b"],
            "capability": "continuity",
            "disposition": "absorb",
            "routes": [_route("continuity", fixture_digest)],
            "parity_tests": ["tests/test_memflow_parity_fixtures.py::test_continuity"],
            "deletion_proof": [],
        },
        {
            "source_operation": "flow_tasks",
            "source_commit": SOURCE_COMMIT,
            "source_tests": ["tests/test_tasks.py::test_create"],
            "evidence_ids": ["e-task-a"],
            "capability": "tasks",
            "disposition": "absorb",
            "routes": [_route("tasks", fixture_digest)],
            "parity_tests": ["tests/test_memflow_parity_fixtures.py::test_tasks"],
            "deletion_proof": [],
        },
        {
            "source_operation": "flow_unused",
            "source_commit": SOURCE_COMMIT,
            "source_tests": ["tests/test_unused.py::test_absent"],
            "evidence_ids": [],
            "capability": "unused",
            "disposition": "delete",
            "routes": [],
            "parity_tests": [],
            "deletion_proof": ["complete-zero-use-90d", "source-test-retired"],
        },
    ]
    _write_json(memo_snapshot / "mapping-candidates.json", mappings)
    events = {
        "mac-a": [
            {
                "event_id": "evt-cont-a",
                "attempt_id": "live-a",
                "device_id": "mac-a",
                "occurred_at": WINDOW_STARTED_AT,
                "source_operation": "flow_continuity",
                "kind": "call",
                "evidence_id": "e-cont-a",
                "ambiguous": False,
                "synthetic": False,
            },
            {
                "event_id": "evt-task-a",
                "attempt_id": "live-a",
                "device_id": "mac-a",
                "occurred_at": "2026-06-01T00:00:00Z",
                "source_operation": "flow_tasks",
                "kind": "call",
                "evidence_id": "e-task-a",
                "ambiguous": ambiguous_task,
                "synthetic": False,
            },
            {
                "event_id": "evt-audit-1",
                "attempt_id": "audit-1",
                "device_id": "mac-a",
                "occurred_at": "2026-06-02T00:00:00Z",
                "source_operation": "flow_unknown_audit",
                "kind": "call",
                "evidence_id": "e-audit",
                "ambiguous": False,
                "synthetic": True,
            },
        ],
        "mac-b": [
            {
                "event_id": "evt-cont-b",
                "attempt_id": "live-b",
                "device_id": "mac-b",
                "occurred_at": FROZEN_AT,
                "source_operation": (
                    "flow_unknown_live" if unknown_operation else "flow_continuity"
                ),
                "kind": "daemon",
                "evidence_id": "e-cont-b",
                "ambiguous": False,
                "synthetic": False,
            }
        ],
    }
    _write_json(
        usage_snapshot / "usage.json",
        {
            "frozen_at": FROZEN_AT,
            "window_started_at": WINDOW_STARTED_AT,
            "window_ended_at": FROZEN_AT,
            "machine_ids": ["mac-a", "mac-b"],
            "events": events,
            "proofs": [],
            "slo_baselines": [
                {
                    "baseline_id": "continuity-live",
                    "source_commit": SOURCE_COMMIT,
                    "workload_id": "flow_continuity",
                    "machine_class": "apple-silicon",
                    "window_started_at": WINDOW_STARTED_AT,
                    "window_ended_at": FROZEN_AT,
                    "sample_count": slo_samples,
                    "visibility_p50_ms": 10.0,
                    "visibility_p95_ms": 20.0,
                    "visibility_p99_ms": 30.0,
                    "visibility_max_ms": 40.0,
                    "recovery_max_ms": 100.0,
                    "error_rate": 0.0,
                    "data_loss_count": 0,
                    "duplicate_count": 0,
                    "tolerance_ratio": 1.25,
                }
            ],
            "coverage_gaps": ["2026-06-01T03:00:00Z"] if coverage_gap else [],
            "fresh_source_receipts": {"mac-a": True, "mac-b": True},
        },
    )
    return memo_snapshot, memflow_snapshot, usage_snapshot, events


def _signed_inputs(
    tmp_path: Path,
    usage_snapshot: Path,
    events: dict[str, list[dict[str, Any]]],
    keys: DeviceKeyStore,
    roster: VerificationRoster,
) -> AuditExclusions:
    mac_a_key_id = next(key.key_id for key in roster.keys if key.device_id == "mac-a")
    audit_signer = OperationalSigner(keys, roster_version=roster.version)
    unsigned = AuditExclusions(
        schema="memo.cutover_audit_exclusions.v1",
        event_ids=("evt-audit-1",),
        attempt_ids=("audit-1",),
        window_started_at=WINDOW_STARTED_AT,
        window_ended_at=FROZEN_AT,
        signer_device_id="mac-a",
        signer_key_id=mac_a_key_id,
        roster_version=roster.version,
        issued_at="2026-07-30T00:01:00Z",
        signature="",
    )
    exclusions = sign_audit_exclusions(unsigned, signer=audit_signer)
    usage = json.loads((usage_snapshot / "usage.json").read_text(encoding="utf-8"))
    proofs: list[dict[str, object]] = []
    exclusion_digest = hashlib.sha256(
        canonical_json_bytes(
            {"event_ids": list(exclusions.event_ids), "attempt_ids": list(exclusions.attempt_ids)}
        )
    ).hexdigest()
    for device_id in ("mac-a", "mac-b"):
        key_id = next(key.key_id for key in roster.keys if key.device_id == device_id)
        proof = UsageProof(
            schema="memo.cutover_usage_proof.v1",
            device_id=device_id,
            key_id=key_id,
            roster_version=roster.version,
            query_version="usage-v1",
            window_started_at=WINDOW_STARTED_AT,
            window_ended_at=FROZEN_AT,
            snapshot_commit_oid=SOURCE_COMMIT,
            raw_event_set_sha256=hashlib.sha256(
                canonical_json_bytes(events[device_id])
            ).hexdigest(),
            exclusion_set_sha256=exclusion_digest,
            issued_at="2026-07-30T00:02:00Z",
            signature="",
        )
        proofs.append(
            sign_usage_proof(
                proof,
                signer=OperationalSigner(keys, roster_version=roster.version),
            ).to_dict()
        )
    usage["proofs"] = proofs
    # Source receipts are the authoritative, signed aggregate for each machine.
    # Build one bucket per hour so the exact cutover window is covered.
    start = datetime.fromisoformat(WINDOW_STARTED_AT.replace("Z", "+00:00"))
    end = datetime.fromisoformat(FROZEN_AT.replace("Z", "+00:00"))
    receipts: list[dict[str, object]] = []
    for device_id in ("mac-a", "mac-b"):
        device_events = events[device_id]
        digest = hashlib.sha256(
            json.dumps(device_events, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        buckets: list[SourceBucket] = []
        cursor = start
        while cursor < end:
            bucket_end = cursor + timedelta(hours=1)
            count = sum(
                1
                for event in device_events
                if cursor <= datetime.fromisoformat(event["occurred_at"].replace("Z", "+00:00")) < bucket_end
            )
            buckets.append(
                SourceBucket(
                    start=cursor.isoformat().replace("+00:00", "Z"),
                    end=bucket_end.isoformat().replace("+00:00", "Z"),
                    count=count,
                    digest=hashlib.sha256(f"{device_id}:{cursor.isoformat()}".encode()).hexdigest(),
                )
            )
            cursor = bucket_end
        key_id = next(key.key_id for key in roster.keys if key.device_id == device_id)
        receipt = SourceReceiptV2(
            device_id=device_id,
            key_id=key_id,
            roster_id=roster.roster_hash,
            query="memflow.cutover.events",
            extractor_version="memflow-extractor-v2",
            snapshot_commit=SOURCE_COMMIT,
            raw_event_set_sha256=digest,
            window_start=WINDOW_STARTED_AT,
            window_end=FROZEN_AT,
            issued_at=FROZEN_AT,
            collected_at=FROZEN_AT,
            cursor=f"cursor-{device_id}-final",
            extraction_complete=True,
            hourly_buckets=tuple(buckets),
            frozen_at=FROZEN_AT,
        )
        receipts.append(sign_source_receipt(receipt, signer=OperationalSigner(keys, roster_version=roster.version)).to_dict())
    usage["source_receipts_v2"] = receipts
    _write_json(usage_snapshot / "usage.json", usage)
    return exclusions


def _build(
    tmp_path: Path,
    *,
    ambiguous_task: bool = True,
    unknown_operation: bool = False,
    coverage_gap: bool = False,
    slo_samples: int = 100,
):
    keys, roster = _authority(tmp_path / "authority")
    memo, memflow, usage, events = _fixture_tree(
        tmp_path,
        ambiguous_task=ambiguous_task,
        unknown_operation=unknown_operation,
        coverage_gap=coverage_gap,
        slo_samples=slo_samples,
    )
    exclusions = _signed_inputs(tmp_path, usage, events, keys, roster)
    signer = OperationalSigner(keys, roster_version=roster.version)
    manifest = build_capability_manifest(
        memo,
        memflow,
        usage,
        exclusions,
        signer=signer,
        signer_key_id=roster.local_key_id,
        roster=roster,
    )
    return manifest, exclusions, signer, roster


def test_manifest_excludes_exact_audit_and_blocks_ambiguous_traffic(tmp_path: Path) -> None:
    manifest, _exclusions, _signer, _roster = _build(tmp_path)

    assert manifest.by_name("continuity").disposition == "absorb"
    assert manifest.by_name("tasks").disposition == "absorb"
    assert manifest.by_name("audit-query") is None
    assert manifest.by_name("unused").disposition == "delete"
    assert manifest.frozen is False
    assert manifest.blockers == ("tasks:ambiguous-traffic",)
    assert manifest.signature == ""


def test_complete_signed_manifest_freezes_and_verifies(tmp_path: Path) -> None:
    manifest, _exclusions, _signer, roster = _build(tmp_path, ambiguous_task=False)

    assert manifest.frozen is True
    assert manifest.blockers == ()
    assert manifest.window_started_at == WINDOW_STARTED_AT
    assert manifest.window_ended_at == FROZEN_AT
    assert manifest.machine_ids == ("mac-a", "mac-b")
    assert (
        manifest.operation_map_sha256 == hashlib.sha256(manifest.operation_map_bytes()).hexdigest()
    )
    assert manifest.slo_baseline_sha256 == hashlib.sha256(manifest.slo_baseline_bytes()).hexdigest()
    verify_capability_manifest(manifest, roster=roster)
    OperationalVerifier().verify(
        domain=CAPABILITY_MANIFEST_DOMAIN,
        payload=manifest.signed_bytes(),
        envelope=manifest.signature_envelope(),
        roster=roster,
    )


@pytest.mark.parametrize(
    ("options", "blocker"),
    [
        ({"unknown_operation": True, "ambiguous_task": False}, "unknown:flow_unknown_live"),
        ({"coverage_gap": True, "ambiguous_task": False}, "usage:coverage-gap"),
        ({"slo_samples": 99, "ambiguous_task": False}, "slo:continuity-live:sample-count"),
    ],
)
def test_manifest_fail_closed_blockers(
    tmp_path: Path,
    options: dict[str, object],
    blocker: str,
) -> None:
    manifest, _exclusions, _signer, _roster = _build(tmp_path, **options)

    assert manifest.frozen is False
    assert blocker in manifest.blockers


def test_exclusion_tamper_or_window_mismatch_is_rejected(tmp_path: Path) -> None:
    manifest, exclusions, signer, roster = _build(tmp_path, ambiguous_task=False)
    assert manifest.frozen
    tampered = replace(exclusions, event_ids=("evt-audit-1", "evt-live"))

    memo, memflow, usage, _events = _fixture_tree(tmp_path / "retry", ambiguous_task=False)
    with pytest.raises(ManifestError, match="exclusion signature"):
        build_capability_manifest(
            memo,
            memflow,
            usage,
            tampered,
            signer=signer,
            signer_key_id=roster.local_key_id,
            roster=roster,
        )


def test_duplicate_operation_disposition_is_rejected(tmp_path: Path) -> None:
    keys, roster = _authority(tmp_path / "authority")
    memo, memflow, usage, events = _fixture_tree(tmp_path, ambiguous_task=False)
    mappings = json.loads((memo / "mapping-candidates.json").read_text(encoding="utf-8"))
    mappings.append(mappings[0])
    _write_json(memo / "mapping-candidates.json", mappings)
    exclusions = _signed_inputs(tmp_path, usage, events, keys, roster)

    with pytest.raises(ManifestError, match="exactly one disposition"):
        build_capability_manifest(
            memo,
            memflow,
            usage,
            exclusions,
            signer=OperationalSigner(keys, roster_version=roster.version),
            signer_key_id=roster.local_key_id,
            roster=roster,
        )


def test_signed_usage_proof_tamper_is_rejected(tmp_path: Path) -> None:
    keys, roster = _authority(tmp_path / "authority")
    memo, memflow, usage, events = _fixture_tree(tmp_path, ambiguous_task=False)
    exclusions = _signed_inputs(tmp_path, usage, events, keys, roster)
    usage_record = json.loads((usage / "usage.json").read_text(encoding="utf-8"))
    usage_record["events"]["mac-b"][0]["evidence_id"] = "tampered"
    _write_json(usage / "usage.json", usage_record)

    with pytest.raises(ManifestError, match="provenance"):
        build_capability_manifest(
            memo,
            memflow,
            usage,
            exclusions,
            signer=OperationalSigner(keys, roster_version=roster.version),
            signer_key_id=roster.local_key_id,
            roster=roster,
        )


def test_usage_on_delete_operation_blocks_freeze(tmp_path: Path) -> None:
    keys, roster = _authority(tmp_path / "authority")
    memo, memflow, usage, events = _fixture_tree(tmp_path, ambiguous_task=False)
    events["mac-b"].append(
        {
            "event_id": "evt-unused-live",
            "attempt_id": "live-b",
            "device_id": "mac-b",
            "occurred_at": "2026-07-01T00:00:00Z",
            "source_operation": "flow_unused",
            "kind": "call",
            "evidence_id": "e-unused",
            "ambiguous": False,
            "synthetic": False,
            "client_name": "audit-like-name",
            "topic": "evaluation-suffix",
        }
    )
    usage_record = json.loads((usage / "usage.json").read_text(encoding="utf-8"))
    usage_record["events"] = events
    _write_json(usage / "usage.json", usage_record)
    exclusions = _signed_inputs(tmp_path, usage, events, keys, roster)

    manifest = build_capability_manifest(
        memo,
        memflow,
        usage,
        exclusions,
        signer=OperationalSigner(keys, roster_version=roster.version),
        signer_key_id=roster.local_key_id,
        roster=roster,
    )

    assert "unused:deletion-proof" in manifest.blockers
    assert manifest.by_name("unused").observed_calls == 1
    assert manifest.signature == ""


def test_route_predicates_use_closed_match_language(tmp_path: Path) -> None:
    keys, roster = _authority(tmp_path / "authority")
    memo, memflow, usage, events = _fixture_tree(tmp_path, ambiguous_task=False)
    mappings = json.loads((memo / "mapping-candidates.json").read_text(encoding="utf-8"))
    mappings[0]["routes"][0]["predicate"] = {"mode": {"regex": ".*"}}
    _write_json(memo / "mapping-candidates.json", mappings)
    exclusions = _signed_inputs(tmp_path, usage, events, keys, roster)

    with pytest.raises(ManifestError, match="closed match expression"):
        build_capability_manifest(
            memo,
            memflow,
            usage,
            exclusions,
            signer=OperationalSigner(keys, roster_version=roster.version),
            signer_key_id=roster.local_key_id,
            roster=roster,
        )


def test_overlapping_route_predicates_block_freeze(tmp_path: Path) -> None:
    keys, roster = _authority(tmp_path / "authority")
    memo, memflow, usage, events = _fixture_tree(tmp_path, ambiguous_task=False)
    mappings = json.loads((memo / "mapping-candidates.json").read_text(encoding="utf-8"))
    first = mappings[0]["routes"][0]
    overlapping = dict(first)
    overlapping["route_id"] = "continuity-overlap"
    overlapping["predicate"] = {"mode": {"in": ["default", "other"]}}
    mappings[0]["routes"].append(overlapping)
    _write_json(memo / "mapping-candidates.json", mappings)
    exclusions = _signed_inputs(tmp_path, usage, events, keys, roster)

    manifest = build_capability_manifest(
        memo,
        memflow,
        usage,
        exclusions,
        signer=OperationalSigner(keys, roster_version=roster.version),
        signer_key_id=roster.local_key_id,
        roster=roster,
    )

    assert "continuity:operation-map" in manifest.blockers


def test_catalog_preflight_requires_canonical_signed_two_mac_receipts(tmp_path: Path) -> None:
    keys, roster = _authority(tmp_path / "authority")
    _memo, _memflow, usage, events = _fixture_tree(tmp_path, ambiguous_task=False)
    exclusions = _signed_inputs(tmp_path, usage, events, keys, roster)
    usage_record = json.loads((usage / "usage.json").read_text(encoding="utf-8"))
    proof_paths = []
    for index, proof in enumerate(usage_record["proofs"]):
        path = tmp_path / f"proof-{index}.json"
        _write_json(path, proof)
        proof_paths.append(path)
    exclusion_path = tmp_path / "exclusions.json"
    _write_json(exclusion_path, exclusions.to_dict())
    args = SimpleNamespace(usage_proof=proof_paths, exclusion=[exclusion_path])

    proof_ids, exclusion_ids = _verified_receipt_ids(args, roster, SOURCE_COMMIT)

    assert proof_ids == sorted(proof_ids)
    assert {proof_id.split(":")[0] for proof_id in proof_ids} == {"mac-a", "mac-b"}
    assert exclusion_ids == [f"mac-a:{exclusions.signer_key_id}"]

    tampered = json.loads(proof_paths[0].read_text(encoding="utf-8"))
    tampered["signature"] = "invalid"
    _write_json(proof_paths[0], tampered)
    with pytest.raises(SystemExit, match="receipt validation"):
        _verified_receipt_ids(args, roster, SOURCE_COMMIT)


def test_catalog_preflight_rejects_signed_proof_for_another_commit(tmp_path: Path) -> None:
    keys, roster = _authority(tmp_path / "authority")
    _memo, _memflow, usage, events = _fixture_tree(tmp_path, ambiguous_task=False)
    exclusions = _signed_inputs(tmp_path, usage, events, keys, roster)
    usage_record = json.loads((usage / "usage.json").read_text(encoding="utf-8"))
    proof_paths = []
    for index, value in enumerate(usage_record["proofs"]):
        proof = UsageProof(**value)
        if index == 0:
            proof = sign_usage_proof(
                replace(proof, snapshot_commit_oid="e" * 40, signature=""),
                signer=OperationalSigner(keys, roster_version=roster.version),
            )
        path = tmp_path / f"proof-{index}.json"
        _write_json(path, proof.to_dict())
        proof_paths.append(path)
    exclusion_path = tmp_path / "exclusions.json"
    _write_json(exclusion_path, exclusions.to_dict())
    args = SimpleNamespace(usage_proof=proof_paths, exclusion=[exclusion_path])

    with pytest.raises(SystemExit, match="snapshot commit"):
        _verified_receipt_ids(args, roster, SOURCE_COMMIT)


def test_audit_exclusion_rejects_forged_signer_device_id(tmp_path: Path) -> None:
    keys, roster = _authority(tmp_path / "authority")
    _memo, _memflow, usage, events = _fixture_tree(tmp_path, ambiguous_task=False)
    exclusions = _signed_inputs(tmp_path, usage, events, keys, roster)

    with pytest.raises(ManifestError, match="provenance"):
        verify_audit_exclusions(
            replace(exclusions, signer_device_id="mac-b"),
            roster=roster,
        )
