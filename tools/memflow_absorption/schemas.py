"""Immutable schemas used by the Memflow absorption operator tools."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Literal

from memo.operational_event import canonical_json_bytes
from memo.operational_signing import SignatureEnvelope


@dataclass(frozen=True)
class SnapshotReceipt:
    schema: Literal["memo.cutover_snapshot_receipt.v1"]
    source: str
    target: str
    source_size: int
    source_mtime_ns: int
    source_mode: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SynapseDataReceipt:
    """Auditable outcome of one bounded Synapse data import attempt."""

    attempt_id: str
    input_sha256: str
    feedback_imported: int
    feedback_skipped: int
    eval_fixture_count: int
    event_ids: tuple[str, ...]
    status: Literal["applied", "reused"]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AuditExclusions:
    schema: Literal["memo.cutover_audit_exclusions.v1"]
    event_ids: tuple[str, ...]
    attempt_ids: tuple[str, ...]
    window_started_at: str
    window_ended_at: str
    signer_device_id: str
    signer_key_id: str
    roster_version: int
    issued_at: str
    signature: str

    def to_dict(self, *, blank_signature: bool = False) -> dict[str, object]:
        body = asdict(self)
        if blank_signature:
            body["signature"] = ""
        return body

    def signed_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict(blank_signature=True))

    def signature_envelope(self) -> SignatureEnvelope:
        return SignatureEnvelope(
            algorithm="ed25519",
            key_id=self.signer_key_id,
            roster_version=self.roster_version,
            signature=self.signature,
        )


@dataclass(frozen=True)
class UsageProof:
    schema: Literal["memo.cutover_usage_proof.v1"]
    device_id: str
    key_id: str
    roster_version: int
    query_version: str
    window_started_at: str
    window_ended_at: str
    snapshot_commit_oid: str
    raw_event_set_sha256: str
    exclusion_set_sha256: str
    issued_at: str
    signature: str

    def to_dict(self, *, blank_signature: bool = False) -> dict[str, object]:
        body = asdict(self)
        if blank_signature:
            body["signature"] = ""
        return body

    def signed_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict(blank_signature=True))

    def signature_envelope(self) -> SignatureEnvelope:
        return SignatureEnvelope(
            algorithm="ed25519",
            key_id=self.key_id,
            roster_version=self.roster_version,
            signature=self.signature,
        )


@dataclass(frozen=True)
class OperationRoute:
    route_id: str
    predicate: Mapping[str, object]
    memo_methods: tuple[str, ...]
    memo_mcp: tuple[str, ...]
    memo_cli: tuple[str, ...]
    parameter_mapping: Mapping[str, str]
    defaults: Mapping[str, object]
    result_mapping: Mapping[str, str]
    error_mapping: Mapping[str, str]
    transform_id: str
    fixture_sha256: tuple[str, ...]
    atomic_group: str | None
    fixture_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "route_id": self.route_id,
            "predicate": dict(self.predicate),
            "memo_methods": list(self.memo_methods),
            "memo_mcp": list(self.memo_mcp),
            "memo_cli": list(self.memo_cli),
            "parameter_mapping": dict(self.parameter_mapping),
            "defaults": dict(self.defaults),
            "result_mapping": dict(self.result_mapping),
            "error_mapping": dict(self.error_mapping),
            "transform_id": self.transform_id,
            "fixture_sha256": list(self.fixture_sha256),
            "atomic_group": self.atomic_group,
            "fixture_paths": list(self.fixture_paths),
        }


@dataclass(frozen=True)
class OperationMappingRow:
    source_operation: str
    source_commit: str
    source_tests: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    capability: str
    disposition: Literal["memo_native", "absorb", "internal", "delete"]
    routes: tuple[OperationRoute, ...]
    parity_tests: tuple[str, ...]
    deletion_proof: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_operation": self.source_operation,
            "source_commit": self.source_commit,
            "source_tests": list(self.source_tests),
            "evidence_ids": list(self.evidence_ids),
            "capability": self.capability,
            "disposition": self.disposition,
            "routes": [route.to_dict() for route in self.routes],
            "parity_tests": list(self.parity_tests),
            "deletion_proof": list(self.deletion_proof),
        }


@dataclass(frozen=True)
class SloBaseline:
    baseline_id: str
    source_commit: str
    workload_id: str
    machine_class: str
    window_started_at: str
    window_ended_at: str
    sample_count: int
    visibility_p50_ms: float
    visibility_p95_ms: float
    visibility_p99_ms: float
    visibility_max_ms: float
    recovery_max_ms: float
    error_rate: float
    data_loss_count: int
    duplicate_count: int
    tolerance_ratio: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CapabilityRow:
    name: str
    sources: tuple[str, ...]
    consumers: tuple[str, ...]
    window_started_at: str
    window_ended_at: str
    observed_calls: int
    observed_daemon_events: int
    machines: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    exclusion_counts: Mapping[str, int]
    evidence_complete: bool
    source_operations: tuple[str, ...]
    operation_mappings: tuple[OperationMappingRow, ...]
    slo_baseline_ids: tuple[str, ...]
    dependencies: tuple[str, ...]
    disposition: Literal["memo_native", "absorb", "internal", "delete"]
    memo_target: str
    parity_tests: tuple[str, ...]
    deletion_proof: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "sources": list(self.sources),
            "consumers": list(self.consumers),
            "window_started_at": self.window_started_at,
            "window_ended_at": self.window_ended_at,
            "observed_calls": self.observed_calls,
            "observed_daemon_events": self.observed_daemon_events,
            "machines": list(self.machines),
            "evidence_ids": list(self.evidence_ids),
            "exclusion_counts": dict(self.exclusion_counts),
            "evidence_complete": self.evidence_complete,
            "source_operations": list(self.source_operations),
            "operation_mappings": [mapping.to_dict() for mapping in self.operation_mappings],
            "slo_baseline_ids": list(self.slo_baseline_ids),
            "dependencies": list(self.dependencies),
            "disposition": self.disposition,
            "memo_target": self.memo_target,
            "parity_tests": list(self.parity_tests),
            "deletion_proof": list(self.deletion_proof),
        }


@dataclass(frozen=True)
class CapabilityManifest:
    schema: Literal["memo.cutover_capability_manifest.v1"]
    frozen_at: str
    window_started_at: str
    window_ended_at: str
    machine_ids: tuple[str, ...]
    source_receipt_sha256: Mapping[str, str]
    capabilities: tuple[CapabilityRow, ...]
    operation_mappings: tuple[OperationMappingRow, ...]
    slo_baselines: tuple[SloBaseline, ...]
    operation_map_sha256: str
    slo_baseline_sha256: str
    blockers: tuple[str, ...]
    frozen: bool
    signer_device_id: str
    signer_key_id: str
    roster_version: int
    signature: str

    def by_name(self, name: str) -> CapabilityRow | None:
        return next((row for row in self.capabilities if row.name == name), None)

    def operation_map_bytes(self) -> bytes:
        return canonical_json_bytes([row.to_dict() for row in self.operation_mappings])

    def slo_baseline_bytes(self) -> bytes:
        return canonical_json_bytes([row.to_dict() for row in self.slo_baselines])

    def to_dict(self, *, blank_signature: bool = False) -> dict[str, object]:
        body: dict[str, object] = {
            "schema": self.schema,
            "frozen_at": self.frozen_at,
            "window_started_at": self.window_started_at,
            "window_ended_at": self.window_ended_at,
            "machine_ids": list(self.machine_ids),
            "source_receipt_sha256": dict(self.source_receipt_sha256),
            "capabilities": [row.to_dict() for row in self.capabilities],
            "operation_mappings": [row.to_dict() for row in self.operation_mappings],
            "slo_baselines": [row.to_dict() for row in self.slo_baselines],
            "operation_map_sha256": self.operation_map_sha256,
            "slo_baseline_sha256": self.slo_baseline_sha256,
            "blockers": list(self.blockers),
            "frozen": self.frozen,
            "signer_device_id": self.signer_device_id,
            "signer_key_id": self.signer_key_id,
            "roster_version": self.roster_version,
            "signature": "" if blank_signature else self.signature,
        }
        return body

    def signed_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict(blank_signature=True))

    def signature_envelope(self) -> SignatureEnvelope:
        return SignatureEnvelope(
            algorithm="ed25519",
            key_id=self.signer_key_id,
            roster_version=self.roster_version,
            signature=self.signature,
        )


@dataclass(frozen=True)
class ProcessRecord:
    pid: int
    executable: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class ProcessSnapshot:
    captured_at: str
    records: tuple[ProcessRecord, ...]


@dataclass(frozen=True)
class LaunchdRecord:
    label: str
    plist_path: str
    program_arguments: tuple[str, ...]
    environment_keys: tuple[str, ...]
    loaded: bool


@dataclass(frozen=True)
class LaunchdSnapshot:
    captured_at: str
    records: tuple[LaunchdRecord, ...]


@dataclass(frozen=True)
class ConsumerInventoryRow:
    kind: Literal["source", "process", "launchd"]
    location: str
    references: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ConsumerInventory:
    schema: Literal["memo.cutover_consumer_inventory.v1"]
    rows: tuple[ConsumerInventoryRow, ...]
    blockers: tuple[str, ...]
    source_scan_sha256: str
    signer_device_id: str = ""
    signer_key_id: str = ""
    roster_version: int = 0
    signature: str = ""

    def to_dict(self, *, blank_signature: bool = False) -> dict[str, object]:
        return {
            "schema": self.schema,
            "rows": [row.to_dict() for row in self.rows],
            "blockers": list(self.blockers),
            "source_scan_sha256": self.source_scan_sha256,
            "signer_device_id": self.signer_device_id,
            "signer_key_id": self.signer_key_id,
            "roster_version": self.roster_version,
            "signature": "" if blank_signature else self.signature,
        }

    def signed_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict(blank_signature=True))

    def signature_envelope(self) -> SignatureEnvelope:
        return SignatureEnvelope(
            algorithm="ed25519",
            key_id=self.signer_key_id,
            roster_version=self.roster_version,
            signature=self.signature,
        )


@dataclass(frozen=True)
class SynapseOperation:
    """One canonical, source-backed Synapse operation surface."""

    source_operation: str
    source_files: tuple[str, ...]
    source_symbols: tuple[str, ...]
    consumers: tuple[str, ...]
    daemon_routes: tuple[str, ...]
    exclusion_reason: str | None
    fixture_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_operation": self.source_operation,
            "source_files": list(self.source_files),
            "source_symbols": list(self.source_symbols),
            "consumers": list(self.consumers),
            "daemon_routes": list(self.daemon_routes),
            "exclusion_reason": self.exclusion_reason,
            "fixture_paths": list(self.fixture_paths),
        }


@dataclass(frozen=True)
class SynapseRetirementManifest:
    schema: Literal["memo.synapse_retirement.v2"]
    source_commit: str
    files: tuple[str, ...]
    symbols: tuple[str, ...]
    tests: tuple[str, ...]
    goldens: tuple[str, ...]
    active_reference_sha256: str
    signer_key_id: str
    signature: str
    operations: tuple[SynapseOperation, ...] = ()

    def to_dict(self, *, blank_signature: bool = False) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_commit": self.source_commit,
            "files": list(self.files),
            "symbols": list(self.symbols),
            "tests": list(self.tests),
            "goldens": list(self.goldens),
            "active_reference_sha256": self.active_reference_sha256,
            "signer_key_id": self.signer_key_id,
            "signature": "" if blank_signature else self.signature,
            "operations": [row.to_dict() for row in self.operations],
        }

    def signed_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict(blank_signature=True))

    def signature_envelope(self, *, roster_version: int) -> SignatureEnvelope:
        return SignatureEnvelope(
            algorithm="ed25519",
            key_id=self.signer_key_id,
            roster_version=roster_version,
            signature=self.signature,
        )


class CutoverState(StrEnum):
    PREPARING = "PREPARING"
    READY = "READY"
    QUIESCING = "QUIESCING"
    QUIESCED = "QUIESCED"
    STAGED = "STAGED"
    ACTIVATION_READY = "ACTIVATION_READY"
    EPOCH_COMMITTED = "EPOCH_COMMITTED"
    ACTIVATED = "ACTIVATED"
    VERIFIED = "VERIFIED"
    ABORTING = "ABORTING"
    ABORTED = "ABORTED"
    RETIRED = "RETIRED"


class CutoverMode(StrEnum):
    ACTIVE = "active"
    QUIESCING = "quiescing"
    RETIRED = "retired"


@dataclass(frozen=True)
class FenceMarker:
    schema: Literal["memo.cutover_fence.v1"]
    attempt_id: str
    mode: CutoverMode
    epoch: int
    expected_commit: str
    runtime_digest: str
    device_id: str
    key_id: str
    issued_at: str
    expires_at: str
    control_oid: str
    control_sequence: int
    previous_control_oid: str
    signature: str

    def to_dict(self, *, blank_signature: bool = False) -> dict[str, object]:
        return {
            "schema": self.schema,
            "attempt_id": self.attempt_id,
            "mode": self.mode.value,
            "epoch": self.epoch,
            "expected_commit": self.expected_commit,
            "runtime_digest": self.runtime_digest,
            "device_id": self.device_id,
            "key_id": self.key_id,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "control_oid": self.control_oid,
            "control_sequence": self.control_sequence,
            "previous_control_oid": self.previous_control_oid,
            "signature": "" if blank_signature else self.signature,
        }

    def signed_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict(blank_signature=True))


@dataclass(frozen=True)
class DrainSnapshot:
    schema: Literal["memo.cutover_drain_snapshot.v1"]
    captured_at: str
    requests: int
    event_append: int
    delivery: int
    ack: int
    cursor: int
    sync: int
    git_push: int
    autonomous_loops: int
    writable_handles: int
    inflight_total: int
    clean: bool
    last_fsync_at: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FinalFenceProof:
    schema: Literal["memo.cutover_final_fence_proof.v1"]
    attempt_id: str
    control_oid: str
    zero_drain_at: str
    memflow_remote_commit_oid: str
    origin_heads: Mapping[str, Mapping[str, object]]
    source_snapshot_sha256: Mapping[str, str]
    signer_device_id: str
    signer_key_id: str
    roster_version: int
    signature: str

    def to_dict(self, *, blank_signature: bool = False) -> dict[str, object]:
        return {
            "schema": self.schema,
            "attempt_id": self.attempt_id,
            "control_oid": self.control_oid,
            "zero_drain_at": self.zero_drain_at,
            "memflow_remote_commit_oid": self.memflow_remote_commit_oid,
            "origin_heads": {key: dict(value) for key, value in self.origin_heads.items()},
            "source_snapshot_sha256": dict(self.source_snapshot_sha256),
            "signer_device_id": self.signer_device_id,
            "signer_key_id": self.signer_key_id,
            "roster_version": self.roster_version,
            "signature": "" if blank_signature else self.signature,
        }

    def signed_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict(blank_signature=True))


@dataclass(frozen=True)
class CutoverControlRecord:
    schema: Literal["memo.cutover_control_record.v1"]
    control_oid: str
    state: CutoverState
    sequence: int
    previous_control_oid: str
    attempt_id: str
    roster_version: int
    signer_device_id: str
    signer_key_id: str
    issued_at: str
    signature: str

    def to_dict(self, *, blank_signature: bool = False) -> dict[str, object]:
        return {
            "schema": self.schema,
            "control_oid": self.control_oid,
            "state": self.state.value,
            "sequence": self.sequence,
            "previous_control_oid": self.previous_control_oid,
            "attempt_id": self.attempt_id,
            "roster_version": self.roster_version,
            "signer_device_id": self.signer_device_id,
            "signer_key_id": self.signer_key_id,
            "issued_at": self.issued_at,
            "signature": "" if blank_signature else self.signature,
        }

    @property
    def canonical_payload(self) -> bytes:
        return canonical_json_bytes(self.to_dict(blank_signature=True))

    def signature_envelope(self) -> SignatureEnvelope:
        return SignatureEnvelope(
            algorithm="ed25519",
            key_id=self.signer_key_id,
            roster_version=self.roster_version,
            signature=self.signature,
        )


@dataclass(frozen=True)
class VerifiedControlRecord:
    control_oid: str
    canonical_payload: bytes
    state: CutoverState
    sequence: int
    previous_control_oid: str
    roster_version: int
    verified_at: str
    signer_device_id: str
    signer_key_id: str


JsonObject = dict[str, Any]
