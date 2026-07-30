"""Build and verify the signed capability authority for Memflow absorption."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from memo.atomic_io import open_secure_directory
from memo.errors import SignatureError
from memo.operational_event import canonical_json_bytes
from memo.operational_roster import VerificationRoster
from memo.operational_signing import OperationalSigner, OperationalVerifier, SignatureEnvelope
from tools.memflow_absorption.schemas import (
    AuditExclusions,
    CapabilityManifest,
    CapabilityRow,
    OperationMappingRow,
    OperationRoute,
    SloBaseline,
    SynapseOperation,
    UsageProof,
)
from tools.memflow_absorption.source_receipt import (
    SourceBucket,
    SourceReceiptV2,
    verify_source_receipt,
)
from tools.memflow_absorption.synapse_catalog import (
    SynapseCatalogError,
    discover_synapse_operations,
)
from tools.memflow_absorption.transforms import FrozenTransformRegistry, verify_route_fixtures

AUDIT_EXCLUSIONS_DOMAIN = "memo.cutover.audit_exclusions.v1"
USAGE_PROOF_DOMAIN = "memo.cutover.usage_proof.v1"
CAPABILITY_MANIFEST_DOMAIN = "memo.cutover.capability_manifest.v1"
_SHA256_LENGTH = 64


class ManifestError(RuntimeError):
    """Cutover evidence is incomplete, ambiguous, or unauthenticated."""


def _load_json(path: Path) -> Any:
    try:
        absolute = Path(os.path.abspath(os.fspath(path)))
        receipt_name = f"{absolute.name}.receipt.json"
        with open_secure_directory(absolute.parent) as directory:
            encoded, observed = directory.read_bytes_snapshot(absolute.name)
            receipt_encoded, _receipt_stat = directory.read_bytes_snapshot(receipt_name)
        receipt = json.loads(receipt_encoded)
        if canonical_json_bytes(receipt) != receipt_encoded or receipt.get("schema") != "memo.cutover_snapshot_receipt.v2":
            raise ManifestError(f"invalid snapshot receipt: {path}")
        required_receipt_keys = {
            "schema", "source", "target", "source_size", "source_mtime_ns", "source_mode",
            "source_device", "source_inode", "target_size", "target_mtime_ns", "target_mode",
            "target_device", "target_inode", "sha256",
        }
        if set(receipt) != required_receipt_keys:
            raise ManifestError(f"snapshot receipt schema is not exact: {path}")
        if receipt.get("target") != str(absolute) or receipt.get("target_size") != len(encoded):
            raise ManifestError(f"snapshot receipt target mismatch: {path}")
        fields = {
            "target_mtime_ns": observed.st_mtime_ns,
            "target_mode": stat.S_IMODE(observed.st_mode),
            "target_device": observed.st_dev,
            "target_inode": observed.st_ino,
        }
        if any(receipt.get(key) != value for key, value in fields.items()):
            raise ManifestError(f"snapshot receipt metadata mismatch: {path}")
        source_path = Path(receipt.get("source", ""))
        if not source_path.is_absolute():
            raise ManifestError(f"snapshot receipt source is not absolute: {path}")
        with open_secure_directory(source_path.parent) as source_directory:
            source_stat = source_directory.stat(source_path.name)
        source_fields = {
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "source_mode": stat.S_IMODE(source_stat.st_mode),
            "source_device": source_stat.st_dev,
            "source_inode": source_stat.st_ino,
        }
        if any(receipt.get(key) != value for key, value in source_fields.items()):
            raise ManifestError(f"snapshot receipt source metadata mismatch: {path}")
        if receipt.get("sha256") != hashlib.sha256(encoded).hexdigest():
            raise ManifestError(f"snapshot receipt digest mismatch: {path}")
        value = json.loads(encoded)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ManifestError(f"invalid snapshot input: {path}") from exc
    if canonical_json_bytes(value) != encoded:
        raise ManifestError(f"snapshot input is not canonical JSON: {path}")
    return value


def _object(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ManifestError(f"{description} must be an object")
    return cast(dict[str, Any], value)


def _objects(value: Any, description: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ManifestError(f"{description} must be a list")
    return [_object(item, description) for item in value]


def _strings(value: Any, description: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ManifestError(f"{description} must contain non-empty strings")
    return tuple(value)


def _timestamp(value: str, description: str) -> datetime:
    try:
        observed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError) as exc:
        raise ManifestError(f"{description} is not an ISO timestamp") from exc
    if observed.tzinfo is None:
        raise ManifestError(f"{description} must include UTC")
    return observed.astimezone(UTC)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_digest(value: Any, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ManifestError(f"{description} must be a lowercase SHA-256")
    return value


def _require_oid(value: Any, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ManifestError(f"{description} must be a Git object id")
    return value


def sign_audit_exclusions(
    exclusions: AuditExclusions,
    *,
    signer: OperationalSigner,
) -> AuditExclusions:
    if exclusions.signature:
        raise ManifestError("audit exclusions must be unsigned before signing")
    if (
        tuple(sorted(set(exclusions.event_ids))) != exclusions.event_ids
        or tuple(sorted(set(exclusions.attempt_ids))) != exclusions.attempt_ids
    ):
        raise ManifestError("audit exclusion ids must be sorted and unique")
    envelope = signer.sign(
        domain=AUDIT_EXCLUSIONS_DOMAIN,
        payload=exclusions.signed_bytes(),
        key_id=exclusions.signer_key_id,
    )
    return replace(exclusions, signature=envelope.signature)


def sign_usage_proof(
    proof: UsageProof,
    *,
    signer: OperationalSigner,
) -> UsageProof:
    if proof.signature:
        raise ManifestError("usage proof must be unsigned before signing")
    envelope = signer.sign(
        domain=USAGE_PROOF_DOMAIN,
        payload=proof.signed_bytes(),
        key_id=proof.key_id,
    )
    return replace(proof, signature=envelope.signature)


def verify_usage_proof(
    proof: UsageProof,
    *,
    roster: VerificationRoster,
) -> None:
    """Verify one canonical two-Mac usage receipt independently of a manifest."""

    try:
        valid = (
            proof.schema == "memo.cutover_usage_proof.v1"
            and bool(proof.device_id)
            and bool(proof.key_id)
            and proof.roster_version == roster.version
            and bool(proof.query_version)
            and _require_oid(proof.snapshot_commit_oid, "usage proof source commit")
            and _require_digest(proof.raw_event_set_sha256, "usage proof event digest")
            and _require_digest(proof.exclusion_set_sha256, "usage proof exclusion digest")
            and _timestamp(proof.window_started_at, "usage proof window start")
            <= _timestamp(proof.window_ended_at, "usage proof window end")
            <= _timestamp(proof.issued_at, "usage proof issuance")
            <= _timestamp(proof.window_ended_at, "usage proof window end") + timedelta(hours=24)
            and roster.key(proof.key_id).device_id == proof.device_id
        )
    except (ManifestError, SignatureError):
        valid = False
    if not valid:
        raise ManifestError("usage proof provenance mismatch")
    try:
        OperationalVerifier().verify(
            domain=USAGE_PROOF_DOMAIN,
            payload=proof.signed_bytes(),
            envelope=proof.signature_envelope(),
            roster=roster,
        )
    except SignatureError as exc:
        raise ManifestError("usage proof signature is invalid") from exc


def verify_audit_exclusions(
    exclusions: AuditExclusions,
    *,
    roster: VerificationRoster,
) -> None:
    """Verify a standalone canonical audit-exclusion receipt."""

    _verify_exclusions(
        exclusions,
        roster=roster,
        window_started_at=exclusions.window_started_at,
        window_ended_at=exclusions.window_ended_at,
    )


def _verify_exclusions(
    exclusions: AuditExclusions,
    *,
    roster: VerificationRoster,
    window_started_at: str,
    window_ended_at: str,
) -> None:
    window_end = _timestamp(window_ended_at, "exclusion window end")
    issued_at = _timestamp(exclusions.issued_at, "exclusion issuance")
    try:
        signer_device_id = roster.key(exclusions.signer_key_id).device_id
    except SignatureError as exc:
        raise ManifestError("audit exclusion signer is unknown") from exc
    if (
        exclusions.schema != "memo.cutover_audit_exclusions.v1"
        or exclusions.window_started_at != window_started_at
        or exclusions.window_ended_at != window_ended_at
        or not window_end <= issued_at <= window_end + timedelta(hours=24)
        or tuple(sorted(set(exclusions.event_ids))) != exclusions.event_ids
        or tuple(sorted(set(exclusions.attempt_ids))) != exclusions.attempt_ids
        or signer_device_id != exclusions.signer_device_id
    ):
        raise ManifestError("audit exclusion signature/window/provenance is invalid")
    try:
        OperationalVerifier().verify(
            domain=AUDIT_EXCLUSIONS_DOMAIN,
            payload=exclusions.signed_bytes(),
            envelope=exclusions.signature_envelope(),
            roster=roster,
        )
    except SignatureError as exc:
        raise ManifestError("audit exclusion signature is invalid") from exc


def _usage_proof(value: Mapping[str, Any]) -> UsageProof:
    try:
        return UsageProof(
            schema=value["schema"],
            device_id=value["device_id"],
            key_id=value["key_id"],
            roster_version=value["roster_version"],
            query_version=value["query_version"],
            window_started_at=value["window_started_at"],
            window_ended_at=value["window_ended_at"],
            snapshot_commit_oid=value["snapshot_commit_oid"],
            raw_event_set_sha256=value["raw_event_set_sha256"],
            exclusion_set_sha256=value["exclusion_set_sha256"],
            issued_at=value["issued_at"],
            signature=value["signature"],
        )
    except (KeyError, TypeError) as exc:
        raise ManifestError("usage proof schema is incomplete") from exc


def audit_exclusions_from_dict(value: Mapping[str, Any]) -> AuditExclusions:
    """Decode a canonical audit-exclusion object without accepting JSON lists as tuples."""

    try:
        return AuditExclusions(
            schema=value["schema"],
            event_ids=_strings(value["event_ids"], "audit exclusion event ids"),
            attempt_ids=_strings(value["attempt_ids"], "audit exclusion attempt ids"),
            window_started_at=value["window_started_at"],
            window_ended_at=value["window_ended_at"],
            signer_device_id=value["signer_device_id"],
            signer_key_id=value["signer_key_id"],
            roster_version=value["roster_version"],
            issued_at=value["issued_at"],
            signature=value["signature"],
        )
    except (KeyError, TypeError) as exc:
        raise ManifestError("audit exclusion schema is incomplete") from exc


def _route(value: Mapping[str, Any]) -> OperationRoute:
    try:
        predicate = _object(value["predicate"], "route predicate")
        parameter_mapping = _object(value["parameter_mapping"], "route parameter mapping")
        defaults = _object(value["defaults"], "route defaults")
        result_mapping = _object(value["result_mapping"], "route result mapping")
        error_mapping = _object(value["error_mapping"], "route error mapping")
        route = OperationRoute(
            route_id=value["route_id"],
            predicate=predicate,
            memo_methods=_strings(value["memo_methods"], "route Memo methods"),
            memo_mcp=_strings(value["memo_mcp"], "route Memo MCP methods"),
            memo_cli=_strings(value["memo_cli"], "route Memo CLI commands"),
            parameter_mapping=cast(dict[str, str], parameter_mapping),
            defaults=defaults,
            result_mapping=cast(dict[str, str], result_mapping),
            error_mapping=cast(dict[str, str], error_mapping),
            transform_id=value["transform_id"],
            fixture_sha256=_strings(value["fixture_sha256"], "route fixtures"),
            atomic_group=value.get("atomic_group"),
            fixture_paths=(
                _strings(value["fixture_paths"], "route fixture paths")
                if "fixture_paths" in value
                else ()
            ),
        )
    except (KeyError, TypeError) as exc:
        raise ManifestError("operation route is incomplete") from exc
    if (
        not isinstance(route.route_id, str)
        or not route.route_id
        or not route.predicate
        or not route.parameter_mapping
        or not route.result_mapping
        or not route.error_mapping
        or not isinstance(route.transform_id, str)
        or not route.transform_id
        or any(
            _require_digest(digest, "route fixture") != digest for digest in route.fixture_sha256
        )
        or (route.fixture_paths and len(route.fixture_paths) != len(route.fixture_sha256))
        or len(set(route.fixture_paths)) != len(route.fixture_paths)
    ):
        raise ManifestError("operation route lacks executable authority data")
    if not _valid_predicate(route.predicate):
        raise ManifestError("operation route predicate is not a closed match expression")
    for mapping in (
        route.parameter_mapping,
        route.result_mapping,
        route.error_mapping,
    ):
        if any(
            not isinstance(key, str) or not key or not isinstance(item, str) or not item
            for key, item in mapping.items()
        ):
            raise ManifestError("operation route mapping is invalid")
    return route


def _valid_predicate(predicate: Mapping[str, object]) -> bool:
    for argument, raw_matcher in predicate.items():
        if not isinstance(argument, str) or not argument or not isinstance(raw_matcher, dict):
            return False
        matcher = cast(dict[object, object], raw_matcher)
        if len(matcher) != 1:
            return False
        operator, operand = next(iter(matcher.items()))
        if operator == "eq":
            try:
                canonical_json_bytes(operand)
            except (TypeError, ValueError):
                return False
        elif operator == "in":
            if not isinstance(operand, list) or not operand:
                return False
            try:
                canonical_values = [canonical_json_bytes(item) for item in operand]
            except (TypeError, ValueError):
                return False
            if len(canonical_values) != len(set(canonical_values)):
                return False
        elif operator == "present":
            if not isinstance(operand, bool):
                return False
        else:
            return False
    return True


def _mapping(value: Mapping[str, Any]) -> OperationMappingRow:
    try:
        disposition = value["disposition"]
        if disposition not in {"memo_native", "absorb", "internal", "delete"}:
            raise ManifestError("operation mapping disposition is invalid")
        return OperationMappingRow(
            source_operation=value["source_operation"],
            source_commit=_require_oid(value["source_commit"], "mapping source commit"),
            source_tests=_strings(value["source_tests"], "mapping source tests"),
            evidence_ids=tuple(sorted(_strings(value["evidence_ids"], "mapping evidence ids"))),
            capability=value["capability"],
            disposition=disposition,
            routes=tuple(_route(route) for route in _objects(value["routes"], "routes")),
            parity_tests=_strings(value["parity_tests"], "mapping parity tests"),
            deletion_proof=_strings(value["deletion_proof"], "mapping deletion proof"),
        )
    except (KeyError, TypeError) as exc:
        raise ManifestError("operation mapping is incomplete") from exc


def _slo(value: Mapping[str, Any]) -> SloBaseline:
    try:
        return SloBaseline(
            baseline_id=value["baseline_id"],
            source_commit=_require_oid(value["source_commit"], "SLO source commit"),
            workload_id=value["workload_id"],
            machine_class=value["machine_class"],
            window_started_at=value["window_started_at"],
            window_ended_at=value["window_ended_at"],
            sample_count=value["sample_count"],
            visibility_p50_ms=float(value["visibility_p50_ms"]),
            visibility_p95_ms=float(value["visibility_p95_ms"]),
            visibility_p99_ms=float(value["visibility_p99_ms"]),
            visibility_max_ms=float(value["visibility_max_ms"]),
            recovery_max_ms=float(value["recovery_max_ms"]),
            error_rate=float(value["error_rate"]),
            data_loss_count=value["data_loss_count"],
            duplicate_count=value["duplicate_count"],
            tolerance_ratio=float(value["tolerance_ratio"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError("SLO baseline is incomplete") from exc


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    description: str,
) -> None:
    if set(value) != expected:
        raise ManifestError(f"{description} fields are invalid")


def _capability(value: Mapping[str, Any]) -> CapabilityRow:
    _require_exact_fields(
        value,
        {
            "name",
            "sources",
            "consumers",
            "window_started_at",
            "window_ended_at",
            "observed_calls",
            "observed_daemon_events",
            "machines",
            "evidence_ids",
            "exclusion_counts",
            "evidence_complete",
            "source_operations",
            "operation_mappings",
            "slo_baseline_ids",
            "dependencies",
            "disposition",
            "memo_target",
            "parity_tests",
            "deletion_proof",
        },
        "capability row",
    )
    exclusions = _object(value["exclusion_counts"], "capability exclusion counts")
    if (
        value["disposition"] not in {"memo_native", "absorb", "internal", "delete"}
        or any(
            not isinstance(value[field], str)
            for field in (
                "name",
                "window_started_at",
                "window_ended_at",
                "memo_target",
            )
        )
        or any(
            isinstance(value[field], bool) or not isinstance(value[field], int)
            for field in ("observed_calls", "observed_daemon_events")
        )
        or not isinstance(value["evidence_complete"], bool)
        or any(
            not isinstance(key, str)
            or isinstance(count, bool)
            or not isinstance(count, int)
            for key, count in exclusions.items()
        )
    ):
        raise ManifestError("capability row values are invalid")
    mappings = _objects(value["operation_mappings"], "capability operation mappings")
    return CapabilityRow(
        name=value["name"],
        sources=_strings(value["sources"], "capability sources"),
        consumers=_strings(value["consumers"], "capability consumers"),
        window_started_at=value["window_started_at"],
        window_ended_at=value["window_ended_at"],
        observed_calls=value["observed_calls"],
        observed_daemon_events=value["observed_daemon_events"],
        machines=_strings(value["machines"], "capability machines"),
        evidence_ids=_strings(value["evidence_ids"], "capability evidence ids"),
        exclusion_counts=cast(dict[str, int], exclusions),
        evidence_complete=value["evidence_complete"],
        source_operations=_strings(
            value["source_operations"], "capability source operations"
        ),
        operation_mappings=tuple(
            _mapping_exact(mapping) for mapping in mappings
        ),
        slo_baseline_ids=_strings(
            value["slo_baseline_ids"], "capability SLO baseline ids"
        ),
        dependencies=_strings(value["dependencies"], "capability dependencies"),
        disposition=value["disposition"],
        memo_target=value["memo_target"],
        parity_tests=_strings(value["parity_tests"], "capability parity tests"),
        deletion_proof=_strings(
            value["deletion_proof"], "capability deletion proof"
        ),
    )


def _route_exact(value: Mapping[str, Any]) -> OperationRoute:
    _require_exact_fields(
        value,
        {
            "route_id",
            "predicate",
            "memo_methods",
            "memo_mcp",
            "memo_cli",
            "parameter_mapping",
            "defaults",
            "result_mapping",
            "error_mapping",
            "transform_id",
            "fixture_sha256",
            "atomic_group",
            "fixture_paths",
        },
        "operation route",
    )
    return _route(value)


def _mapping_exact(value: Mapping[str, Any]) -> OperationMappingRow:
    _require_exact_fields(
        value,
        {
            "source_operation",
            "source_commit",
            "source_tests",
            "evidence_ids",
            "capability",
            "disposition",
            "routes",
            "parity_tests",
            "deletion_proof",
        },
        "operation mapping",
    )
    routes = _objects(value["routes"], "operation mapping routes")
    normalized = dict(value)
    normalized["routes"] = [route.to_dict() for route in map(_route_exact, routes)]
    return _mapping(normalized)


def capability_manifest_from_dict(value: Mapping[str, Any]) -> CapabilityManifest:
    """Decode the exact typed capability-manifest authority."""

    _require_exact_fields(
        value,
        {
            "schema",
            "frozen_at",
            "window_started_at",
            "window_ended_at",
            "machine_ids",
            "source_receipt_sha256",
            "capabilities",
            "operation_mappings",
            "slo_baselines",
            "operation_map_sha256",
            "slo_baseline_sha256",
            "blockers",
            "frozen",
            "signer_device_id",
            "signer_key_id",
            "roster_version",
            "signature",
            "registry_authority_sha256",
            "fixture_authority_sha256",
        },
        "capability manifest",
    )
    source_receipts = _object(
        value["source_receipt_sha256"], "capability source receipts"
    )
    capabilities = _objects(value["capabilities"], "capabilities")
    mappings = _objects(value["operation_mappings"], "operation mappings")
    slos = _objects(value["slo_baselines"], "SLO baselines")
    string_fields = (
        "schema",
        "frozen_at",
        "window_started_at",
        "window_ended_at",
        "operation_map_sha256",
        "slo_baseline_sha256",
        "signer_device_id",
        "signer_key_id",
        "signature",
        "registry_authority_sha256",
        "fixture_authority_sha256",
    )
    if (
        any(not isinstance(value[field], str) for field in string_fields)
        or not isinstance(value["frozen"], bool)
        or isinstance(value["roster_version"], bool)
        or not isinstance(value["roster_version"], int)
        or any(
            not isinstance(key, str) or not isinstance(digest, str)
            for key, digest in source_receipts.items()
        )
    ):
        raise ManifestError("capability manifest values are invalid")
    for slo in slos:
        _require_exact_fields(
            slo,
            {
                "baseline_id",
                "source_commit",
                "workload_id",
                "machine_class",
                "window_started_at",
                "window_ended_at",
                "sample_count",
                "visibility_p50_ms",
                "visibility_p95_ms",
                "visibility_p99_ms",
                "visibility_max_ms",
                "recovery_max_ms",
                "error_rate",
                "data_loss_count",
                "duplicate_count",
                "tolerance_ratio",
            },
            "SLO baseline",
        )
    try:
        return CapabilityManifest(
            schema=cast(Any, value["schema"]),
            frozen_at=value["frozen_at"],
            window_started_at=value["window_started_at"],
            window_ended_at=value["window_ended_at"],
            machine_ids=_strings(value["machine_ids"], "capability machine ids"),
            source_receipt_sha256=cast(dict[str, str], source_receipts),
            capabilities=tuple(_capability(row) for row in capabilities),
            operation_mappings=tuple(_mapping_exact(row) for row in mappings),
            slo_baselines=tuple(_slo(row) for row in slos),
            operation_map_sha256=value["operation_map_sha256"],
            slo_baseline_sha256=value["slo_baseline_sha256"],
            blockers=_strings(value["blockers"], "capability blockers"),
            frozen=value["frozen"],
            signer_device_id=value["signer_device_id"],
            signer_key_id=value["signer_key_id"],
            roster_version=value["roster_version"],
            signature=value["signature"],
            registry_authority_sha256=value["registry_authority_sha256"],
            fixture_authority_sha256=value["fixture_authority_sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError("capability manifest schema is incomplete") from exc


def _predicate_overlap(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    """Whether two closed predicate expressions can match the same request."""

    for argument in set(left) & set(right):
        left_operator, left_value = next(iter(cast(dict[str, object], left[argument]).items()))
        right_operator, right_value = next(iter(cast(dict[str, object], right[argument]).items()))
        if left_operator == right_operator == "present":
            if left_value != right_value:
                return False
            continue
        if left_operator == "present":
            if left_value is False:
                return False
            continue
        if right_operator == "present":
            if right_value is False:
                return False
            continue
        left_values = (
            {canonical_json_bytes(left_value)}
            if left_operator == "eq"
            else {canonical_json_bytes(value) for value in cast(list[object], left_value)}
        )
        right_values = (
            {canonical_json_bytes(right_value)}
            if right_operator == "eq"
            else {canonical_json_bytes(value) for value in cast(list[object], right_value)}
        )
        if left_values.isdisjoint(right_values):
            return False
    return True


def _routes_are_disjoint(routes: tuple[OperationRoute, ...]) -> bool:
    return not any(
        _predicate_overlap(left.predicate, right.predicate)
        for index, left in enumerate(routes)
        for right in routes[index + 1 :]
    )


def _receipt_digest(root: Path) -> str:
    receipt = _object(
        _load_json(root / "snapshot-receipt.json"),
        "snapshot receipt",
    )
    return _require_digest(receipt.get("sha256"), "snapshot receipt digest")


def _verify_proofs(
    *,
    usage: dict[str, Any],
    events_by_device: dict[str, Any],
    machine_ids: tuple[str, ...],
    exclusions: AuditExclusions,
    source_commit: str,
    roster: VerificationRoster,
) -> None:
    proof_values = _objects(usage.get("proofs"), "usage proofs")
    proofs = [_usage_proof(value) for value in proof_values]
    if tuple(sorted(proof.device_id for proof in proofs)) != machine_ids:
        raise ManifestError("usage proof set does not exactly cover both machines")
    exclusion_digest = _sha256(
        canonical_json_bytes(
            {
                "event_ids": list(exclusions.event_ids),
                "attempt_ids": list(exclusions.attempt_ids),
            }
        )
    )
    for proof in proofs:
        raw_events = events_by_device.get(proof.device_id)
        if not isinstance(raw_events, list):
            raise ManifestError("usage proof has no raw event set")
        if (
            proof.schema != "memo.cutover_usage_proof.v1"
            or not proof.query_version
            or proof.window_started_at != usage["window_started_at"]
            or proof.window_ended_at != usage["window_ended_at"]
            or proof.snapshot_commit_oid != source_commit
            or proof.raw_event_set_sha256 != _sha256(canonical_json_bytes(raw_events))
            or proof.exclusion_set_sha256 != exclusion_digest
            or not (
                _timestamp(proof.window_ended_at, "usage proof window end")
                <= _timestamp(proof.issued_at, "usage proof issuance")
                <= _timestamp(proof.window_ended_at, "usage proof window end") + timedelta(hours=24)
            )
        ):
            raise ManifestError("usage proof provenance mismatch")
        verify_usage_proof(proof, roster=roster)


def build_capability_manifest(
    memo_snapshot: Path,
    memflow_snapshot: Path,
    usage_snapshot: Path,
    audit_exclusions: AuditExclusions,
    *,
    signer: OperationalSigner,
    signer_key_id: str,
    roster: VerificationRoster,
    source_operation_records: tuple[SynapseOperation, ...] = (),
    fixture_root: Path | None = None,
    transform_registry: object | None = None,
) -> CapabilityManifest:
    """Join pinned source, mappings, and signed 90-day telemetry fail-closed."""

    source = _object(
        _load_json(memflow_snapshot / "source.json"),
        "Memflow source snapshot",
    )
    source_commit = _require_oid(source.get("source_commit"), "Memflow source commit")
    source_operations = _objects(source.get("operations", []), "source operations")
    if source_operation_records:
        source_operations = [
            {
                "source_operation": row.source_operation,
                "source_tests": list(row.fixture_paths),
                "capability": row.source_operation,
                "latency_sensitive": bool(row.daemon_routes),
            }
            for row in source_operation_records
        ]
    source_by_name: dict[str, dict[str, Any]] = {}
    for operation in source_operations:
        name = operation.get("source_operation")
        if not isinstance(name, str) or not name or name in source_by_name:
            raise ManifestError("source operations must be named and unique")
        source_tests = _strings(operation.get("source_tests"), "source tests")
        if tuple(sorted(set(source_tests))) != source_tests:
            raise ManifestError("source tests must be sorted and unique")
        source_by_name[name] = operation

    usage = _object(_load_json(usage_snapshot / "usage.json"), "usage snapshot")
    frozen_at = usage.get("frozen_at")
    window_started_at = usage.get("window_started_at")
    window_ended_at = usage.get("window_ended_at")
    if not all(isinstance(value, str) for value in (frozen_at, window_started_at, window_ended_at)):
        raise ManifestError("usage window is incomplete")
    frozen_at = cast(str, frozen_at)
    window_started_at = cast(str, window_started_at)
    window_ended_at = cast(str, window_ended_at)
    start = _timestamp(window_started_at, "window start")
    end = _timestamp(window_ended_at, "window end")
    if _timestamp(frozen_at, "frozen at") != end or end - start != timedelta(days=90):
        raise ManifestError("usage window must be the exact inclusive 90-day window")
    machine_ids = tuple(sorted(_strings(usage.get("machine_ids"), "machine ids")))
    if len(machine_ids) != 2 or len(set(machine_ids)) != 2:
        raise ManifestError("usage evidence must cover exactly two distinct Macs")
    events_by_device = _object(usage.get("events"), "usage events")

    _verify_exclusions(
        audit_exclusions,
        roster=roster,
        window_started_at=window_started_at,
        window_ended_at=window_ended_at,
    )
    _verify_proofs(
        usage=usage,
        events_by_device=events_by_device,
        machine_ids=machine_ids,
        exclusions=audit_exclusions,
        source_commit=source_commit,
        roster=roster,
    )

    mapping_values = _objects(
        _load_json(memo_snapshot / "mapping-candidates.json"),
        "mapping candidates",
    )
    mappings = tuple(
        sorted((_mapping(value) for value in mapping_values), key=lambda row: row.source_operation)
    )
    mapping_names = [row.source_operation for row in mappings]
    if len(mapping_names) != len(set(mapping_names)) or set(mapping_names) != set(source_by_name):
        raise ManifestError("every source operation must have exactly one disposition")
    routes = tuple(r for m in mappings for r in m.routes)
    if routes:
        if fixture_root is None or not isinstance(transform_registry, FrozenTransformRegistry):
            raise ManifestError("non-empty routes require fixture_root and FrozenTransformRegistry")
        _verify_fixture_bindings(mappings, fixture_root)
        try:
            verify_route_fixtures(routes, transform_registry, fixture_root)
        except Exception as exc:
            raise ManifestError("route transform fixture verification failed") from exc

    blockers: set[str] = set()
    receipts = usage.get("source_receipts_v2")
    if not isinstance(receipts, list) or len(receipts) != len(machine_ids):
        raise ManifestError("signed hourly source receipts v2 are required")
    seen_devices: set[str] = set()
    expected_hours = int((end - start).total_seconds() // 3600)
    for receipt in receipts:
        if not isinstance(receipt, dict) or receipt.get("schema") != "memo.cutover_source_receipt.v2":
            raise ManifestError("invalid source receipt schema")
        device = receipt.get("device_id")
        buckets = receipt.get("hourly_buckets")
        if device not in machine_ids or device in seen_devices or not isinstance(buckets, list):
            raise ManifestError("source receipt device/buckets invalid")
        required = ("key_id", "roster_id", "query", "extractor_version", "snapshot_commit",
                    "raw_event_set_sha256", "window_start", "window_end", "issued_at",
                    "collected_at", "cursor", "signature")
        if any(not receipt.get(field) for field in required):
            raise ManifestError("source receipt provenance/signature incomplete")
        if len(buckets) != expected_hours or not receipt.get("extraction_complete"):
            raise ManifestError("source receipt coverage incomplete")
        previous = None
        for bucket in buckets:
            if not isinstance(bucket, dict) or any(key not in bucket for key in ("start", "end", "count", "digest")):
                raise ManifestError("source receipt hourly bucket malformed")
            if previous is not None and bucket["start"] <= previous:
                raise ManifestError("source receipt buckets out of order")
            previous = bucket["start"]
        try:
            env = SignatureEnvelope(algorithm=receipt["algorithm"], key_id=receipt["key_id"],
                                    roster_version=int(receipt["roster_version"]), signature=receipt["signature"])
            model = SourceReceiptV2(device_id=device, key_id=receipt["key_id"], roster_id=receipt["roster_id"],
                query=receipt["query"], extractor_version=receipt["extractor_version"], snapshot_commit=receipt["snapshot_commit"],
                raw_event_set_sha256=receipt["raw_event_set_sha256"], window_start=receipt["window_start"], window_end=receipt["window_end"],
                issued_at=receipt["issued_at"], collected_at=receipt["collected_at"], cursor=receipt["cursor"],
                extraction_complete=receipt["extraction_complete"], hourly_buckets=tuple(SourceBucket(**b) for b in buckets),
                frozen_at=receipt.get("frozen_at"), signature=env)
            verify_source_receipt(model, roster=roster, frozen_at=frozen_at,
                                  window_start=window_started_at, window_end=window_ended_at,
                                  authoritative_events=_objects(events_by_device.get(device), f"events for {device}"))
        except Exception as exc:
            raise ManifestError("source receipt signature verification failed") from exc
        seen_devices.add(device)
    if seen_devices != set(machine_ids):
        raise ManifestError("source receipts do not cover all devices")

    excluded_events = set(audit_exclusions.event_ids)
    excluded_attempts = set(audit_exclusions.attempt_ids)
    active_events: list[dict[str, Any]] = []
    excluded_event_count = 0
    excluded_attempt_count = 0
    for device_id in machine_ids:
        values = _objects(events_by_device.get(device_id), f"events for {device_id}")
        for event in values:
            if event.get("device_id") != device_id:
                raise ManifestError("event device does not match signed event set")
            occurred_at = event.get("occurred_at")
            if not isinstance(occurred_at, str):
                raise ManifestError("usage event timestamp is missing")
            observed_at = _timestamp(occurred_at, "usage event timestamp")
            if observed_at < start or observed_at > end:
                blockers.add("usage:event-outside-window")
            if event.get("event_id") in excluded_events:
                excluded_event_count += 1
                continue
            if event.get("attempt_id") in excluded_attempts:
                excluded_attempt_count += 1
                continue
            active_events.append(event)

    for event in active_events:
        source_operation = event.get("source_operation")
        if not isinstance(source_operation, str) or source_operation not in source_by_name:
            blockers.add(f"unknown:{source_operation}")

    events_by_operation: dict[str, list[dict[str, Any]]] = {
        operation: [] for operation in source_by_name
    }
    for event in active_events:
        observed_operation = event.get("source_operation")
        if isinstance(observed_operation, str) and observed_operation in events_by_operation:
            events_by_operation[observed_operation].append(event)

    slo_values = _objects(usage.get("slo_baselines"), "SLO baselines")
    slos = tuple(sorted((_slo(value) for value in slo_values), key=lambda row: row.baseline_id))
    if len({row.baseline_id for row in slos}) != len(slos):
        raise ManifestError("SLO baseline ids must be unique")
    slo_by_workload: dict[str, list[SloBaseline]] = {}
    for slo in slos:
        if slo.workload_id not in source_by_name:
            blockers.add(f"slo:{slo.baseline_id}:unknown-workload")
        slo_by_workload.setdefault(slo.workload_id, []).append(slo)
        if slo.source_commit != source_commit:
            blockers.add(f"slo:{slo.baseline_id}:source-commit")
        if slo.window_started_at != window_started_at or slo.window_ended_at != window_ended_at:
            blockers.add(f"slo:{slo.baseline_id}:window")
        if slo.sample_count < 100:
            blockers.add(f"slo:{slo.baseline_id}:sample-count")
        if slo.data_loss_count != 0:
            blockers.add(f"slo:{slo.baseline_id}:data-loss")
        if slo.duplicate_count != 0:
            blockers.add(f"slo:{slo.baseline_id}:duplicates")
        if not (
            0
            <= slo.visibility_p50_ms
            <= slo.visibility_p95_ms
            <= slo.visibility_p99_ms
            <= slo.visibility_max_ms
        ):
            blockers.add(f"slo:{slo.baseline_id}:distribution")
        if not 0 <= slo.error_rate <= 1 or slo.recovery_max_ms < 0 or slo.tolerance_ratio < 1:
            blockers.add(f"slo:{slo.baseline_id}:bounds")

    source_rows: list[CapabilityRow] = []
    for mapping in mappings:
        source_record = source_by_name[mapping.source_operation]
        if (
            mapping.source_commit != source_commit
            or tuple(source_record.get("source_tests", ())) != mapping.source_tests
        ):
            blockers.add(f"{mapping.capability}:source-provenance")
        operation_events = events_by_operation[mapping.source_operation]
        evidence_ids = tuple(
            sorted(
                {
                    evidence
                    for event in operation_events
                    if isinstance((evidence := event.get("evidence_id")), str) and evidence
                }
            )
        )
        if any(event.get("ambiguous") is True for event in operation_events):
            blockers.add(f"{mapping.capability}:ambiguous-traffic")
        if mapping.disposition == "delete":
            required_deletion = {"complete-zero-use-90d", "source-test-retired"}
            if (
                operation_events
                or mapping.evidence_ids
                or not required_deletion.issubset(mapping.deletion_proof)
                or mapping.routes
                or mapping.parity_tests
            ):
                blockers.add(f"{mapping.capability}:deletion-proof")
        else:
            if (
                not mapping.routes
                or not mapping.parity_tests
                or not _routes_are_disjoint(mapping.routes)
            ):
                blockers.add(f"{mapping.capability}:operation-map")
            if evidence_ids != mapping.evidence_ids:
                blockers.add(f"{mapping.capability}:evidence-mismatch")
        latency_sensitive = source_record.get("latency_sensitive") is True
        related_slos = tuple(slo_by_workload.get(mapping.source_operation, ()))
        if latency_sensitive and not related_slos:
            blockers.add(f"{mapping.capability}:missing-slo")
        calls = sum(event.get("kind") == "call" for event in operation_events)
        daemon_events = sum(event.get("kind") == "daemon" for event in operation_events)
        machines = tuple(
            sorted(
                {
                    device
                    for event in operation_events
                    if isinstance((device := event.get("device_id")), str)
                }
            )
        )
        source_rows.append(
            CapabilityRow(
                name=mapping.capability,
                sources=(mapping.source_operation,),
                consumers=machines,
                window_started_at=window_started_at,
                window_ended_at=window_ended_at,
                observed_calls=calls,
                observed_daemon_events=daemon_events,
                machines=machines,
                evidence_ids=evidence_ids,
                exclusion_counts={
                    "event_ids": excluded_event_count,
                    "attempt_ids": excluded_attempt_count,
                },
                evidence_complete=True,
                source_operations=(mapping.source_operation,),
                operation_mappings=(mapping,),
                slo_baseline_ids=tuple(row.baseline_id for row in related_slos),
                dependencies=(),
                disposition=mapping.disposition,
                memo_target=(
                    ""
                    if mapping.disposition == "delete"
                    else ",".join(
                        sorted(
                            {method for route in mapping.routes for method in route.memo_methods}
                        )
                    )
                ),
                parity_tests=mapping.parity_tests,
                deletion_proof=mapping.deletion_proof,
            )
        )

    rows: list[CapabilityRow] = []
    for capability in sorted({row.name for row in source_rows}):
        grouped = [row for row in source_rows if row.name == capability]
        dispositions = {row.disposition for row in grouped}
        if len(dispositions) != 1:
            blockers.add(f"{capability}:mixed-disposition")
        precedence = ("absorb", "memo_native", "internal", "delete")
        disposition = next(candidate for candidate in precedence if candidate in dispositions)
        rows.append(
            CapabilityRow(
                name=capability,
                sources=tuple(sorted({source for row in grouped for source in row.sources})),
                consumers=tuple(
                    sorted({consumer for row in grouped for consumer in row.consumers})
                ),
                window_started_at=window_started_at,
                window_ended_at=window_ended_at,
                observed_calls=sum(row.observed_calls for row in grouped),
                observed_daemon_events=sum(row.observed_daemon_events for row in grouped),
                machines=tuple(sorted({machine for row in grouped for machine in row.machines})),
                evidence_ids=tuple(
                    sorted({evidence for row in grouped for evidence in row.evidence_ids})
                ),
                exclusion_counts={
                    "event_ids": excluded_event_count,
                    "attempt_ids": excluded_attempt_count,
                },
                evidence_complete=all(row.evidence_complete for row in grouped),
                source_operations=tuple(
                    sorted({operation for row in grouped for operation in row.source_operations})
                ),
                operation_mappings=tuple(
                    sorted(
                        (mapping for row in grouped for mapping in row.operation_mappings),
                        key=lambda mapping: mapping.source_operation,
                    )
                ),
                slo_baseline_ids=tuple(
                    sorted({baseline for row in grouped for baseline in row.slo_baseline_ids})
                ),
                dependencies=tuple(
                    sorted({dependency for row in grouped for dependency in row.dependencies})
                ),
                disposition=disposition,
                memo_target=",".join(
                    sorted(
                        {
                            target
                            for row in grouped
                            for target in row.memo_target.split(",")
                            if target
                        }
                    )
                ),
                parity_tests=tuple(sorted({test for row in grouped for test in row.parity_tests})),
                deletion_proof=tuple(
                    sorted({proof for row in grouped for proof in row.deletion_proof})
                ),
            )
        )

    registry_authority_sha256 = (
        transform_registry.digest()
        if routes and isinstance(transform_registry, FrozenTransformRegistry)
        else ""
    )
    fixture_authority_sha256 = _sha256(canonical_json_bytes(sorted({p: d for m in mappings for r in m.routes for p, d in zip(r.fixture_paths, r.fixture_sha256, strict=True)}.items())))
    operation_map = canonical_json_bytes({"mappings": [row.to_dict() for row in mappings], "registry_authority_sha256": registry_authority_sha256, "fixture_authority_sha256": fixture_authority_sha256})
    slo_baseline = canonical_json_bytes([row.to_dict() for row in slos])
    signer_key = roster.key(signer_key_id)
    sorted_blockers = tuple(sorted(blockers))
    unsigned = CapabilityManifest(
        schema="memo.cutover_capability_manifest.v1",
        frozen_at=frozen_at,
        window_started_at=window_started_at,
        window_ended_at=window_ended_at,
        machine_ids=machine_ids,
        source_receipt_sha256={
            "memo": _receipt_digest(memo_snapshot),
            "memflow": _receipt_digest(memflow_snapshot),
            "usage": _receipt_digest(usage_snapshot),
        },
        capabilities=tuple(sorted(rows, key=lambda row: row.name)),
        operation_mappings=mappings,
        slo_baselines=slos,
        operation_map_sha256=_sha256(operation_map),
        slo_baseline_sha256=_sha256(slo_baseline),
        blockers=sorted_blockers,
        frozen=not sorted_blockers,
        signer_device_id=signer_key.device_id,
        signer_key_id=signer_key_id,
        roster_version=roster.version,
        signature="",
        registry_authority_sha256=registry_authority_sha256,
        fixture_authority_sha256=fixture_authority_sha256,
    )
    if not unsigned.frozen:
        return unsigned
    envelope = signer.sign(
        domain=CAPABILITY_MANIFEST_DOMAIN,
        payload=unsigned.signed_bytes(),
        key_id=signer_key_id,
    )
    signed = replace(unsigned, signature=envelope.signature)
    verify_capability_manifest(signed, roster=roster)
    return signed


def _verify_fixture_bindings(mappings: tuple[OperationMappingRow, ...], root: Path) -> None:
    """Bind each Synapse route digest to a regular fixture in the pinned tree."""

    absolute_root = root.resolve(strict=True)
    for mapping in mappings:
        for route in mapping.routes:
            if not route.fixture_paths:
                raise ManifestError("Synapse route lacks fixture path authority")
            for relative, digest in zip(route.fixture_paths, route.fixture_sha256, strict=True):
                candidate = absolute_root / relative
                try:
                    resolved = candidate.resolve(strict=True)
                except OSError as exc:
                    raise ManifestError("Synapse route fixture is unavailable") from exc
                if absolute_root not in (resolved, *resolved.parents) or candidate.is_symlink():
                    raise ManifestError("Synapse route fixture is unsafe")
                if _sha256(resolved.read_bytes()) != digest:
                    raise ManifestError("Synapse route fixture digest mismatch")


def build_synapse_capability_manifest(
    snapshot: Path,
    usage_proofs: tuple[Path, ...],
    exclusions: tuple[Path, ...],
    *,
    memo_snapshot: Path,
    usage_snapshot: Path,
    audit_exclusions: AuditExclusions,
    signer: OperationalSigner,
    signer_key_id: str,
    roster: VerificationRoster,
    transform_registry: FrozenTransformRegistry,
) -> CapabilityManifest:
    """Build capability authority over the canonical, pinned Synapse catalog.

    ``usage_proofs`` and ``exclusions`` are immutable input receipts.  They
    must byte-match the evidence embedded in ``usage_snapshot`` and the signed
    exclusion record, preventing a caller from silently substituting telemetry
    after inspection.
    """

    try:
        operations = discover_synapse_operations(snapshot)
    except SynapseCatalogError as exc:
        raise ManifestError(str(exc)) from exc
    usage = _object(_load_json(usage_snapshot / "usage.json"), "usage snapshot")
    proof_bytes = {
        canonical_json_bytes(value)
        for value in _objects(usage.get("proofs"), "usage proofs")
    }
    supplied_proofs = {path.read_bytes() for path in usage_proofs}
    if len(supplied_proofs) != 2 or supplied_proofs != proof_bytes:
        raise ManifestError("Synapse usage proofs do not exactly match the two-Mac snapshot")
    supplied_exclusions = {path.read_bytes() for path in exclusions}
    if supplied_exclusions != {canonical_json_bytes(audit_exclusions.to_dict())}:
        raise ManifestError("Synapse exclusion receipts do not match signed authority")
    return build_capability_manifest(
        memo_snapshot,
        snapshot,
        usage_snapshot,
        audit_exclusions,
        signer=signer,
        signer_key_id=signer_key_id,
        roster=roster,
        source_operation_records=operations,
        fixture_root=snapshot,
        transform_registry=transform_registry,
    )


def verify_capability_manifest(
    manifest: CapabilityManifest,
    *,
    roster: VerificationRoster,
) -> None:
    """Verify canonical digests and the authority signature."""

    if (
        manifest.schema != "memo.cutover_capability_manifest.v1"
        or not manifest.frozen
        or manifest.blockers
        or not manifest.signature
    ):
        raise ManifestError("capability manifest is not frozen and signed")
    if manifest.roster_version != roster.version:
        raise ManifestError("capability manifest roster authority is invalid")
    if (
        _sha256(manifest.operation_map_bytes()) != manifest.operation_map_sha256
        or _sha256(manifest.slo_baseline_bytes()) != manifest.slo_baseline_sha256
    ):
        raise ManifestError("capability manifest canonical digest mismatch")
    try:
        key = roster.key(manifest.signer_key_id)
        if key.device_id != manifest.signer_device_id:
            raise SignatureError("capability manifest signer does not own roster key")
        OperationalVerifier().verify(
            domain=CAPABILITY_MANIFEST_DOMAIN,
            payload=manifest.signed_bytes(),
            envelope=manifest.signature_envelope(),
            roster=roster,
        )
    except (KeyError, SignatureError) as exc:
        raise ManifestError("capability manifest signature is invalid") from exc


__all__ = [
    "AUDIT_EXCLUSIONS_DOMAIN",
    "CAPABILITY_MANIFEST_DOMAIN",
    "USAGE_PROOF_DOMAIN",
    "ManifestError",
    "audit_exclusions_from_dict",
    "build_capability_manifest",
    "build_synapse_capability_manifest",
    "capability_manifest_from_dict",
    "sign_audit_exclusions",
    "sign_usage_proof",
    "verify_audit_exclusions",
    "verify_capability_manifest",
    "verify_usage_proof",
]
