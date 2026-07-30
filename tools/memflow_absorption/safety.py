"""Filesystem authority checks for cutover attempts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from memo.atomic_io import open_secure_directory
from memo.errors import SignatureError
from memo.operational_event import canonical_json_bytes
from memo.operational_roster import VerificationRoster
from memo.operational_signing import OperationalSigner, OperationalVerifier
from tools.memflow_absorption.control_record import (
    CutoverSafetyError,
    prepare_synapse_retirement,
)
from tools.memflow_absorption.inventory import (
    InventoryError,
    verify_consumer_inventory,
    verify_synapse_retirement_manifest,
)
from tools.memflow_absorption.schemas import (
    ConsumerInventory,
    CutoverState,
    IndependenceObservation,
    IndependenceReceipt,
    IndependenceScanReceipt,
    SynapseRetirementManifest,
    SynapseRetirementState,
    VerifiedControlRecord,
)

ATTEMPT_SENTINEL = ".memo-cutover-attempt.json"
_ATTEMPT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CLEANUP_DIGEST_KEYS = frozenset(
    {
        "bounded_data_receipt",
        "consumer_replacement_receipt",
        "control_record",
        "independence_receipt",
        "retirement_manifest",
    }
)
_INDEPENDENCE_SURFACES = (
    "launchagent",
    "mcp_gateway_route",
    "port",
    "process",
    "shell_config_path",
    "state_root",
)
SYNAPSE_INDEPENDENCE_SCAN_DOMAIN = "memo.cutover.synapse_independence_scan.v1"
SYNAPSE_INDEPENDENCE_RECEIPT_DOMAIN = (
    "memo.cutover.synapse_independence_receipt.v1"
)


class SafetyError(RuntimeError):
    """A requested cutover path is outside the operator's authority."""


def _contains_unresolved(value: str) -> bool:
    return value.startswith("~") or "$" in value or "%" in value


def _reject_symlink_components(path: Path) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            observed = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(observed.st_mode):
            raise SafetyError(f"cutover path contains symlink component: {current}")


def _is_repository(path: Path) -> bool:
    return (path / ".git").exists() or (
        (path / "pyproject.toml").is_file() and (path / "src").is_dir()
    )


def assert_safe_attempt_root(
    path: Path,
    attempt_id: str,
    *,
    require_sentinel: bool = False,
    manifest_sha256: str | None = None,
) -> Path:
    """Validate an exact ``memo/cutover/<attempt-id>`` authority root."""

    raw = os.fspath(path)
    if _contains_unresolved(raw):
        raise SafetyError("cutover path contains unresolved shell syntax")
    if not _ATTEMPT_RE.fullmatch(attempt_id):
        raise SafetyError("attempt id is unsafe")
    candidate = Path(os.path.abspath(raw))
    if candidate == Path(candidate.anchor) or candidate == Path.home():
        raise SafetyError("cutover target is too broad")
    if _is_repository(candidate):
        raise SafetyError("cutover target is a repository")
    if len(candidate.parts) < 4 or candidate.parts[-3:] != (
        "memo",
        "cutover",
        attempt_id,
    ):
        raise SafetyError("attempt root must end with memo/cutover/<exact-attempt-id>")
    _reject_symlink_components(candidate)
    if require_sentinel:
        if manifest_sha256 is None or not _SHA256_RE.fullmatch(manifest_sha256):
            raise SafetyError("exact lowercase manifest SHA-256 is required")
        sentinel = candidate / ATTEMPT_SENTINEL
        try:
            with open_secure_directory(candidate) as directory:
                encoded = directory.read_bytes(ATTEMPT_SENTINEL)
            payload = json.loads(encoded)
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
            raise SafetyError("cutover attempt sentinel is missing or invalid") from exc
        expected = {
            "schema": "memo.cutover_attempt.v1",
            "attempt_id": attempt_id,
            "manifest_sha256": manifest_sha256,
        }
        if payload != expected or canonical_json_bytes(payload) != encoded:
            raise SafetyError("cutover attempt sentinel or manifest does not match authority")
        if sentinel.is_symlink():
            raise SafetyError("cutover attempt sentinel is a symlink")
    return candidate


def initialize_attempt_root(path: Path, attempt_id: str, manifest_sha256: str) -> Path:
    """Create a new attempt root and immutable authority sentinel."""

    candidate = assert_safe_attempt_root(path, attempt_id)
    if not _SHA256_RE.fullmatch(manifest_sha256):
        raise SafetyError("exact lowercase manifest SHA-256 is required")
    payload = canonical_json_bytes(
        {
            "schema": "memo.cutover_attempt.v1",
            "attempt_id": attempt_id,
            "manifest_sha256": manifest_sha256,
        }
    )
    try:
        with open_secure_directory(candidate, create=True) as directory:
            directory.create_bytes_exclusive(ATTEMPT_SENTINEL, payload, mode=0o400)
    except (FileExistsError, OSError, ValueError) as exc:
        raise SafetyError("could not initialize an exclusive cutover attempt") from exc
    return assert_safe_attempt_root(
        candidate,
        attempt_id,
        require_sentinel=True,
        manifest_sha256=manifest_sha256,
    )


def resolve_under_attempt(
    root: Path,
    relative: str,
    attempt_id: str,
    manifest_sha256: str,
) -> Path:
    """Resolve a lexical child without allowing escape or symlink traversal."""

    authority = assert_safe_attempt_root(
        root,
        attempt_id,
        require_sentinel=True,
        manifest_sha256=manifest_sha256,
    )
    requested = Path(relative)
    if (
        requested.is_absolute()
        or _contains_unresolved(relative)
        or any(part in {"", ".", ".."} for part in requested.parts)
    ):
        raise SafetyError("requested path escapes cutover attempt authority")
    target = authority.joinpath(*requested.parts)
    _reject_symlink_components(target)
    try:
        target.relative_to(authority)
    except ValueError as exc:
        raise SafetyError("requested path escapes cutover attempt authority") from exc
    return target


def assert_retirement_cleanup_authority(
    control: VerifiedControlRecord,
    *,
    expected_digests: Mapping[str, str],
    observed_digests: Mapping[str, str],
    cleanup_paths: tuple[Path, ...] | list[Path],
) -> None:
    """Reject cleanup until all exact authority and runtime blockers are closed.

    This validator intentionally has no success path today.  The signed
    control record binds the consumer plan, final retirement manifest, and
    independence receipt, but the repository has neither a signed deletion
    plan for exact filesystem targets nor evidence that the retirement fence
    is wired at the production Synapse runtime boundary.  Validating the
    available evidence before reporting that blocker makes the future operator
    command fail for the most specific authority defect while remaining
    incapable of deleting anything.
    """

    if (
        control.state is not CutoverState.VERIFIED
        or control.synapse_state is not SynapseRetirementState.VERIFIED
        or control.retirement_epoch < 1
    ):
        raise CutoverSafetyError(
            "retirement cleanup requires a VERIFIED control record"
        )
    if (
        set(expected_digests) != _CLEANUP_DIGEST_KEYS
        or set(observed_digests) != _CLEANUP_DIGEST_KEYS
        or any(
            not isinstance(value, str) or not _SHA256_RE.fullmatch(value)
            for value in (*expected_digests.values(), *observed_digests.values())
        )
    ):
        raise CutoverSafetyError(
            "retirement cleanup requires all exact artifact digests"
        )
    mismatches = sorted(
        key
        for key in _CLEANUP_DIGEST_KEYS
        if expected_digests[key] != observed_digests[key]
    )
    if mismatches:
        raise CutoverSafetyError(
            "retirement cleanup artifact digest mismatch: " + ",".join(mismatches)
        )
    if (
        expected_digests["retirement_manifest"]
        != control.synapse_manifest_sha256
        or expected_digests["consumer_replacement_receipt"]
        != control.consumer_plan_sha256
        or expected_digests["independence_receipt"]
        != control.independence_receipt_sha256
    ):
        raise CutoverSafetyError(
            "retirement cleanup digest mismatch with VERIFIED control authority"
        )
    if not cleanup_paths:
        raise CutoverSafetyError("retirement cleanup requires exact cleanup paths")
    normalized: list[Path] = []
    for path in cleanup_paths:
        raw = os.fspath(path)
        if _contains_unresolved(raw) or not Path(raw).is_absolute():
            raise CutoverSafetyError(
                f"retirement cleanup path is unresolved or relative: {raw}"
            )
        candidate = Path(os.path.abspath(raw))
        if (
            candidate == Path(candidate.anchor)
            or candidate == Path.home()
            or _is_repository(candidate)
        ):
            raise CutoverSafetyError(
                f"retirement cleanup path is broad or a repository: {candidate}"
            )
        try:
            _reject_symlink_components(candidate)
        except SafetyError as exc:
            raise CutoverSafetyError(
                f"retirement cleanup path is unsafe: {candidate}"
            ) from exc
        normalized.append(candidate)
    if len(normalized) != len(set(normalized)):
        raise CutoverSafetyError("retirement cleanup paths are duplicated")
    for index, path in enumerate(normalized):
        for other in normalized[index + 1 :]:
            if path in other.parents or other in path.parents:
                raise CutoverSafetyError("retirement cleanup paths overlap")
    raise CutoverSafetyError(
        "retirement cleanup is blocked: production runtime gate evidence "
        "and a signed exact-path deletion plan are unavailable"
    )


def verify_synapse_retired(
    control: VerifiedControlRecord,
    inventory: ConsumerInventory,
    manifest: SynapseRetirementManifest,
    post_stop_scan: IndependenceScanReceipt,
    post_reboot_scan: IndependenceScanReceipt,
    *,
    roster: VerificationRoster,
    signer: OperationalSigner,
    signer_key_id: str,
) -> IndependenceReceipt:
    """Create a signed proof from two independently signed negative scans."""

    if control.synapse_state not in {
        SynapseRetirementState.COMMITTED,
        SynapseRetirementState.VERIFIED,
    }:
        raise CutoverSafetyError("Synapse retirement is not committed")
    if control.retirement_epoch < 1:
        raise CutoverSafetyError("Synapse retirement epoch is missing")
    try:
        verify_synapse_retirement_manifest(manifest, roster=roster)
        verify_consumer_inventory(inventory, roster=roster)
    except InventoryError as exc:
        raise CutoverSafetyError(
            "Synapse final manifest or inventory signature is invalid"
        ) from exc
    manifest_sha256 = hashlib.sha256(manifest.signed_bytes()).hexdigest()
    if manifest_sha256 != control.synapse_manifest_sha256:
        raise CutoverSafetyError("Synapse retirement manifest digest mismatch")
    stop_digest, stop_time = _verify_independence_scan(
        post_stop_scan,
        phase="post_stop",
        roster=roster,
    )
    reboot_digest, reboot_time = _verify_independence_scan(
        post_reboot_scan,
        phase="post_reboot",
        roster=roster,
    )
    if post_stop_scan.boot_id == post_reboot_scan.boot_id:
        raise CutoverSafetyError("post-reboot scan did not cross a boot boundary")
    if reboot_time <= stop_time:
        raise CutoverSafetyError("post-reboot scan capture time is not later")
    scan_digests = (stop_digest, reboot_digest)
    if inventory.scan_receipt_sha256 != scan_digests:
        raise CutoverSafetyError("signed inventory does not bind both scan receipts")
    expected_scan_source = hashlib.sha256(
        canonical_json_bytes(list(scan_digests))
    ).hexdigest()
    if inventory.source_scan_sha256 != expected_scan_source:
        raise CutoverSafetyError("signed inventory source digest does not bind final scans")
    inventory_sha256 = hashlib.sha256(inventory.signed_bytes()).hexdigest()
    try:
        signer_device_id = roster.key(signer_key_id).device_id
    except KeyError as exc:
        raise CutoverSafetyError("independence receipt signer is not in roster") from exc
    unsigned = IndependenceReceipt(
        schema="memo.synapse_independence_receipt.v1",
        attempt_id=control.attempt_id,
        control_oid=control.control_oid,
        retirement_epoch=control.retirement_epoch,
        synapse_manifest_sha256=manifest_sha256,
        consumer_inventory_sha256=inventory_sha256,
        scan_receipt_sha256=scan_digests,
        verified_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        signer_device_id=signer_device_id,
        signer_key_id=signer_key_id,
        roster_version=roster.version,
        signature="",
    )
    envelope = signer.sign(
        domain=SYNAPSE_INDEPENDENCE_RECEIPT_DOMAIN,
        payload=unsigned.signed_bytes(),
        key_id=signer_key_id,
    )
    receipt = replace(unsigned, signature=envelope.signature)
    verify_independence_receipt(
        receipt,
        control,
        inventory,
        manifest,
        post_stop_scan,
        post_reboot_scan,
        roster=roster,
    )
    return receipt


def _captured_at(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CutoverSafetyError("independence scan capture time is invalid") from exc
    if parsed.tzinfo is None:
        raise CutoverSafetyError("independence scan capture time lacks timezone")
    return parsed.astimezone(UTC)


def _verify_independence_scan(
    scan: IndependenceScanReceipt,
    *,
    phase: str,
    roster: VerificationRoster,
) -> tuple[str, datetime]:
    if (
        scan.schema != "memo.synapse_independence_scan.v1"
        or scan.phase != phase
        or not scan.boot_id
        or not _SHA256_RE.fullmatch(scan.source_scan_sha256)
        or not scan.signature
    ):
        raise CutoverSafetyError(f"{phase} independence scan is malformed")
    try:
        key = roster.key(scan.signer_key_id)
        if key.device_id != scan.signer_device_id:
            raise SignatureError("scan signer device does not own its key")
        OperationalVerifier().verify(
            domain=SYNAPSE_INDEPENDENCE_SCAN_DOMAIN,
            payload=scan.signed_bytes(),
            envelope=scan.signature_envelope(),
            roster=roster,
        )
    except (KeyError, SignatureError) as exc:
        raise CutoverSafetyError(f"{phase} independence scan signature is invalid") from exc
    surfaces = {row.surface for row in scan.observations}
    if surfaces != set(_INDEPENDENCE_SURFACES):
        raise CutoverSafetyError(f"{phase} scan observations are incomplete")
    identifiers = [(row.surface, row.identifier) for row in scan.observations]
    if (
        len(identifiers) != len(set(identifiers))
        or any(not row.identifier for row in scan.observations)
    ):
        raise CutoverSafetyError(f"{phase} scan observations are ambiguous")
    resurrected = [
        f"{row.surface}:{row.identifier}"
        for row in scan.observations
        if row.active or row.references
    ]
    if resurrected:
        raise CutoverSafetyError(
            "Synapse active reference resurrected: " + ",".join(sorted(resurrected))
        )
    observed_digest = hashlib.sha256(
        canonical_json_bytes([row.to_dict() for row in scan.observations])
    ).hexdigest()
    if observed_digest != scan.source_scan_sha256:
        raise CutoverSafetyError(f"{phase} scan source digest mismatch")
    return hashlib.sha256(scan.signed_bytes()).hexdigest(), _captured_at(scan.captured_at)


def verify_independence_receipt(
    receipt: IndependenceReceipt,
    control: VerifiedControlRecord,
    inventory: ConsumerInventory,
    manifest: SynapseRetirementManifest,
    post_stop_scan: IndependenceScanReceipt,
    post_reboot_scan: IndependenceScanReceipt,
    *,
    roster: VerificationRoster,
) -> None:
    try:
        verify_consumer_inventory(inventory, roster=roster)
        verify_synapse_retirement_manifest(manifest, roster=roster)
    except InventoryError as exc:
        raise CutoverSafetyError(
            "independence receipt artifact signature is invalid"
        ) from exc
    stop_digest, stop_time = _verify_independence_scan(
        post_stop_scan,
        phase="post_stop",
        roster=roster,
    )
    reboot_digest, reboot_time = _verify_independence_scan(
        post_reboot_scan,
        phase="post_reboot",
        roster=roster,
    )
    scan_digests = (
        stop_digest,
        reboot_digest,
    )
    if post_stop_scan.boot_id == post_reboot_scan.boot_id:
        raise CutoverSafetyError("post-reboot scan did not cross a boot boundary")
    if reboot_time <= stop_time:
        raise CutoverSafetyError("post-reboot scan capture time is not later")
    if inventory.scan_receipt_sha256 != scan_digests:
        raise CutoverSafetyError("signed inventory does not bind both scan receipts")
    expected_scan_source = hashlib.sha256(
        canonical_json_bytes(list(scan_digests))
    ).hexdigest()
    if inventory.source_scan_sha256 != expected_scan_source:
        raise CutoverSafetyError("signed inventory source digest does not bind final scans")
    receipt_sha256 = hashlib.sha256(receipt.signed_bytes()).hexdigest()
    control_binding = (
        control.synapse_state is SynapseRetirementState.COMMITTED
        and receipt.control_oid == control.control_oid
    ) or (
        control.synapse_state is SynapseRetirementState.VERIFIED
        and receipt.control_oid == control.previous_control_oid
        and receipt_sha256 == control.independence_receipt_sha256
    )
    expected = (
        control.synapse_state
        in {SynapseRetirementState.COMMITTED, SynapseRetirementState.VERIFIED}
        and control.retirement_epoch > 0
        and control.synapse_manifest_sha256
        == hashlib.sha256(manifest.signed_bytes()).hexdigest()
        and receipt.schema == "memo.synapse_independence_receipt.v1"
        and receipt.attempt_id == control.attempt_id
        and control_binding
        and receipt.retirement_epoch == control.retirement_epoch
        and receipt.synapse_manifest_sha256
        == hashlib.sha256(manifest.signed_bytes()).hexdigest()
        and receipt.consumer_inventory_sha256
        == hashlib.sha256(inventory.signed_bytes()).hexdigest()
        and receipt.scan_receipt_sha256 == scan_digests
    )
    if not expected:
        raise CutoverSafetyError("independence receipt is not bound to final authority")
    _captured_at(receipt.verified_at)
    try:
        key = roster.key(receipt.signer_key_id)
        if key.device_id != receipt.signer_device_id:
            raise SignatureError("receipt signer device does not own its key")
        OperationalVerifier().verify(
            domain=SYNAPSE_INDEPENDENCE_RECEIPT_DOMAIN,
            payload=receipt.signed_bytes(),
            envelope=receipt.signature_envelope(),
            roster=roster,
        )
    except (KeyError, SignatureError) as exc:
        raise CutoverSafetyError("independence receipt signature is invalid") from exc


def independence_scan_from_dict(value: dict[str, Any]) -> IndependenceScanReceipt:
    expected = {
        "schema",
        "phase",
        "boot_id",
        "captured_at",
        "source_scan_sha256",
        "observations",
        "signer_device_id",
        "signer_key_id",
        "roster_version",
        "signature",
    }
    if set(value) != expected or not isinstance(value.get("observations"), list):
        raise CutoverSafetyError("independence scan fields are invalid")
    observations: list[IndependenceObservation] = []
    for raw in value["observations"]:
        if (
            not isinstance(raw, dict)
            or set(raw) != {"surface", "identifier", "active", "references"}
            or raw["surface"] not in _INDEPENDENCE_SURFACES
            or not isinstance(raw["identifier"], str)
            or not isinstance(raw["active"], bool)
            or not isinstance(raw["references"], list)
            or any(not isinstance(item, str) for item in raw["references"])
        ):
            raise CutoverSafetyError("independence scan observation is invalid")
        observations.append(
            IndependenceObservation(
                surface=cast(Any, raw["surface"]),
                identifier=raw["identifier"],
                active=raw["active"],
                references=tuple(raw["references"]),
            )
        )
    string_fields = (
        "schema",
        "phase",
        "boot_id",
        "captured_at",
        "source_scan_sha256",
        "signer_device_id",
        "signer_key_id",
        "signature",
    )
    if (
        any(not isinstance(value.get(field), str) for field in string_fields)
        or isinstance(value.get("roster_version"), bool)
        or not isinstance(value.get("roster_version"), int)
    ):
        raise CutoverSafetyError("independence scan values are invalid")
    return IndependenceScanReceipt(
        schema=cast(Any, value["schema"]),
        phase=cast(Any, value["phase"]),
        boot_id=cast(str, value["boot_id"]),
        captured_at=cast(str, value["captured_at"]),
        source_scan_sha256=cast(str, value["source_scan_sha256"]),
        observations=tuple(observations),
        signer_device_id=cast(str, value["signer_device_id"]),
        signer_key_id=cast(str, value["signer_key_id"]),
        roster_version=cast(int, value["roster_version"]),
        signature=cast(str, value["signature"]),
    )


def independence_receipt_from_dict(value: dict[str, Any]) -> IndependenceReceipt:
    expected = {
        "schema",
        "attempt_id",
        "control_oid",
        "retirement_epoch",
        "synapse_manifest_sha256",
        "consumer_inventory_sha256",
        "scan_receipt_sha256",
        "verified_at",
        "signer_device_id",
        "signer_key_id",
        "roster_version",
        "signature",
    }
    if set(value) != expected:
        raise CutoverSafetyError("independence receipt fields are invalid")
    scans = value["scan_receipt_sha256"]
    string_fields = (
        "schema",
        "attempt_id",
        "control_oid",
        "synapse_manifest_sha256",
        "consumer_inventory_sha256",
        "verified_at",
        "signer_device_id",
        "signer_key_id",
        "signature",
    )
    if (
        any(not isinstance(value[field], str) or not value[field] for field in string_fields)
        or isinstance(value["retirement_epoch"], bool)
        or not isinstance(value["retirement_epoch"], int)
        or isinstance(value["roster_version"], bool)
        or not isinstance(value["roster_version"], int)
        or not isinstance(scans, list)
        or len(scans) != 2
        or any(not isinstance(item, str) for item in scans)
    ):
        raise CutoverSafetyError("independence receipt scan digests are invalid")
    try:
        return IndependenceReceipt(
            schema=cast(Any, value["schema"]),
            attempt_id=cast(str, value["attempt_id"]),
            control_oid=cast(str, value["control_oid"]),
            retirement_epoch=cast(int, value["retirement_epoch"]),
            synapse_manifest_sha256=cast(str, value["synapse_manifest_sha256"]),
            consumer_inventory_sha256=cast(str, value["consumer_inventory_sha256"]),
            scan_receipt_sha256=(scans[0], scans[1]),
            verified_at=cast(str, value["verified_at"]),
            signer_device_id=cast(str, value["signer_device_id"]),
            signer_key_id=cast(str, value["signer_key_id"]),
            roster_version=cast(int, value["roster_version"]),
            signature=cast(str, value["signature"]),
        )
    except (TypeError, ValueError) as exc:
        raise CutoverSafetyError("independence receipt values are invalid") from exc


__all__ = [
    "ATTEMPT_SENTINEL",
    "SYNAPSE_INDEPENDENCE_RECEIPT_DOMAIN",
    "SYNAPSE_INDEPENDENCE_SCAN_DOMAIN",
    "CutoverSafetyError",
    "SafetyError",
    "assert_retirement_cleanup_authority",
    "assert_safe_attempt_root",
    "independence_receipt_from_dict",
    "independence_scan_from_dict",
    "initialize_attempt_root",
    "prepare_synapse_retirement",
    "resolve_under_attempt",
    "verify_independence_receipt",
    "verify_synapse_retired",
]
