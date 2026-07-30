"""Filesystem authority checks for cutover attempts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from memo.atomic_io import open_secure_directory
from memo.operational_event import canonical_json_bytes
from tools.memflow_absorption.control_record import (
    CutoverSafetyError,
    prepare_synapse_retirement,
)
from tools.memflow_absorption.schemas import (
    ConsumerInventory,
    IndependenceReceipt,
    SynapseRetirementManifest,
    SynapseRetirementState,
    VerifiedControlRecord,
)

ATTEMPT_SENTINEL = ".memo-cutover-attempt.json"
_ATTEMPT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_INDEPENDENCE_PHASES: tuple[Literal["post_stop", "post_reboot"], ...] = (
    "post_stop",
    "post_reboot",
)
_INDEPENDENCE_SURFACES = (
    "launchagent",
    "mcp_gateway_route",
    "port",
    "process",
    "shell_config_path",
    "state_root",
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


def verify_synapse_retired(
    control: VerifiedControlRecord,
    inventory: ConsumerInventory,
    manifest: SynapseRetirementManifest,
) -> IndependenceReceipt:
    """Prove retirement from signed, post-stop and post-reboot negative scans."""

    if control.synapse_state not in {
        SynapseRetirementState.COMMITTED,
        SynapseRetirementState.VERIFIED,
    }:
        raise CutoverSafetyError("Synapse retirement is not committed")
    if control.retirement_epoch < 1:
        raise CutoverSafetyError("Synapse retirement epoch is missing")
    if manifest.schema != "memo.synapse_retirement.v2" or not manifest.signature:
        raise CutoverSafetyError("Synapse retirement manifest is unsigned")
    if (
        not _SHA256_RE.fullmatch(manifest.active_reference_sha256)
        or not manifest.signer_key_id
    ):
        raise CutoverSafetyError("Synapse retirement manifest authority is malformed")
    manifest_sha256 = hashlib.sha256(manifest.signed_bytes()).hexdigest()
    if manifest_sha256 != control.synapse_manifest_sha256:
        raise CutoverSafetyError("Synapse retirement manifest digest mismatch")
    if (
        inventory.schema != "memo.cutover_consumer_inventory.v1"
        or not inventory.signature
        or inventory.blockers
        or not _SHA256_RE.fullmatch(inventory.source_scan_sha256)
        or not inventory.signer_device_id
        or not inventory.signer_key_id
        or inventory.roster_version < 1
    ):
        raise CutoverSafetyError("Synapse independence inventory is blocked or unsigned")
    if inventory.verification_phases != _INDEPENDENCE_PHASES:
        raise CutoverSafetyError(
            "Synapse independence requires post-stop and post-reboot scans"
        )
    if inventory.covered_surfaces != _INDEPENDENCE_SURFACES:
        raise CutoverSafetyError("Synapse independence surface coverage is incomplete")
    active_rows = tuple(row for row in inventory.rows if row.active or row.references)
    if active_rows:
        kinds = ",".join(
            sorted({f"{row.kind}:{row.location}" for row in active_rows})
        )
        raise CutoverSafetyError(f"Synapse active reference resurrected: {kinds}")
    inventory_sha256 = hashlib.sha256(inventory.signed_bytes()).hexdigest()
    return IndependenceReceipt(
        schema="memo.synapse_independence_receipt.v1",
        attempt_id=control.attempt_id,
        control_oid=control.control_oid,
        retirement_epoch=control.retirement_epoch,
        synapse_manifest_sha256=manifest_sha256,
        consumer_inventory_sha256=inventory_sha256,
        verification_phases=_INDEPENDENCE_PHASES,
        covered_surfaces=_INDEPENDENCE_SURFACES,
        verified_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )


__all__ = [
    "ATTEMPT_SENTINEL",
    "CutoverSafetyError",
    "SafetyError",
    "assert_safe_attempt_root",
    "initialize_attempt_root",
    "prepare_synapse_retirement",
    "resolve_under_attempt",
    "verify_synapse_retired",
]
