"""Read-only consumer and Synapse retirement inventory builders."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

from memo.atomic_io import open_secure_directory
from memo.errors import SignatureError
from memo.operational_event import canonical_json_bytes
from memo.operational_roster import VerificationRoster
from memo.operational_signing import OperationalSigner, OperationalVerifier
from tools.memflow_absorption.schemas import (
    ConsumerInventory,
    ConsumerInventoryRow,
    LaunchdSnapshot,
    ProcessRecord,
    ProcessSnapshot,
    SynapseOperation,
    SynapseRetirementManifest,
)
from tools.memflow_absorption.synapse_catalog import (
    SynapseCatalogError,
    discover_synapse_operations,
)

_REFERENCE_RE = re.compile(r"(?i)(?:\bmemflow\b|memflow[-_.a-z0-9]*|synapse_memflow_[a-z0-9_]+)")
_CONSUMER_REFERENCE_RE = re.compile(
    r"(?i)(?:\bmemflow\b|memflow[-_.a-z0-9]*|synapse_memflow_[a-z0-9_]+|\bsynapse\b|synapse[-_.a-z0-9]*)"
)
_SYMBOL_RE = re.compile(r"\b(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)")
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
CONSUMER_INVENTORY_DOMAIN = "memo.cutover.consumer_inventory.v1"
# The schema evolved, but its Ed25519 purpose remains the pre-approved
# retirement authority domain.  Do not silently introduce an unsigned domain.
SYNAPSE_RETIREMENT_DOMAIN = "memo.cutover.synapse_retirement.v1"


class InventoryError(RuntimeError):
    """An inventory input cannot be scanned without following unsafe paths."""


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
            raise InventoryError(f"inventory root contains symlink component: {current}")


def _safe_files(root: Path) -> Iterator[tuple[Path, str]]:
    absolute = Path(os.path.abspath(os.fspath(root)))
    _reject_symlink_components(absolute)
    try:
        observed = absolute.stat()
    except OSError as exc:
        raise InventoryError(f"inventory root is unavailable: {root}") from exc
    if not stat.S_ISDIR(observed.st_mode):
        raise InventoryError(f"inventory root is not a directory: {root}")
    for directory, directory_names, file_names in os.walk(
        absolute,
        topdown=True,
        followlinks=False,
    ):
        parent = Path(directory)
        retained: list[str] = []
        for name in sorted(directory_names):
            child = parent / name
            if child.is_symlink():
                yield child, "symlink"
            else:
                retained.append(name)
        directory_names[:] = retained
        for name in sorted(file_names):
            path = parent / name
            if path.is_symlink():
                yield path, "symlink"
                continue
            try:
                if path.is_file():
                    yield path, "file"
            except OSError as exc:
                raise InventoryError(f"inventory input changed during scan: {path}") from exc


def _read_text(path: Path) -> tuple[bytes, str]:
    try:
        with open_secure_directory(path.parent) as directory:
            data = directory.read_bytes(path.name)
    except (OSError, ValueError) as exc:
        raise InventoryError(f"inventory file is unreadable: {path}") from exc
    try:
        return data, data.decode("utf-8")
    except UnicodeDecodeError:
        return data, ""


def _references(text: str) -> tuple[str, ...]:
    return tuple(sorted(set(match.group(0) for match in _REFERENCE_RE.finditer(text))))


def _consumer_references(text: str) -> tuple[str, ...]:
    """Return retired-runtime references without broadening the source manifest.

    The Synapse retirement manifest intentionally inventories the historical
    Memflow-specific surface.  Consumer staging additionally needs a plain
    ``synapse`` label so a live LaunchAgent that has already stopped naming
    Memflow cannot disappear from the replacement plan.
    """
    return tuple(sorted(set(match.group(0) for match in _CONSUMER_REFERENCE_RE.finditer(text))))


def _process_launchd_matches(
    process: ProcessRecord,
    launchd_snapshot: LaunchdSnapshot,
) -> tuple[str, ...]:
    candidates = {(process.executable, *process.argv), process.argv}
    return tuple(
        sorted(
            job.label
            for job in launchd_snapshot.records
            if job.loaded and job.program_arguments in candidates
        )
    )


def build_consumer_inventory(
    roots: tuple[Path, ...] | list[Path],
    process_snapshot: ProcessSnapshot,
    launchd_snapshot: LaunchdSnapshot,
    *,
    signer: OperationalSigner | None = None,
    signer_key_id: str = "",
    roster: VerificationRoster | None = None,
) -> ConsumerInventory:
    """Combine explicit source roots with already-captured process/launchd inputs."""

    rows: list[ConsumerInventoryRow] = []
    blockers: set[str] = set()
    scan_records: list[dict[str, str]] = []
    for root_index, root in enumerate(roots):
        absolute = Path(os.path.abspath(os.fspath(root)))
        for path, kind in _safe_files(root):
            relative = path.relative_to(absolute).as_posix()
            if kind == "symlink":
                blockers.add(f"symlink-skipped:{relative}")
                scan_records.append(
                    {
                        "root_index": str(root_index),
                        "path": relative,
                        "kind": "symlink",
                    }
                )
                continue
            data, text = _read_text(path)
            scan_records.append(
                {
                    "root_index": str(root_index),
                    "path": relative,
                    "kind": "file",
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            # Source rows preserve the original Memflow-specific inventory
            # contract.  Plain ``synapse`` is intentionally broadened only for
            # process/launchd observations below; otherwise every source file
            # that merely documents Synapse becomes a fake runnable consumer.
            references = _references(relative + "\n" + text)
            if references:
                rows.append(
                    ConsumerInventoryRow(
                        kind="source",
                        location=str(path),
                        references=references,
                        label=relative,
                    )
                )
    for process in sorted(process_snapshot.records, key=lambda row: row.pid):
        text = "\n".join((process.executable, *process.argv))
        references = _consumer_references(text)
        if references:
            correlated = _process_launchd_matches(process, launchd_snapshot)
            if len(correlated) > 1:
                blockers.add(f"ambiguous-process-launchd-correlation:pid:{process.pid}")
            rows.append(
                ConsumerInventoryRow(
                    kind="process",
                    location=f"pid:{process.pid}:{process.executable}",
                    references=references,
                    label=text,
                    program_arguments=(process.executable, *process.argv),
                    correlated_launchd_label=correlated[0] if len(correlated) == 1 else "",
                )
            )
    for job in sorted(launchd_snapshot.records, key=lambda row: row.label):
        text = "\n".join(
            (
                job.label,
                job.plist_path,
                *job.program_arguments,
                *job.environment_keys,
            )
        )
        references = _consumer_references(text)
        if references:
            rows.append(
                ConsumerInventoryRow(
                    kind="launchd",
                    location=f"{job.label}:{job.plist_path}",
                    references=references,
                    label=job.label,
                    active=job.loaded,
                    program_arguments=job.program_arguments,
                    run_at_load=job.run_at_load,
                    keep_alive=job.keep_alive,
                    start_interval_seconds=job.start_interval_seconds,
                    start_calendar_interval=job.start_calendar_interval,
                    watch_paths=job.watch_paths,
                    throttle_interval_seconds=job.throttle_interval_seconds,
                    environment=job.environment,
                )
            )
    if (signer is None) != (roster is None) or (signer is None) != (signer_key_id == ""):
        raise InventoryError("inventory signing authority must be complete")
    signer_device_id = ""
    roster_version = 0
    if roster is not None:
        signer_device_id = roster.key(signer_key_id).device_id
        roster_version = roster.version
    unsigned = ConsumerInventory(
        schema="memo.cutover_consumer_inventory.v1",
        rows=tuple(sorted(rows, key=lambda row: (row.kind, row.location))),
        blockers=tuple(sorted(blockers)),
        source_scan_sha256=hashlib.sha256(canonical_json_bytes(scan_records)).hexdigest(),
        signer_device_id=signer_device_id,
        signer_key_id=signer_key_id,
        roster_version=roster_version,
        signature="",
    )
    if signer is None or unsigned.blockers:
        return unsigned
    assert roster is not None
    envelope = signer.sign(
        domain=CONSUMER_INVENTORY_DOMAIN,
        payload=unsigned.signed_bytes(),
        key_id=signer_key_id,
    )
    signed = replace(unsigned, signature=envelope.signature)
    verify_consumer_inventory(signed, roster=roster)
    return signed


def build_synapse_retirement_manifest(
    snapshot: Path,
    *,
    signer: OperationalSigner | None = None,
    signer_key_id: str = "",
    roster: VerificationRoster | None = None,
) -> SynapseRetirementManifest:
    """Enumerate the complete Memflow-specific surface in a pinned Synapse tree."""

    root = Path(os.path.abspath(os.fspath(snapshot)))
    try:
        source_bytes, _source_text = _read_text(root / "source.json")
        source_record = json.loads(source_bytes)
        if canonical_json_bytes(source_record) != source_bytes:
            raise InventoryError("Synapse source record is not canonical JSON")
        source_commit = source_record["source_commit"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise InventoryError("Synapse snapshot lacks a pinned source commit") from exc
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise InventoryError("Synapse source commit is invalid")

    try:
        operations: tuple[SynapseOperation, ...] = discover_synapse_operations(root)
    except SynapseCatalogError as exc:
        raise InventoryError(str(exc)) from exc
    if not operations:
        raise InventoryError("Synapse canonical operation catalog is empty")

    files: set[str] = set()
    symbols: set[str] = set()
    tests: set[str] = set()
    goldens: set[str] = set()
    scan_records: list[dict[str, str]] = []
    for path, kind in _safe_files(root):
        relative = path.relative_to(root).as_posix()
        if kind == "symlink":
            raise InventoryError(f"Synapse snapshot contains symlink: {relative}")
        data, text = _read_text(path)
        scan_records.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest()})
        if not _references(relative + "\n" + text):
            continue
        if relative.startswith("tests/goldens/"):
            goldens.add(relative)
        elif relative.startswith("tests/"):
            tests.add(relative)
        else:
            files.add(relative)
        symbols.update(
            name
            for name in (*_SYMBOL_RE.findall(text), *_IDENTIFIER_RE.findall(text))
            if "memflow" in name.casefold()
        )
    if (signer is None) != (roster is None) or (signer is None) != (signer_key_id == ""):
        raise InventoryError("retirement signing authority must be complete")
    unsigned = SynapseRetirementManifest(
        schema="memo.synapse_retirement.v2",
        source_commit=source_commit,
        files=tuple(sorted(files)),
        symbols=tuple(sorted(symbols)),
        tests=tuple(sorted(tests)),
        goldens=tuple(sorted(goldens)),
        active_reference_sha256=hashlib.sha256(canonical_json_bytes(scan_records)).hexdigest(),
        signer_key_id=signer_key_id,
        signature="",
        operations=operations,
    )
    if signer is None:
        return unsigned
    assert roster is not None
    envelope = signer.sign(
        domain=SYNAPSE_RETIREMENT_DOMAIN,
        payload=unsigned.signed_bytes(),
        key_id=signer_key_id,
    )
    signed = replace(unsigned, signature=envelope.signature)
    verify_synapse_retirement_manifest(signed, roster=roster)
    return signed


def verify_consumer_inventory(
    inventory: ConsumerInventory,
    *,
    roster: VerificationRoster,
) -> None:
    if inventory.blockers or not inventory.signature:
        raise InventoryError("consumer inventory is blocked or unsigned")
    try:
        OperationalVerifier().verify(
            domain=CONSUMER_INVENTORY_DOMAIN,
            payload=inventory.signed_bytes(),
            envelope=inventory.signature_envelope(),
            roster=roster,
        )
    except SignatureError as exc:
        raise InventoryError("consumer inventory signature is invalid") from exc


def verify_synapse_retirement_manifest(
    manifest: SynapseRetirementManifest,
    *,
    roster: VerificationRoster,
) -> None:
    if not manifest.signature:
        raise InventoryError("Synapse retirement manifest is unsigned")
    try:
        OperationalVerifier().verify(
            domain=SYNAPSE_RETIREMENT_DOMAIN,
            payload=manifest.signed_bytes(),
            envelope=manifest.signature_envelope(roster_version=roster.version),
            roster=roster,
        )
    except SignatureError as exc:
        raise InventoryError("Synapse retirement signature is invalid") from exc


__all__ = [
    "CONSUMER_INVENTORY_DOMAIN",
    "SYNAPSE_RETIREMENT_DOMAIN",
    "InventoryError",
    "build_consumer_inventory",
    "build_synapse_retirement_manifest",
    "verify_consumer_inventory",
    "verify_synapse_retirement_manifest",
]
