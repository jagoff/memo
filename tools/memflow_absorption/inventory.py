"""Read-only consumer and Synapse retirement inventory builders."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

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
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ACTIVE_RETIREMENT_RE = re.compile(
    r"(?i)(?:"
    r"\bSYNAPSE_[A-Z0-9_]+\b"
    r"|\bsynapse(?:[-_.][a-z0-9]+)*\b"
    r"|\bmemflow(?:[-_.][a-z0-9]+)*\b"
    r")"
)
CONSUMER_INVENTORY_DOMAIN = "memo.cutover.consumer_inventory.v1"
# The schema evolved, but its Ed25519 purpose remains the pre-approved
# retirement authority domain.  Do not silently introduce an unsigned domain.
SYNAPSE_RETIREMENT_DOMAIN = "memo.cutover.synapse_retirement.v1"


class InventoryError(RuntimeError):
    """An inventory input cannot be scanned without following unsafe paths."""


@dataclass(frozen=True)
class RetirementAuditReceipt:
    """Deterministic read-only result of the final filesystem negative scan."""

    status: Literal["verified"]
    source_scan_sha256: str
    manifest_sha256: str
    roots: tuple[str, ...]
    archived_roots: tuple[str, ...]
    archived_provenance: tuple[str, ...]
    file_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "source_scan_sha256": self.source_scan_sha256,
            "manifest_sha256": self.manifest_sha256,
            "roots": list(self.roots),
            "archived_roots": list(self.archived_roots),
            "archived_provenance": list(self.archived_provenance),
            "file_count": self.file_count,
        }


def _string_tuple(value: Any, description: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise InventoryError(f"{description} must be a string list")
    return tuple(value)


def consumer_inventory_from_dict(value: Mapping[str, Any]) -> ConsumerInventory:
    """Parse the exact signed inventory schema before any authority check."""

    expected = {
        "schema",
        "rows",
        "blockers",
        "source_scan_sha256",
        "signer_device_id",
        "signer_key_id",
        "roster_version",
        "signature",
        "verification_phases",
        "covered_surfaces",
        "surface_observations",
        "scan_receipt_sha256",
    }
    if set(value) != expected or not isinstance(value.get("rows"), list):
        raise InventoryError("consumer inventory fields are invalid")
    rows: list[ConsumerInventoryRow] = []
    row_fields = {
        "kind",
        "location",
        "references",
        "label",
        "active",
        "program_arguments",
        "correlated_launchd_label",
        "run_at_load",
        "keep_alive",
        "start_interval_seconds",
        "start_calendar_interval",
        "watch_paths",
        "throttle_interval_seconds",
        "environment_keys",
        "environment",
    }
    for raw in value["rows"]:
        if not isinstance(raw, dict) or set(raw) != row_fields:
            raise InventoryError("consumer inventory row fields are invalid")
        calendar = raw["start_calendar_interval"]
        environment = raw["environment"]
        keep_alive = raw["keep_alive"]
        if (
            raw["kind"] not in {"source", "process", "launchd"}
            or not isinstance(raw["location"], str)
            or not isinstance(raw["label"], str)
            or not isinstance(raw["active"], bool)
            or not isinstance(raw["correlated_launchd_label"], str)
            or not isinstance(raw["run_at_load"], bool)
            or not (
                isinstance(keep_alive, bool)
                or (
                    isinstance(keep_alive, dict)
                    and all(isinstance(key, str) for key in keep_alive)
                )
            )
            or (
                raw["start_interval_seconds"] is not None
                and (
                    isinstance(raw["start_interval_seconds"], bool)
                    or not isinstance(raw["start_interval_seconds"], int)
                )
            )
            or (
                raw["throttle_interval_seconds"] is not None
                and (
                    isinstance(raw["throttle_interval_seconds"], bool)
                    or not isinstance(raw["throttle_interval_seconds"], int)
                )
            )
            or not isinstance(calendar, list)
            or any(
                not isinstance(item, dict)
                or any(
                    not isinstance(key, str)
                    or isinstance(number, bool)
                    or not isinstance(number, int)
                    for key, number in item.items()
                )
                for item in calendar
            )
            or not isinstance(environment, list)
            or any(
                not isinstance(item, list)
                or len(item) != 2
                or any(not isinstance(part, str) for part in item)
                for item in environment
            )
        ):
            raise InventoryError("consumer inventory row values are invalid")
        rows.append(
            ConsumerInventoryRow(
                kind=cast(Any, raw["kind"]),
                location=raw["location"],
                references=_string_tuple(raw["references"], "consumer references"),
                label=raw["label"],
                active=raw["active"],
                program_arguments=_string_tuple(
                    raw["program_arguments"], "consumer arguments"
                ),
                correlated_launchd_label=raw["correlated_launchd_label"],
                run_at_load=raw["run_at_load"],
                keep_alive=keep_alive,
                start_interval_seconds=raw["start_interval_seconds"],
                start_calendar_interval=tuple(
                    tuple(sorted(item.items())) for item in calendar
                ),
                watch_paths=_string_tuple(raw["watch_paths"], "consumer watch paths"),
                throttle_interval_seconds=raw["throttle_interval_seconds"],
                environment_keys=_string_tuple(
                    raw["environment_keys"], "consumer environment keys"
                ),
                environment=tuple((item[0], item[1]) for item in environment),
            )
        )
    observations = value.get("surface_observations")
    if not isinstance(observations, dict) or any(
        not isinstance(key, str) or not isinstance(items, list)
        for key, items in observations.items()
    ):
        raise InventoryError("consumer inventory surface observations are invalid")
    string_fields = (
        "schema",
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
        raise InventoryError("consumer inventory signer fields are invalid")
    return ConsumerInventory(
        schema=cast(Any, value["schema"]),
        rows=tuple(rows),
        blockers=_string_tuple(value["blockers"], "consumer blockers"),
        source_scan_sha256=cast(str, value["source_scan_sha256"]),
        signer_device_id=cast(str, value["signer_device_id"]),
        signer_key_id=cast(str, value["signer_key_id"]),
        roster_version=cast(int, value["roster_version"]),
        signature=cast(str, value["signature"]),
        verification_phases=cast(
            Any, _string_tuple(value["verification_phases"], "verification phases")
        ),
        covered_surfaces=_string_tuple(
            value["covered_surfaces"], "covered surfaces"
        ),
        surface_observations={
            key: _string_tuple(items, f"surface observation {key}")
            for key, items in observations.items()
        },
        scan_receipt_sha256=_string_tuple(
            value["scan_receipt_sha256"], "scan receipt digests"
        ),
    )


def synapse_retirement_manifest_from_dict(
    value: Mapping[str, Any],
) -> SynapseRetirementManifest:
    """Parse the exact signed retirement manifest schema."""

    expected = {
        "schema",
        "source_commit",
        "files",
        "symbols",
        "tests",
        "goldens",
        "active_reference_sha256",
        "signer_key_id",
        "signature",
        "operations",
    }
    if set(value) != expected or not isinstance(value.get("operations"), list):
        raise InventoryError("Synapse retirement manifest fields are invalid")
    operations: list[SynapseOperation] = []
    operation_fields = {
        "source_operation",
        "source_files",
        "source_symbols",
        "consumers",
        "daemon_routes",
        "exclusion_reason",
        "fixture_paths",
    }
    for raw in value["operations"]:
        if (
            not isinstance(raw, dict)
            or set(raw) != operation_fields
            or not isinstance(raw["source_operation"], str)
            or not (
                raw["exclusion_reason"] is None
                or isinstance(raw["exclusion_reason"], str)
            )
        ):
            raise InventoryError("Synapse retirement operation fields are invalid")
        operations.append(
            SynapseOperation(
                source_operation=raw["source_operation"],
                source_files=_string_tuple(raw["source_files"], "operation source files"),
                source_symbols=_string_tuple(
                    raw["source_symbols"], "operation source symbols"
                ),
                consumers=_string_tuple(raw["consumers"], "operation consumers"),
                daemon_routes=_string_tuple(
                    raw["daemon_routes"], "operation daemon routes"
                ),
                exclusion_reason=raw["exclusion_reason"],
                fixture_paths=_string_tuple(
                    raw["fixture_paths"], "operation fixture paths"
                ),
            )
        )
    for field in (
        "schema",
        "source_commit",
        "active_reference_sha256",
        "signer_key_id",
        "signature",
    ):
        if not isinstance(value.get(field), str):
            raise InventoryError("Synapse retirement manifest values are invalid")
    return SynapseRetirementManifest(
        schema=cast(Any, value["schema"]),
        source_commit=cast(str, value["source_commit"]),
        files=_string_tuple(value["files"], "Synapse retirement files"),
        symbols=_string_tuple(value["symbols"], "Synapse retirement symbols"),
        tests=_string_tuple(value["tests"], "Synapse retirement tests"),
        goldens=_string_tuple(value["goldens"], "Synapse retirement goldens"),
        active_reference_sha256=cast(str, value["active_reference_sha256"]),
        signer_key_id=cast(str, value["signer_key_id"]),
        signature=cast(str, value["signature"]),
        operations=tuple(operations),
    )


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


def _retirement_references(text: str) -> tuple[str, ...]:
    return tuple(
        sorted(set(match.group(0) for match in _ACTIVE_RETIREMENT_RE.finditer(text)))
    )


def _normalized_scan_roots(
    roots: tuple[Path, ...] | list[Path],
    *,
    description: str,
) -> tuple[Path, ...]:
    normalized = tuple(
        Path(os.path.abspath(os.fspath(root)))
        for root in roots
    )
    if not normalized:
        raise InventoryError(f"{description} requires at least one root")
    if len(normalized) != len(set(normalized)):
        raise InventoryError(f"{description} contains duplicate roots")
    for index, root in enumerate(normalized):
        for other in normalized[index + 1 :]:
            if root in other.parents or other in root.parents:
                raise InventoryError(f"{description} contains overlapping roots")
    return normalized


def build_independence_receipt(
    roots: tuple[Path, ...] | list[Path],
    *,
    manifest: SynapseRetirementManifest,
    archived_roots: tuple[Path, ...] | list[Path] = (),
    roster: VerificationRoster | None = None,
) -> RetirementAuditReceipt:
    """Prove that installed roots contain no active retired-runtime reference.

    ``archived_roots`` is deliberately separate from installed roots.  Inside
    those roots, only paths enumerated by the final retirement manifest may
    contain a retired reference.  A path match in an installed root is never
    treated as provenance, because that would bless the still-active source
    tree that this audit is intended to detect.

    The walk is descriptor-safe through :func:`_safe_files`, never follows a
    symlink, and performs no filesystem mutation.
    """

    installed = _normalized_scan_roots(roots, description="retirement audit")
    archives = (
        _normalized_scan_roots(
            archived_roots,
            description="retirement archive audit",
        )
        if archived_roots
        else ()
    )
    if archives:
        if roster is None:
            raise InventoryError(
                "archived provenance requires a roster-verified signed manifest"
            )
        verify_synapse_retirement_manifest(manifest, roster=roster)
    if any(
        installed_root == archive_root
        or installed_root in archive_root.parents
        or archive_root in installed_root.parents
        for installed_root in installed
        for archive_root in archives
    ):
        raise InventoryError("installed and archived retirement roots overlap")
    allowed_archive_paths = frozenset(
        (*manifest.files, *manifest.tests, *manifest.goldens)
    )
    if any(
        not path
        or Path(path).is_absolute()
        or any(part in {"", ".", ".."} for part in Path(path).parts)
        for path in allowed_archive_paths
    ):
        raise InventoryError("retirement manifest contains an unsafe archive path")

    records: list[dict[str, object]] = []
    archived_provenance: set[str] = set()
    file_count = 0
    for role, scan_roots in (("installed", installed), ("archive", archives)):
        for root_index, root in enumerate(scan_roots):
            for path, kind in _safe_files(root):
                relative = path.relative_to(root).as_posix()
                if kind == "symlink":
                    raise InventoryError(
                        f"retirement audit cannot prove a symlink: {path}"
                    )
                data, text = _read_text(path)
                file_count += 1
                references = _retirement_references(relative + "\n" + text)
                allowed = (
                    role == "archive"
                    and relative in allowed_archive_paths
                    and bool(references)
                )
                if references and not allowed:
                    raise InventoryError(
                        "unlisted active reference: "
                        f"{path} ({','.join(references)})"
                    )
                if allowed:
                    archived_provenance.add(relative)
                records.append(
                    {
                        "role": role,
                        "root_index": root_index,
                        "path": relative,
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "archived_provenance": allowed,
                    }
                )
    manifest_sha256 = hashlib.sha256(manifest.signed_bytes()).hexdigest()
    roots_as_text = tuple(str(root) for root in installed)
    archives_as_text = tuple(str(root) for root in archives)
    return RetirementAuditReceipt(
        status="verified",
        source_scan_sha256=hashlib.sha256(
            canonical_json_bytes(
                {
                    "manifest_sha256": manifest_sha256,
                    "roots": list(roots_as_text),
                    "archived_roots": list(archives_as_text),
                    "records": records,
                }
            )
        ).hexdigest(),
        manifest_sha256=manifest_sha256,
        roots=roots_as_text,
        archived_roots=archives_as_text,
        archived_provenance=tuple(sorted(archived_provenance)),
        file_count=file_count,
    )


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
    surface_observations: Mapping[str, tuple[str, ...]] | None = None,
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
                    environment_keys=job.environment_keys,
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
        surface_observations=(
            {
                key: tuple(values)
                for key, values in sorted(surface_observations.items())
            }
            if surface_observations is not None
            else {}
        ),
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
    if inventory.schema != "memo.cutover_consumer_inventory.v1":
        raise InventoryError("consumer inventory schema is invalid")
    if inventory.blockers or not inventory.signature:
        raise InventoryError("consumer inventory is blocked or unsigned")
    if not _SHA256_RE.fullmatch(inventory.source_scan_sha256):
        raise InventoryError("consumer inventory source digest is invalid")
    if not inventory.signer_key_id or not inventory.signer_device_id or inventory.roster_version < 1:
        raise InventoryError("consumer inventory signer fields are incomplete")
    if inventory.roster_version != roster.version:
        raise InventoryError("consumer inventory roster version is invalid")
    try:
        key = roster.key(inventory.signer_key_id)
    except Exception as exc:
        raise InventoryError("consumer inventory signer key is not in roster") from exc
    if key.device_id != inventory.signer_device_id or "origin" not in key.roles:
        raise InventoryError("consumer inventory signer key ownership is invalid")
    if any(not isinstance(k, str) or not isinstance(v, tuple) or any(not isinstance(x, str) for x in v)
           for k, v in inventory.surface_observations.items()):
        raise InventoryError("consumer inventory surface observations are invalid")
    if any(row.kind not in {"source", "process", "launchd"} or not row.location or not row.references
           for row in inventory.rows):
        raise InventoryError("consumer inventory row is malformed")
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
    if manifest.schema != "memo.synapse_retirement.v2":
        raise InventoryError("Synapse retirement schema is invalid")
    if not manifest.signature:
        raise InventoryError("Synapse retirement manifest is unsigned")
    if (not re.fullmatch(r"[0-9a-f]{40}", manifest.source_commit)
            or not _SHA256_RE.fullmatch(manifest.active_reference_sha256)):
        raise InventoryError("Synapse retirement digest fields are invalid")
    if not manifest.signer_key_id:
        raise InventoryError("Synapse retirement signer key is incomplete")
    try:
        key = roster.key(manifest.signer_key_id)
    except Exception as exc:
        raise InventoryError("Synapse retirement signer key is not in roster") from exc
    if "origin" not in key.roles:
        raise InventoryError("Synapse retirement signer key ownership is invalid")
    if not manifest.operations:
        raise InventoryError("Synapse retirement operation catalog is empty")
    for operation in manifest.operations:
        if (not operation.source_operation or not operation.source_files
                or any(not isinstance(path, str) or not path for path in operation.source_files)
                or any(not isinstance(path, str) or not path for path in operation.fixture_paths)):
            raise InventoryError("Synapse retirement operation is malformed")
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
    "RetirementAuditReceipt",
    "build_consumer_inventory",
    "build_independence_receipt",
    "build_synapse_retirement_manifest",
    "consumer_inventory_from_dict",
    "synapse_retirement_manifest_from_dict",
    "verify_consumer_inventory",
    "verify_synapse_retirement_manifest",
]
