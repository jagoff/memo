"""Authority-gated, operator-only staging for retired Synapse consumers."""

from __future__ import annotations

import hashlib
import os
import plistlib
import stat
from dataclasses import dataclass
from pathlib import Path

from memo.atomic_io import SecureDirectory, open_secure_directory
from memo.operational_event import canonical_json_bytes
from memo.operational_roster import VerificationRoster
from tools.memflow_absorption.inventory import InventoryError, verify_consumer_inventory
from tools.memflow_absorption.manifest import ManifestError, verify_capability_manifest
from tools.memflow_absorption.schemas import (
    CapabilityManifest,
    ConsumerInventory,
    ConsumerInventoryRow,
    ConsumerReplacement,
    ConsumerReplacementPlan,
    OperationMappingRow,
)


class ConsumerMigrationError(RuntimeError):
    """The observed consumers cannot be safely staged for replacement."""


@dataclass(frozen=True)
class _ReplacementSpec:
    new_label: str
    command_tail: tuple[str, ...]
    owner: str
    authority_operation: str
    schedule_kind: str
    whatsapp: bool = False


# Exact labels are part of the reviewed cutover contract. A substring match can
# silently bless a newly introduced service, so unknown labels always block.
_REPLACEMENTS: dict[str, _ReplacementSpec] = {
    "com.synapse.whatsapp-ingest": _ReplacementSpec(
        "com.memo.import-whatsapp",
        (),
        "memo_native",
        "synapse.whatsapp_live.message",
        "periodic",
        whatsapp=True,
    ),
    "com.synapse.watcher": _ReplacementSpec(
        "com.memo.watch",
        ("watch",),
        "memo_native",
        "synapse.watcher.event",
        "persistent",
    ),
    "com.synapse.memo-recall-daemon": _ReplacementSpec(
        "com.memo.recall-daemon",
        ("recall-daemon", "_serve"),
        "memo_native",
        "synapse.cli.ops",
        "persistent",
    ),
    "com.synapse.memo-nightly": _ReplacementSpec(
        "com.memo.dream-nightly",
        ("dream", "run"),
        "memo_native",
        "synapse.cli.ops",
        "periodic",
    ),
    "com.synapse.morning-digest": _ReplacementSpec(
        "com.memo.digest",
        ("digest",),
        "memo_native",
        "synapse.morning_digest.run",
        "periodic",
    ),
    "com.synapse.dream-synthesis": _ReplacementSpec(
        "com.memo.dream-synthesis",
        ("dream", "run"),
        "memo_native",
        "synapse.cli.ops",
        "periodic",
    ),
    "com.synapse.vault-ingest": _ReplacementSpec(
        "com.memo.reindex-vault",
        ("reindex",),
        "memo_native",
        "synapse.cli.ops",
        "periodic",
    ),
    "com.synapse.vault-archive": _ReplacementSpec(
        "com.memo.reindex-archive",
        ("reindex",),
        "memo_native",
        "synapse.vault_archive.move",
        "periodic",
    ),
    "com.synapse.dashboard": _ReplacementSpec(
        "client.close-reconnect",
        ("client-close-reconnect",),
        "client",
        "synapse.chat.ask",
        "client",
    ),
    "com.synapse.dashboard-relay": _ReplacementSpec(
        "client.close-reconnect",
        ("client-close-reconnect",),
        "client",
        "synapse.chat.ask",
        "client",
    ),
    "com.synapse.gateway": _ReplacementSpec(
        "client.close-reconnect",
        ("client-close-reconnect",),
        "client",
        "synapse.federate.query",
        "client",
    ),
    "com.synapse.mcp": _ReplacementSpec(
        "client.close-reconnect",
        ("client-close-reconnect",),
        "client",
        "synapse.federate.query",
        "client",
    ),
    "com.synapse.memo-daemon": _ReplacementSpec(
        "client.close-reconnect",
        ("client-close-reconnect",),
        "client",
        "synapse.chat.ask",
        "client",
    ),
}

_LAUNCHD_DIRECTORY = "LaunchAgents"
_WHATSAPP_VALUE_OPTIONS = {
    "--include-chat",
    "--exclude-chat",
    "--retention-days",
    "--since",
    "--notes-dir",
    "--db",
}
_WHATSAPP_FLAG_OPTIONS = {"--all-chats", "--index", "--no-index", "--json"}
_CALENDAR_KEYS = {"Month", "Day", "Weekday", "Hour", "Minute", "Second"}


def _mapping(label: str) -> _ReplacementSpec | None:
    """Return the exact reviewed replacement row for ``label``."""
    return _REPLACEMENTS.get(label)


def _stable_memo_binary(memo_bin: Path) -> str:
    candidate = Path(memo_bin)
    if not candidate.is_absolute():
        raise ConsumerMigrationError("Memo runtime binary must be an absolute path")
    try:
        resolved = candidate.resolve(strict=True)
        observed = resolved.stat()
    except OSError as exc:
        raise ConsumerMigrationError("Memo runtime binary is unavailable") from exc
    if not stat.S_ISREG(observed.st_mode) or not os.access(resolved, os.X_OK):
        raise ConsumerMigrationError("Memo runtime binary must be a regular executable")
    if any(part in {".venv", "venv"} for part in resolved.parts):
        raise ConsumerMigrationError("Memo runtime binary must use a stable isolated runtime")
    return str(resolved)


def _whatsapp_command(row: ConsumerInventoryRow, memo_bin: str) -> tuple[str, ...]:
    arguments = row.program_arguments
    try:
        whatsapp_index = arguments.index("whatsapp")
    except ValueError as exc:
        raise ConsumerMigrationError(
            "WhatsApp replacement lacks authoritative import configuration"
        ) from exc
    if whatsapp_index < 1 or arguments[whatsapp_index - 1] != "import":
        raise ConsumerMigrationError("WhatsApp replacement command is not an admitted import")

    source = arguments[whatsapp_index + 1 :]
    admitted: list[str] = []
    has_all_chats = False
    include_count = 0
    index = 0
    while index < len(source):
        option = source[index]
        if option in _WHATSAPP_FLAG_OPTIONS:
            admitted.append(option)
            has_all_chats = has_all_chats or option == "--all-chats"
            index += 1
            continue
        if option in _WHATSAPP_VALUE_OPTIONS:
            if index + 1 >= len(source) or source[index + 1].startswith("--"):
                raise ConsumerMigrationError(
                    f"WhatsApp authoritative option lacks a value: {option}"
                )
            value = source[index + 1]
            admitted.extend((option, value))
            include_count += option == "--include-chat"
            index += 2
            continue
        raise ConsumerMigrationError(f"WhatsApp authoritative option is not allowed: {option}")
    if has_all_chats == (include_count > 0):
        raise ConsumerMigrationError(
            "WhatsApp replacement requires exactly one authoritative chat scope"
        )
    if "--json" not in admitted:
        admitted.append("--json")
    return (memo_bin, "import", "whatsapp", *admitted)


def _validate_schedule(row: ConsumerInventoryRow, spec: _ReplacementSpec) -> None:
    if spec.schedule_kind == "client":
        return
    if row.start_interval_seconds is not None and row.start_interval_seconds <= 0:
        raise ConsumerMigrationError(f"consumer has an invalid StartInterval: {row.label}")
    if row.throttle_interval_seconds is not None and row.throttle_interval_seconds <= 0:
        raise ConsumerMigrationError(f"consumer has an invalid ThrottleInterval: {row.label}")
    if row.start_calendar_interval != tuple(sorted(set(row.start_calendar_interval))):
        raise ConsumerMigrationError(f"consumer has a noncanonical calendar schedule: {row.label}")
    if any(key not in _CALENDAR_KEYS for key, _value in row.start_calendar_interval):
        raise ConsumerMigrationError(f"consumer has an unsupported calendar schedule: {row.label}")
    if any(not Path(path).is_absolute() for path in row.watch_paths):
        raise ConsumerMigrationError(f"consumer WatchPaths must be absolute: {row.label}")

    has_periodic_trigger = bool(
        row.start_interval_seconds is not None
        or row.start_calendar_interval
        or row.watch_paths
    )
    if spec.schedule_kind == "persistent":
        if not row.keep_alive:
            raise ConsumerMigrationError(
                f"persistent consumer lacks authoritative KeepAlive policy: {row.label}"
            )
    elif not has_periodic_trigger:
        raise ConsumerMigrationError(
            f"periodic consumer lacks authoritative launchd schedule: {row.label}"
        )


def _authorized_mapping(
    manifest: CapabilityManifest,
    spec: _ReplacementSpec,
    old_label: str,
) -> OperationMappingRow:
    matches = tuple(
        row
        for row in manifest.operation_mappings
        if row.source_operation == spec.authority_operation
    )
    if len(matches) != 1:
        raise ConsumerMigrationError(
            f"consumer lacks one admitted operation mapping: {old_label}"
        )
    mapping = matches[0]
    if (
        mapping.disposition not in {"memo_native", "absorb"}
        or not mapping.routes
        or not mapping.parity_tests
        or not mapping.evidence_ids
    ):
        raise ConsumerMigrationError(f"consumer operation mapping is not admitted: {old_label}")
    capabilities = tuple(
        row
        for row in manifest.capabilities
        if mapping.source_operation in row.source_operations
        and any(
            nested.source_operation == mapping.source_operation
            for nested in row.operation_mappings
        )
    )
    if len(capabilities) != 1:
        raise ConsumerMigrationError(
            f"consumer lacks one authoritative capability mapping: {old_label}"
        )
    capability = capabilities[0]
    if (
        not capability.evidence_complete
        or capability.disposition != mapping.disposition
        or not capability.memo_target
    ):
        raise ConsumerMigrationError(f"consumer capability is not admitted: {old_label}")
    return mapping


def _replacement_digest(payload: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _plan_digest(rows: tuple[ConsumerReplacement, ...]) -> str:
    return hashlib.sha256(canonical_json_bytes([row.to_dict() for row in rows])).hexdigest()


def _replacement_payload(
    *,
    old_label: str,
    new_label: str,
    command: tuple[str, ...],
    owner: str,
    manifest_sha256: str,
    mapping: OperationMappingRow,
    row: ConsumerInventoryRow,
) -> dict[str, object]:
    return {
        "old_label": old_label,
        "new_label": new_label,
        "command": list(command),
        "owner": owner,
        "restart_required": True,
        "manifest_sha256": manifest_sha256,
        "authority_operation": mapping.source_operation,
        "run_at_load": row.run_at_load,
        "keep_alive": row.keep_alive,
        "start_interval_seconds": row.start_interval_seconds,
        "start_calendar_interval": [list(item) for item in row.start_calendar_interval],
        "watch_paths": list(row.watch_paths),
        "throttle_interval_seconds": row.throttle_interval_seconds,
    }


def build_consumer_replacement_plan(
    inventory: ConsumerInventory,
    manifest: CapabilityManifest,
    *,
    roster: VerificationRoster,
    memo_bin: Path,
) -> ConsumerReplacementPlan:
    """Build a signed-authority-bound plan for every live retired consumer."""
    try:
        verify_consumer_inventory(inventory, roster=roster)
        verify_capability_manifest(manifest, roster=roster)
    except (InventoryError, ManifestError) as exc:
        raise ConsumerMigrationError("consumer staging authority is invalid") from exc

    stable_memo_bin = _stable_memo_binary(memo_bin)
    active_labels = {
        row.label
        for row in inventory.rows
        if row.kind == "launchd" and row.active
    }
    for row in inventory.rows:
        if row.kind != "process" or not row.active:
            continue
        if not row.correlated_launchd_label:
            raise ConsumerMigrationError(
                f"live retired process requires manual correlation: {row.location}"
            )
        if row.correlated_launchd_label not in active_labels:
            raise ConsumerMigrationError(
                f"live retired process references an inactive LaunchAgent: {row.location}"
            )

    manifest_sha256 = hashlib.sha256(manifest.signed_bytes()).hexdigest()
    replacements: list[ConsumerReplacement] = []
    seen: set[str] = set()
    for row in inventory.rows:
        if row.kind != "launchd" or not row.active:
            continue
        old_label = row.label
        spec = _mapping(old_label)
        if spec is None:
            raise ConsumerMigrationError(
                f"active retired consumer has no Memo-owned replacement: {old_label}"
            )
        if old_label in seen:
            raise ConsumerMigrationError(f"consumer inventory repeats active label: {old_label}")
        seen.add(old_label)
        mapping = _authorized_mapping(manifest, spec, old_label)
        _validate_schedule(row, spec)
        command = (
            spec.command_tail
            if spec.owner == "client"
            else (
                _whatsapp_command(row, stable_memo_bin)
                if spec.whatsapp
                else (stable_memo_bin, *spec.command_tail)
            )
        )
        payload = _replacement_payload(
            old_label=old_label,
            new_label=spec.new_label,
            command=command,
            owner=spec.owner,
            manifest_sha256=manifest_sha256,
            mapping=mapping,
            row=row,
        )
        replacements.append(
            ConsumerReplacement(
                old_label=old_label,
                new_label=spec.new_label,
                command=command,
                owner=spec.owner,
                restart_required=True,
                config_sha256=_replacement_digest(payload),
                rollback_action=(
                    "operator: remove staged Memo config and restore archived original"
                ),
                run_at_load=row.run_at_load if spec.owner == "memo_native" else False,
                keep_alive=row.keep_alive if spec.owner == "memo_native" else False,
                start_interval_seconds=(
                    row.start_interval_seconds if spec.owner == "memo_native" else None
                ),
                start_calendar_interval=(
                    row.start_calendar_interval if spec.owner == "memo_native" else ()
                ),
                watch_paths=row.watch_paths if spec.owner == "memo_native" else (),
                throttle_interval_seconds=(
                    row.throttle_interval_seconds if spec.owner == "memo_native" else None
                ),
            )
        )
    rows = tuple(sorted(replacements, key=lambda replacement: replacement.old_label))
    return ConsumerReplacementPlan(rows=rows, digest=_plan_digest(rows))


def _plist_bytes(row: ConsumerReplacement) -> bytes:
    payload: dict[str, object] = {
        "Label": row.new_label,
        "ProgramArguments": list(row.command),
        "EnvironmentVariables": {"MEMO_NONINTERACTIVE": "1"},
        "RunAtLoad": row.run_at_load,
    }
    if row.keep_alive:
        payload["KeepAlive"] = True
    if row.start_interval_seconds is not None:
        payload["StartInterval"] = row.start_interval_seconds
    if row.start_calendar_interval:
        payload["StartCalendarInterval"] = dict(row.start_calendar_interval)
    if row.watch_paths:
        payload["WatchPaths"] = list(row.watch_paths)
    if row.throttle_interval_seconds is not None:
        payload["ThrottleInterval"] = row.throttle_interval_seconds
    return plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)


def _expected_outputs(plan: ConsumerReplacementPlan) -> dict[str, bytes]:
    if plan.digest != _plan_digest(plan.rows):
        raise ConsumerMigrationError("consumer replacement plan digest is invalid")
    outputs: dict[str, bytes] = {}
    for row in plan.rows:
        if row.owner != "memo_native":
            continue
        name = f"{row.new_label}.plist"
        encoded = _plist_bytes(row)
        previous = outputs.get(name)
        if previous is not None and previous != encoded:
            raise ConsumerMigrationError(f"replacement rows collide on staged label: {name}")
        outputs[name] = encoded
    return outputs


def _staging_root(root: Path) -> Path:
    absolute = Path(os.path.abspath(os.fspath(root)))
    library = Path.home() / "Library"
    if absolute == library or absolute in library.parents or library in absolute.parents:
        raise ConsumerMigrationError("operator staging root overlaps the production Library")
    if absolute.name == _LAUNCHD_DIRECTORY:
        raise ConsumerMigrationError("operator staging root must be a dedicated parent directory")
    return absolute


def _safe_existing_output(directory: SecureDirectory, name: str) -> None:
    try:
        observed = directory.stat(name)
    except FileNotFoundError:
        return
    except ValueError as exc:
        raise ConsumerMigrationError(f"unsafe staged LaunchAgent output: {name}") from exc
    if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        raise ConsumerMigrationError(f"unsafe staged LaunchAgent output: {name}")


def render_memo_launch_agents(
    plan: ConsumerReplacementPlan,
    root: Path,
) -> tuple[Path, ...]:
    """Atomically render the exact plan into a dedicated operator staging root."""
    outputs = _expected_outputs(plan)
    staging_root = _staging_root(root)
    try:
        with open_secure_directory(staging_root, create=True) as directory:
            root_names = set(directory.list_names())
            if root_names - {_LAUNCHD_DIRECTORY}:
                raise ConsumerMigrationError("operator staging root contains stale output")
            directory.ensure_directory(_LAUNCHD_DIRECTORY)
        target = staging_root / _LAUNCHD_DIRECTORY
        with open_secure_directory(target) as directory:
            existing = set(directory.list_names())
            if existing - set(outputs):
                raise ConsumerMigrationError("operator LaunchAgents staging contains stale output")
            for name, encoded in sorted(outputs.items()):
                _safe_existing_output(directory, name)
                directory.atomic_write_bytes(name, encoded, mode=0o644)
    except ConsumerMigrationError:
        raise
    except (OSError, ValueError) as exc:
        raise ConsumerMigrationError("operator staging root is unsafe") from exc
    verify_no_synapse_runtime_reference(staging_root, plan)
    return tuple(staging_root / _LAUNCHD_DIRECTORY / name for name in sorted(outputs))


def verify_no_synapse_runtime_reference(
    root: Path,
    plan: ConsumerReplacementPlan,
) -> None:
    """Verify the exact, regular, plan-derived staged plist set."""
    outputs = _expected_outputs(plan)
    staging_root = _staging_root(root)
    try:
        with open_secure_directory(staging_root) as directory:
            if set(directory.list_names()) != {_LAUNCHD_DIRECTORY}:
                raise ConsumerMigrationError("operator staging root output set is not exact")
        with open_secure_directory(staging_root / _LAUNCHD_DIRECTORY) as directory:
            names = set(directory.list_names())
            if names != set(outputs):
                raise ConsumerMigrationError("staged LaunchAgent output set is not exact")
            for name, expected in outputs.items():
                _safe_existing_output(directory, name)
                encoded = directory.read_bytes(name)
                lowered_name = name.casefold()
                lowered = encoded.casefold()
                if (
                    b"synapse" in lowered
                    or b"memflow" in lowered
                    or "synapse" in lowered_name
                    or "memflow" in lowered_name
                ):
                    raise ConsumerMigrationError(
                        f"retired runtime reference in staged output: {name}"
                    )
                if encoded != expected:
                    raise ConsumerMigrationError(
                        f"staged LaunchAgent does not match its plan: {name}"
                    )
    except ConsumerMigrationError:
        raise
    except (OSError, ValueError) as exc:
        raise ConsumerMigrationError("operator staging output is unsafe") from exc


__all__ = [
    "ConsumerMigrationError",
    "build_consumer_replacement_plan",
    "render_memo_launch_agents",
    "verify_no_synapse_runtime_reference",
]
