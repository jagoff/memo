"""Authority-gated, operator-only staging for retired Synapse consumers."""

from __future__ import annotations

import hashlib
import os
import plistlib
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
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
    authority_cli: str
    schedule_kind: str
    whatsapp: bool = False


@dataclass(frozen=True)
class _AuthorityBinding:
    mapping: OperationMappingRow
    route_id: str
    memo_target: str


# Exact labels are part of the reviewed cutover contract. A substring match can
# silently bless a newly introduced service, so unknown labels always block.
_REPLACEMENTS: dict[str, _ReplacementSpec] = {
    "com.synapse.whatsapp-ingest": _ReplacementSpec(
        "com.memo.import-whatsapp",
        (),
        "memo_native",
        "synapse.whatsapp_live.message",
        "memo import whatsapp",
        "periodic",
        whatsapp=True,
    ),
    "com.synapse.watcher": _ReplacementSpec(
        "com.memo.watch",
        ("watch",),
        "memo_native",
        "synapse.watcher.event",
        "memo watch",
        "persistent",
    ),
    "com.synapse.memo-recall-daemon": _ReplacementSpec(
        "com.memo.recall-daemon",
        ("recall-daemon", "_serve"),
        "memo_native",
        "synapse.cli.ops",
        "memo recall-daemon _serve",
        "persistent",
    ),
    "com.synapse.memo-nightly": _ReplacementSpec(
        "com.memo.dream-nightly",
        ("dream", "run"),
        "memo_native",
        "synapse.cli.ops",
        "memo dream run",
        "periodic",
    ),
    "com.synapse.morning-digest": _ReplacementSpec(
        "com.memo.digest",
        ("digest",),
        "memo_native",
        "synapse.morning_digest.run",
        "memo digest",
        "periodic",
    ),
    "com.synapse.dream-synthesis": _ReplacementSpec(
        "com.memo.dream-synthesis",
        ("dream", "run"),
        "memo_native",
        "synapse.cli.ops",
        "memo dream run",
        "periodic",
    ),
    "com.synapse.vault-ingest": _ReplacementSpec(
        "com.memo.reindex-vault",
        ("reindex",),
        "memo_native",
        "synapse.cli.ops",
        "memo reindex",
        "periodic",
    ),
    "com.synapse.vault-archive": _ReplacementSpec(
        "com.memo.reindex-archive",
        ("reindex",),
        "memo_native",
        "synapse.vault_archive.move",
        "memo reindex",
        "periodic",
    ),
    "com.synapse.dashboard": _ReplacementSpec(
        "client.close-reconnect",
        ("client-close-reconnect",),
        "client",
        "synapse.chat.ask",
        "client-close-reconnect",
        "client",
    ),
    "com.synapse.dashboard-relay": _ReplacementSpec(
        "client.close-reconnect",
        ("client-close-reconnect",),
        "client",
        "synapse.chat.ask",
        "client-close-reconnect",
        "client",
    ),
    "com.synapse.gateway": _ReplacementSpec(
        "client.close-reconnect",
        ("client-close-reconnect",),
        "client",
        "synapse.federate.query",
        "client-close-reconnect",
        "client",
    ),
    "com.synapse.mcp": _ReplacementSpec(
        "client.close-reconnect",
        ("client-close-reconnect",),
        "client",
        "synapse.federate.query",
        "client-close-reconnect",
        "client",
    ),
    "com.synapse.memo-daemon": _ReplacementSpec(
        "client.close-reconnect",
        ("client-close-reconnect",),
        "client",
        "synapse.chat.ask",
        "client-close-reconnect",
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
_WHATSAPP_SINGLE_VALUE_OPTIONS = {
    "--retention-days",
    "--since",
    "--notes-dir",
    "--db",
}
_CALENDAR_KEYS = {"Month", "Day", "Weekday", "Hour", "Minute", "Second"}
_CALENDAR_RANGES = {
    "Month": (1, 12),
    "Day": (1, 31),
    "Weekday": (0, 7),
    "Hour": (0, 23),
    "Minute": (0, 59),
    "Second": (0, 59),
}


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
    if resolved.name != "memo":
        raise ConsumerMigrationError(
            "Memo runtime binary must be named memo for nested isolated commands"
        )
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
    singletons: set[str] = set()
    index = 0
    while index < len(source):
        option = source[index]
        if option in _WHATSAPP_FLAG_OPTIONS:
            if option in singletons:
                raise ConsumerMigrationError(
                    f"WhatsApp authoritative option is repeated: {option}"
                )
            singletons.add(option)
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
            if option in _WHATSAPP_SINGLE_VALUE_OPTIONS:
                if option in singletons:
                    raise ConsumerMigrationError(
                        f"WhatsApp authoritative option is repeated: {option}"
                    )
                singletons.add(option)
            if not value or "\x00" in value:
                raise ConsumerMigrationError(
                    f"WhatsApp authoritative option has an invalid value: {option}"
                )
            if option in {"--db", "--notes-dir"} and not Path(value).is_absolute():
                raise ConsumerMigrationError(
                    f"WhatsApp authoritative path must be absolute: {option}"
                )
            if option == "--retention-days":
                try:
                    retention_days = int(value)
                except ValueError as exc:
                    raise ConsumerMigrationError(
                        "WhatsApp authoritative retention must be an integer"
                    ) from exc
                if str(retention_days) != value or not 1 <= retention_days <= 36_500:
                    raise ConsumerMigrationError(
                        "WhatsApp authoritative retention is invalid"
                    )
            if option == "--since":
                try:
                    parsed = date.fromisoformat(value)
                except ValueError as exc:
                    raise ConsumerMigrationError(
                        "WhatsApp authoritative since date is invalid"
                    ) from exc
                if parsed.isoformat() != value:
                    raise ConsumerMigrationError(
                        "WhatsApp authoritative since date is invalid"
                    )
            admitted.extend((option, value))
            include_count += option == "--include-chat"
            index += 2
            continue
        raise ConsumerMigrationError(f"WhatsApp authoritative option is not allowed: {option}")
    if has_all_chats == (include_count > 0):
        raise ConsumerMigrationError(
            "WhatsApp replacement requires exactly one authoritative chat scope"
        )
    if {"--index", "--no-index"}.issubset(singletons):
        raise ConsumerMigrationError("WhatsApp authoritative index policy conflicts")
    if "--json" not in admitted:
        admitted.append("--json")
    return (memo_bin, "import", "whatsapp", *admitted)


def _validated_keep_alive(
    keep_alive: bool | Mapping[str, object],
    *,
    label: str,
) -> bool | dict[str, object]:
    if isinstance(keep_alive, bool):
        return keep_alive
    if not isinstance(keep_alive, Mapping) or not keep_alive:
        raise ConsumerMigrationError(f"consumer has an invalid KeepAlive policy: {label}")
    bool_keys = {"SuccessfulExit", "NetworkState", "Crashed", "AfterInitialDemand"}
    state_keys = {"PathState", "OtherJobEnabled"}
    if any(
        not isinstance(key, str) or key not in bool_keys | state_keys
        for key in keep_alive
    ):
        raise ConsumerMigrationError(f"consumer has an invalid KeepAlive policy: {label}")
    normalized: dict[str, object] = {}
    for key in sorted(keep_alive):
        value = keep_alive[key]
        if key in bool_keys:
            if not isinstance(value, bool):
                raise ConsumerMigrationError(
                    f"consumer has an invalid KeepAlive policy: {label}"
                )
            normalized[key] = value
            continue
        if not isinstance(value, Mapping) or not value:
            raise ConsumerMigrationError(f"consumer has an invalid KeepAlive policy: {label}")
        if any(
            not isinstance(name, str)
            or not name
            or "\x00" in name
            or not isinstance(expected, bool)
            or (key == "PathState" and not Path(name).is_absolute())
            or (
                key == "OtherJobEnabled"
                and (not name.strip() or "/" in name)
            )
            for name, expected in value.items()
        ):
            raise ConsumerMigrationError(f"consumer has an invalid KeepAlive policy: {label}")
        normalized[key] = dict(sorted(value.items()))
    return normalized


def _validated_calendars(
    calendars: tuple[tuple[tuple[str, int], ...], ...],
    *,
    label: str,
) -> tuple[tuple[tuple[str, int], ...], ...]:
    normalized: list[tuple[tuple[str, int], ...]] = []
    for calendar in calendars:
        if not calendar or calendar != tuple(sorted(calendar)):
            raise ConsumerMigrationError(
                f"consumer has a noncanonical calendar schedule: {label}"
            )
        if len({key for key, _value in calendar}) != len(calendar):
            raise ConsumerMigrationError(
                f"consumer has a noncanonical calendar schedule: {label}"
            )
        if any(key not in _CALENDAR_KEYS for key, _value in calendar):
            raise ConsumerMigrationError(
                f"consumer has an unsupported calendar schedule: {label}"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not (_CALENDAR_RANGES[key][0] <= value <= _CALENDAR_RANGES[key][1])
            for key, value in calendar
        ):
            raise ConsumerMigrationError(
                f"consumer has an invalid calendar schedule: {label}"
            )
        normalized.append(calendar)
    result = tuple(normalized)
    if result != tuple(sorted(set(result))):
        raise ConsumerMigrationError(f"consumer has a noncanonical calendar schedule: {label}")
    return result


def _validate_schedule(row: ConsumerInventoryRow, spec: _ReplacementSpec) -> None:
    if spec.schedule_kind == "client":
        return
    if row.start_interval_seconds is not None and row.start_interval_seconds <= 0:
        raise ConsumerMigrationError(f"consumer has an invalid StartInterval: {row.label}")
    if row.throttle_interval_seconds is not None and row.throttle_interval_seconds <= 0:
        raise ConsumerMigrationError(f"consumer has an invalid ThrottleInterval: {row.label}")
    keep_alive = _validated_keep_alive(row.keep_alive, label=row.label)
    _validated_calendars(row.start_calendar_interval, label=row.label)
    if any(not Path(path).is_absolute() for path in row.watch_paths):
        raise ConsumerMigrationError(f"consumer WatchPaths must be absolute: {row.label}")
    if any(
        retired in path.casefold()
        for path in row.watch_paths
        for retired in ("synapse", "memflow")
    ):
        raise ConsumerMigrationError(f"consumer WatchPaths retain a retired runtime: {row.label}")

    has_periodic_trigger = bool(
        row.start_interval_seconds is not None
        or row.start_calendar_interval
        or row.watch_paths
    )
    if spec.schedule_kind == "persistent":
        if keep_alive is False:
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
) -> _AuthorityBinding:
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
        if row.name == mapping.capability
        and mapping.source_operation in row.source_operations
        and any(
            nested == mapping
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
    route_matches = tuple(
        route for route in mapping.routes if route.memo_cli == (spec.authority_cli,)
    )
    if len(route_matches) != 1:
        raise ConsumerMigrationError(
            f"consumer lacks one exact admitted CLI route: {old_label}"
        )
    route = route_matches[0]
    route_targets = {
        method
        for method in route.memo_methods
    }
    expected_capability_targets = {
        method
        for nested in capability.operation_mappings
        for candidate in nested.routes
        for method in candidate.memo_methods
    }
    capability_targets = {
        target.strip()
        for target in capability.memo_target.split(",")
        if target.strip()
    }
    if (
        not route.memo_methods
        or not route_targets.issubset(capability_targets)
        or capability_targets != expected_capability_targets
    ):
        raise ConsumerMigrationError(
            f"consumer CLI route target is not exactly admitted: {old_label}"
        )
    return _AuthorityBinding(
        mapping=mapping,
        route_id=route.route_id,
        memo_target=",".join(sorted(route.memo_methods)),
    )


def _memo_environment(
    row: ConsumerInventoryRow,
    *,
    memo_bin: str,
) -> tuple[tuple[str, str], ...]:
    source: dict[str, str] = {}
    for key, value in row.environment:
        if (
            not isinstance(key, str)
            or not isinstance(value, str)
            or not key
            or key in source
            or "\x00" in key
            or "\x00" in value
        ):
            raise ConsumerMigrationError(
                f"consumer has an invalid authoritative environment: {row.label}"
            )
        source[key] = value
    retained_keys: set[str] = set()
    for key in row.environment_keys:
        if not isinstance(key, str) or not key or "\x00" in key:
            raise ConsumerMigrationError(
                f"consumer lacks exact authoritative environment values: {row.label}"
            )
        if key == "PATH" or key.startswith("MEMO_"):
            retained_keys.add(key)
    if not retained_keys.issubset(source):
        raise ConsumerMigrationError(
            f"consumer lacks exact authoritative environment values: {row.label}"
        )
    retained = {
        key: value
        for key, value in source.items()
        if key.startswith("MEMO_")
    }
    retained["MEMO_NONINTERACTIVE"] = "1"
    existing_path = source.get("PATH", "")
    path_entries = existing_path.split(":") if existing_path else []
    if any(
        not entry
        or not Path(entry).is_absolute()
        or any(retired in entry.casefold() for retired in ("synapse", "memflow"))
        for entry in path_entries
    ):
        raise ConsumerMigrationError(
            f"consumer has an invalid authoritative PATH: {row.label}"
        )
    memo_parent = str(Path(memo_bin).parent)
    retained["PATH"] = ":".join(
        (memo_parent, *(entry for entry in path_entries if entry != memo_parent))
    )
    if any(
        retired in value.casefold()
        for key, value in retained.items()
        if key != "PATH"
        for retired in ("synapse", "memflow")
    ):
        raise ConsumerMigrationError(
            f"consumer environment retains a retired runtime: {row.label}"
        )
    return tuple(sorted(retained.items()))


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
    binding: _AuthorityBinding,
    row: ConsumerInventoryRow,
    environment: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    return {
        "old_label": old_label,
        "new_label": new_label,
        "command": list(command),
        "owner": owner,
        "restart_required": True,
        "manifest_sha256": manifest_sha256,
        "authority_operation": binding.mapping.source_operation,
        "authority_route_id": binding.route_id,
        "authority_memo_target": binding.memo_target,
        "run_at_load": row.run_at_load,
        "keep_alive": (
            dict(row.keep_alive) if isinstance(row.keep_alive, Mapping) else row.keep_alive
        ),
        "start_interval_seconds": row.start_interval_seconds,
        "start_calendar_interval": [
            {key: value for key, value in calendar}
            for calendar in row.start_calendar_interval
        ],
        "watch_paths": list(row.watch_paths),
        "throttle_interval_seconds": row.throttle_interval_seconds,
        "environment": [list(item) for item in environment],
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
    required_surfaces = {
        "process",
        "port",
        "launchagent",
        "mcp_gateway_route",
        "shell_config_path",
        "state_root",
    }
    observations = {
        key: tuple(values)
        for key, values in inventory.surface_observations.items()
    }
    if (
        set(observations) != required_surfaces
        or any(
            not values
            or values != tuple(sorted(set(values)))
            or any(not value for value in values)
            for values in observations.values()
        )
    ):
        raise ConsumerMigrationError(
            "signed consumer inventory lacks complete Synapse surface observations"
        )

    stable_memo_bin = _stable_memo_binary(memo_bin)
    active_jobs = {
        row.label: row
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
        launchd_row = active_jobs.get(row.correlated_launchd_label)
        if launchd_row is None:
            raise ConsumerMigrationError(
                f"live retired process references an inactive LaunchAgent: {row.location}"
            )
        process_commands = {row.program_arguments, row.program_arguments[1:]}
        if launchd_row.program_arguments not in process_commands:
            raise ConsumerMigrationError(
                f"live retired process correlation is not exact: {row.location}"
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
        binding = _authorized_mapping(manifest, spec, old_label)
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
        generated_cli = (
            " ".join(command)
            if spec.owner == "client"
            else "memo " + " ".join(command[1 : 3 if spec.whatsapp else None])
        )
        if generated_cli != spec.authority_cli:
            raise ConsumerMigrationError(
                f"replacement command does not match admitted CLI route: {old_label}"
            )
        environment = (
            _memo_environment(row, memo_bin=stable_memo_bin)
            if spec.owner == "memo_native"
            else ()
        )
        if spec.owner == "memo_native" and any(
            retired in argument.casefold()
            for argument in command
            for retired in ("synapse", "memflow")
        ):
            raise ConsumerMigrationError(
                f"Memo replacement command retains a retired runtime: {old_label}"
            )
        payload = _replacement_payload(
            old_label=old_label,
            new_label=spec.new_label,
            command=command,
            owner=spec.owner,
            manifest_sha256=manifest_sha256,
            binding=binding,
            row=row,
            environment=environment,
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
                environment=environment,
            )
        )
    rows = tuple(sorted(replacements, key=lambda replacement: replacement.old_label))
    return ConsumerReplacementPlan(
        rows=rows,
        digest=_plan_digest(rows),
        covered_surfaces=observations,
        inventory_sha256=hashlib.sha256(inventory.signed_bytes()).hexdigest(),
        capability_manifest_sha256=hashlib.sha256(manifest.signed_bytes()).hexdigest(),
    )


def _plist_bytes(row: ConsumerReplacement) -> bytes:
    payload: dict[str, object] = {
        "Label": row.new_label,
        "ProgramArguments": list(row.command),
        "EnvironmentVariables": dict(row.environment),
        "RunAtLoad": row.run_at_load,
        "KeepAlive": (
            dict(row.keep_alive)
            if isinstance(row.keep_alive, Mapping)
            else row.keep_alive
        ),
    }
    if row.start_interval_seconds is not None:
        payload["StartInterval"] = row.start_interval_seconds
    if row.start_calendar_interval:
        calendars = [dict(calendar) for calendar in row.start_calendar_interval]
        payload["StartCalendarInterval"] = (
            calendars[0] if len(calendars) == 1 else calendars
        )
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
    protected = {
        Path("/Library"),
        Path("/System/Library"),
        Path("/var/root/Library"),
        Path.home() / "Library",
    }
    parts = absolute.parts
    casefolded_parts = tuple(part.casefold() for part in parts)
    if (
        len(parts) == 3
        and casefolded_parts[1] in {"users", "home"}
        and parts[2]
    ):
        protected.add(absolute / "Library")
    if casefolded_parts == ("/", "var", "root"):
        protected.add(absolute / "Library")
    for index, part in enumerate(parts):
        if (
            part.casefold() == "library"
            and index >= 2
            and casefolded_parts[index - 2] in {"users", "home"}
        ):
            protected.add(Path(*parts[: index + 1]))
        if (
            part.casefold() == "library"
            and index >= 1
            and casefolded_parts[index - 1] == "root"
        ):
            protected.add(Path(*parts[: index + 1]))

    def _casefold_parts(path: Path) -> tuple[str, ...]:
        return tuple(part.casefold() for part in path.parts)

    def _overlaps(left: Path, right: Path) -> bool:
        left_parts = _casefold_parts(left)
        right_parts = _casefold_parts(right)
        shortest = min(len(left_parts), len(right_parts))
        return left_parts[:shortest] == right_parts[:shortest]

    try:
        canonical = absolute.resolve(strict=False)
        protected_identities = tuple(
            (library, library.resolve(strict=False))
            for library in protected
        )
    except OSError as exc:
        raise ConsumerMigrationError("operator staging root identity is unsafe") from exc

    def _identity(
        path: Path,
        *,
        fail_closed: bool,
    ) -> tuple[int, int] | None:
        try:
            observed = path.stat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            if not fail_closed:
                return None
            raise ConsumerMigrationError(
                "operator staging root identity is unsafe"
            ) from exc
        return observed.st_dev, observed.st_ino

    candidate_prefixes = (absolute, *absolute.parents[:-1])
    if any(
        _overlaps(absolute, library)
        or _overlaps(canonical, canonical_library)
        or (
            (library_identity := _identity(library, fail_closed=False)) is not None
            and any(
                _identity(prefix, fail_closed=True) == library_identity
                for prefix in candidate_prefixes
            )
        )
        or (
            (absolute_identity := _identity(absolute, fail_closed=True)) is not None
            and any(
                _identity(prefix, fail_closed=False) == absolute_identity
                for prefix in (library, *library.parents[:-1])
            )
        )
        for library, canonical_library in protected_identities
    ):
        raise ConsumerMigrationError("operator staging root overlaps the production Library")
    if absolute.name.casefold() == _LAUNCHD_DIRECTORY.casefold():
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
            for name in sorted(outputs):
                _safe_existing_output(directory, name)
            for name, encoded in sorted(outputs.items()):
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
                lowered = encoded.lower()
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
