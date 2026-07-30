"""Operator-only staging for retired Synapse consumer replacements.

This module deliberately writes *only* beneath the caller-provided staging
root.  Applying, unloading, or deleting a real LaunchAgent remains an explicit
operator cutover action.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path
from xml.sax.saxutils import escape

from memo.operational_event import canonical_json_bytes
from tools.memflow_absorption.schemas import (
    CapabilityManifest,
    ConsumerInventory,
    ConsumerInventoryRow,
    ConsumerReplacement,
    ConsumerReplacementPlan,
)


class ConsumerMigrationError(RuntimeError):
    """The observed consumers cannot be safely staged for replacement."""


_CLIENT_ROUTE_TOKENS = ("dashboard", "gateway", "mcp")
_LAUNCHD_DIRECTORY = "LaunchAgents"
_SAFE_LABEL = re.compile(r"[^a-z0-9.-]+")


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
            raise ConsumerMigrationError(
                f"operator staging root contains a symlink component: {current}"
            )


def _row_label(row: ConsumerInventoryRow) -> str:
    """Use structured inventory data, with a safe legacy-inventory fallback."""
    if row.label:
        return row.label
    if row.kind == "launchd":
        return row.location.partition(":")[0]
    return row.location


def _row_is_active(row: ConsumerInventoryRow) -> bool:
    # Inventory rows made before ``active`` existed are compatible and treated
    # as active.  A newly captured unloaded LaunchAgent is explicitly false.
    return row.active


def _mapping(label: str) -> tuple[str, tuple[str, ...], str, bool] | None:
    """Return ``new_label, command, owner, restart_required`` for one label."""
    normalized = label.casefold()
    if any(token in normalized for token in _CLIENT_ROUTE_TOKENS):
        return ("client.close-reconnect", ("client-close-reconnect",), "client", True)
    if "whatsapp" in normalized:
        return (
            "com.memo.import-whatsapp",
            ("memo", "import", "whatsapp"),
            "memo_native",
            False,
        )
    if "watcher" in normalized or normalized.endswith(".watch"):
        return ("com.memo.watch", ("memo", "watch"), "memo_native", True)
    if "recall" in normalized and "daemon" in normalized:
        return (
            "com.memo.recall-daemon",
            ("memo", "recall-daemon", "_serve"),
            "memo_native",
            True,
        )
    if "digest" in normalized:
        return ("com.memo.digest", ("memo", "digest"), "memo_native", False)
    if "nightly" in normalized or "dream" in normalized:
        return ("com.memo.dream", ("memo", "dream", "run"), "memo_native", False)
    if "vault" in normalized or "archive" in normalized:
        return ("com.memo.reindex", ("memo", "reindex"), "memo_native", False)
    return None


def _replacement_digest(
    *,
    old_label: str,
    new_label: str,
    command: tuple[str, ...],
    owner: str,
    restart_required: bool,
    manifest_sha256: str,
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "old_label": old_label,
                "new_label": new_label,
                "command": list(command),
                "owner": owner,
                "restart_required": restart_required,
                "manifest_sha256": manifest_sha256,
            }
        )
    ).hexdigest()


def _plan_digest(rows: tuple[ConsumerReplacement, ...]) -> str:
    return hashlib.sha256(canonical_json_bytes([row.to_dict() for row in rows])).hexdigest()


def build_consumer_replacement_plan(
    inventory: ConsumerInventory, manifest: CapabilityManifest
) -> ConsumerReplacementPlan:
    """Map every active, observed retired consumer to its Memo-owned target.

    The capability manifest is bound into every configuration digest.  This
    keeps a plan reviewable against the exact admitted capability evidence
    without attempting signature verification here (the verifier roster is an
    operator-only input and is intentionally absent from this interface).
    """
    if inventory.blockers:
        raise ConsumerMigrationError("consumer inventory has unresolved blockers")
    if manifest.blockers or not manifest.frozen:
        raise ConsumerMigrationError("capability manifest is not ready for consumer staging")

    manifest_sha256 = hashlib.sha256(manifest.signed_bytes()).hexdigest()
    replacements: list[ConsumerReplacement] = []
    seen: set[str] = set()
    for row in inventory.rows:
        # Source and process observations prove that the retired runtime is
        # still referenced, but they do not contain enough launchd scheduling
        # policy to render a safe replacement.  Runnable consumer rows are the
        # active LaunchAgents captured by the signed inventory.
        if row.kind != "launchd":
            continue
        if not _row_is_active(row):
            continue
        old_label = _row_label(row)
        mapped = _mapping(old_label)
        if mapped is None:
            # A consumer inventory is already narrowed to retired-runtime
            # references.  Silently omitting one here would make a cutover
            # plan look complete while leaving a live consumer behind.
            raise ConsumerMigrationError(
                f"active retired consumer has no Memo-owned replacement: {old_label}"
            )
        if old_label in seen:
            raise ConsumerMigrationError(f"consumer inventory repeats active label: {old_label}")
        seen.add(old_label)
        new_label, command, owner, restart_required = mapped
        config_sha256 = _replacement_digest(
            old_label=old_label,
            new_label=new_label,
            command=command,
            owner=owner,
            restart_required=restart_required,
            manifest_sha256=manifest_sha256,
        )
        replacements.append(
            ConsumerReplacement(
                old_label=old_label,
                new_label=new_label,
                command=command,
                owner=owner,
                restart_required=restart_required,
                config_sha256=config_sha256,
                rollback_action=(
                    "operator: remove the staged Memo LaunchAgent and restore the archived original"
                ),
            )
        )
    rows = tuple(sorted(replacements, key=lambda row: row.old_label))
    return ConsumerReplacementPlan(rows=rows, digest=_plan_digest(rows))


def _plist_label(row: ConsumerReplacement) -> str:
    safe = _SAFE_LABEL.sub("-", row.new_label.casefold()).strip(".-")
    if not safe:
        raise ConsumerMigrationError(f"replacement has an unsafe staged label: {row.new_label!r}")
    return safe


def _render_launch_agent(row: ConsumerReplacement) -> str:
    arguments = "\n".join(f"        <string>{escape(item)}</string>" for item in row.command)
    label = _plist_label(row)
    keep_alive = """    <key>KeepAlive</key>
    <true/>
""" if row.restart_required else ""
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{escape(label)}</string>
    <key>ProgramArguments</key>
    <array>
{arguments}
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>MEMO_NONINTERACTIVE</key>
        <string>1</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
{keep_alive}
</dict>
</plist>
'''


def render_memo_launch_agents(plan: ConsumerReplacementPlan, root: Path) -> tuple[Path, ...]:
    """Render Memo LaunchAgents under an operator-owned staging root only."""
    expected_digest = _plan_digest(plan.rows)
    if plan.digest != expected_digest:
        raise ConsumerMigrationError("consumer replacement plan digest is invalid")
    staging_root = Path(root)
    _reject_symlink_components(staging_root)
    if staging_root.exists() and not staging_root.is_dir():
        raise ConsumerMigrationError("operator staging root must be a directory")
    target = staging_root / _LAUNCHD_DIRECTORY
    if target.exists() and target.is_symlink():
        raise ConsumerMigrationError("operator LaunchAgents staging directory must not be a symlink")
    target.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    rendered_labels: dict[str, ConsumerReplacement] = {}
    for row in plan.rows:
        # Dashboard/gateway/MCP consumers require the client to close and
        # reconnect; they are intentionally not replaced by a compatibility
        # daemon or a launchd service.
        if row.owner != "memo_native":
            continue
        label = _plist_label(row)
        previous = rendered_labels.get(label)
        if previous is not None:
            if _render_launch_agent(previous) != _render_launch_agent(row):
                raise ConsumerMigrationError(f"replacement rows collide on staged label: {label}")
            # Several retired schedules may collapse into the same native Memo
            # maintenance job.  Render that identical target once while
            # retaining one plan row (and rollback action) per retired label.
            continue
        rendered_labels[label] = row
        path = target / f"{label}.plist"
        if path.is_symlink():
            raise ConsumerMigrationError(f"staged LaunchAgent path must not be a symlink: {path}")
        path.write_text(_render_launch_agent(row), encoding="utf-8")
        paths.append(path)
    verify_no_synapse_runtime_reference(staging_root, plan)
    return tuple(paths)


def verify_no_synapse_runtime_reference(root: Path, plan: ConsumerReplacementPlan) -> None:
    """Fail closed if staged runtime output names a retired runtime or path."""
    staging_root = Path(root)
    if not staging_root.is_dir():
        raise ConsumerMigrationError(f"operator staging root is unavailable: {staging_root}")
    for directory, directory_names, file_names in os.walk(staging_root, followlinks=False):
        parent = Path(directory)
        if any((parent / name).is_symlink() for name in directory_names + file_names):
            raise ConsumerMigrationError("operator staging output contains a symlink")
        for name in file_names:
            path = parent / name
            lowered_name = name.casefold()
            if "synapse" in lowered_name or "memflow" in lowered_name:
                raise ConsumerMigrationError(f"retired runtime path in staged output: {path}")
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise ConsumerMigrationError(f"operator staging output is not UTF-8: {path}") from exc
            if "synapse" in text.casefold() or "memflow" in text.casefold():
                raise ConsumerMigrationError(f"retired runtime reference in staged output: {path}")
    if plan.digest != _plan_digest(plan.rows):
        raise ConsumerMigrationError("consumer replacement plan digest is invalid")
    target = staging_root / _LAUNCHD_DIRECTORY
    for row in plan.rows:
        if row.owner != "memo_native":
            continue
        path = target / f"{_plist_label(row)}.plist"
        if not path.is_file() or path.is_symlink():
            raise ConsumerMigrationError(f"staged LaunchAgent is missing: {path}")
        if path.read_text(encoding="utf-8") != _render_launch_agent(row):
            raise ConsumerMigrationError(f"staged LaunchAgent does not match its plan: {path}")


__all__ = [
    "ConsumerMigrationError",
    "build_consumer_replacement_plan",
    "render_memo_launch_agents",
    "verify_no_synapse_runtime_reference",
]
