"""Preserving Markdown rendering and transactional configuration commits."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tomllib
import uuid
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tomli_w

from memo.config_md import config_dir, config_home, index_path
from memo.errors import ConfigConflictError, ConfigTransactionError
from memo.tui.config.catalog import (
    DOMAIN_TO_FILE,
    PersistencePolicy,
    catalog_by_key,
    domain_file_for_key,
    persistence_policy_for_key,
)

if TYPE_CHECKING:
    from memo.tui.config.session import ValidationIssue


_TOML_BLOCK_RE = re.compile(r"```toml\s*\n(.*?)\n?```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class FileFingerprint:
    exists: bool
    mtime_ns: int
    size: int
    sha256: str


@dataclass(frozen=True)
class SourceSnapshot:
    files: Mapping[Path, FileFingerprint]
    values: Mapping[str, object]
    key_files: Mapping[str, Path]


@dataclass(frozen=True)
class PlannedChange:
    key: str
    before: object | None
    after: object | None
    unset: bool = False


@dataclass(frozen=True)
class ApplyPlan:
    changes: tuple[PlannedChange, ...]
    issues: tuple[ValidationIssue, ...]
    snapshot: SourceSnapshot

    @property
    def blocked(self) -> bool:
        return any(issue.blocking for issue in self.issues)


@dataclass(frozen=True)
class TransactionReceipt:
    transaction_id: str
    state: str
    files: tuple[Path, ...]
    manifest: Path | None = None


@dataclass(frozen=True)
class _SourceView:
    files: dict[Path, FileFingerprint]
    contents: dict[Path, str]
    values: dict[str, object]
    key_files: dict[str, Path]


class _RenderedDraft(dict[Path, str]):
    def __init__(
        self,
        values: Mapping[Path, str],
        *,
        basis: Mapping[Path, FileFingerprint],
        keys_by_path: Mapping[Path, tuple[str, ...]],
    ) -> None:
        super().__init__(values)
        self.basis = dict(basis)
        self.keys_by_path = dict(keys_by_path)


def _missing_fingerprint() -> FileFingerprint:
    return FileFingerprint(False, 0, 0, "")


def _read_source(path: Path) -> tuple[str, FileFingerprint]:
    try:
        payload = path.read_bytes()
        stat = path.stat()
    except FileNotFoundError:
        return "", _missing_fingerprint()
    except OSError as exc:
        raise ConfigTransactionError(f"failed to read configuration source {path}: {exc}") from exc
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigTransactionError(f"configuration source is not UTF-8: {path}") from exc
    return (
        text,
        FileFingerprint(
            True,
            stat.st_mtime_ns,
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        ),
    )


def _flatten(prefix: str, value: object) -> dict[str, object]:
    if isinstance(value, dict):
        flattened: dict[str, object] = {}
        for key, inner in value.items():
            nested = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(nested, inner))
        return flattened
    return {prefix: value}


def _source_paths(env: Mapping[str, str]) -> tuple[list[Path], set[Path]]:
    index = index_path(env)
    root = config_dir(env)
    expected = {root / filename for filename in set(DOMAIN_TO_FILE.values())}
    existing: set[Path] = set()
    if root.is_dir():
        existing = {
            path
            for path in root.iterdir()
            if path.is_file() and path.name in set(DOMAIN_TO_FILE.values())
        }
    ordered = [index, *sorted(expected | existing)]
    return ordered, {index, *expected, *existing}


def _source_view(env: Mapping[str, str], *, strict: bool) -> _SourceView:
    ordered, paths = _source_paths(env)
    contents: dict[Path, str] = {}
    fingerprints: dict[Path, FileFingerprint] = {}
    for path in paths:
        contents[path], fingerprints[path] = _read_source(path)

    values: dict[str, object] = {}
    key_files: dict[str, Path] = {}
    for path in ordered:
        text = contents[path]
        for block_number, block in enumerate(_TOML_BLOCK_RE.findall(text), start=1):
            try:
                parsed = tomllib.loads(block)
            except tomllib.TOMLDecodeError as exc:
                if strict:
                    raise ConfigTransactionError(
                        f"TOML parse error in {path} block {block_number}: {exc}"
                    ) from exc
                continue
            for key, value in _flatten("", parsed).items():
                values[key] = value
                key_files[key] = path
    return _SourceView(fingerprints, contents, values, key_files)


def snapshot_sources(env: Mapping[str, str] | None = None) -> SourceSnapshot:
    source = dict(os.environ if env is None else env)
    view = _source_view(source, strict=False)
    return SourceSnapshot(dict(view.files), dict(view.values), dict(view.key_files))


def _toml_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _toml_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_toml_value(inner) for inner in value]
    return value


def _render_table(
    text: str,
    table: str,
    changes: tuple[PlannedChange, ...],
    *,
    heading: str,
) -> str:
    matches = list(_TOML_BLOCK_RE.finditer(text))
    for match in reversed(matches):
        block = match.group(1)
        try:
            parsed = tomllib.loads(block)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigTransactionError(f"cannot edit malformed TOML: {exc}") from exc
        table_values = parsed.get(table)
        if not isinstance(table_values, dict):
            continue
        values = dict(table_values)
        for change in changes:
            name = change.key.split(".", 1)[1]
            if change.unset:
                values.pop(name, None)
            else:
                values[name] = _toml_value(change.after)

        header_re = re.compile(rf"(?m)^\[{re.escape(table)}\][ \t]*(?:#.*)?$")
        header = header_re.search(block)
        if header is None:
            continue
        next_header = re.search(r"(?m)^\s*\[", block[header.end() :])
        section_end = (
            header.end() + next_header.start() if next_header is not None else len(block)
        )
        suffix = block[section_end:]
        rendered_table = tomli_w.dumps({table: values}).rstrip()
        separator = "\n\n" if suffix else ""
        rendered_block = block[: header.start()] + rendered_table + separator + suffix
        return text[: match.start(1)] + rendered_block + text[match.end(1) :]

    if all(change.unset for change in changes):
        return text
    values = {
        change.key.split(".", 1)[1]: _toml_value(change.after)
        for change in changes
        if not change.unset
    }
    prefix = text
    if not prefix:
        prefix = f"# {heading}\n"
    if not prefix.endswith("\n"):
        prefix += "\n"
    if not prefix.endswith("\n\n"):
        prefix += "\n"
    return prefix + f"```toml\n{tomli_w.dumps({table: values}).rstrip()}\n```\n"


def _changed_externally(change: PlannedChange, view: _SourceView, snapshot: SourceSnapshot) -> bool:
    baseline_exists = change.key in snapshot.values
    current_exists = change.key in view.values
    if baseline_exists != current_exists:
        return True
    if baseline_exists and view.values[change.key] != snapshot.values[change.key]:
        return True
    return baseline_exists and view.key_files.get(change.key) != snapshot.key_files.get(change.key)


def render_draft(
    plan: ApplyPlan, env: Mapping[str, str] | None = None
) -> dict[Path, str]:
    if plan.blocked:
        messages = "; ".join(issue.message for issue in plan.issues if issue.blocking)
        raise ConfigTransactionError(f"configuration review is blocked: {messages}")

    source = dict(os.environ if env is None else env)
    view = _source_view(source, strict=True)
    conflicts = tuple(
        change.key
        for change in plan.changes
        if _changed_externally(change, view, plan.snapshot)
        and not (
            (not change.unset)
            and change.key in view.values
            and view.values[change.key] == change.after
        )
    )
    if conflicts:
        raise ConfigConflictError(tuple(sorted(conflicts)))

    grouped: dict[Path, dict[str, list[PlannedChange]]] = {}
    for change in plan.changes:
        path = (
            view.key_files.get(change.key)
            or plan.snapshot.key_files.get(change.key)
            or config_dir(source) / domain_file_for_key(change.key)
        )
        table = change.key.split(".", 1)[0]
        grouped.setdefault(path, {}).setdefault(table, []).append(change)

    rendered: dict[Path, str] = {}
    keys_by_path: dict[Path, tuple[str, ...]] = {}
    for path, tables in grouped.items():
        text = view.contents.get(path, "")
        keys: list[str] = []
        for table, changes in tables.items():
            typed_changes = tuple(changes)
            text = _render_table(text, table, typed_changes, heading=f"{table.title()} config")
            keys.extend(change.key for change in typed_changes)
        rendered[path] = text
        keys_by_path[path] = tuple(keys)

    basis = {path: view.files.get(path, _missing_fingerprint()) for path in rendered}
    return _RenderedDraft(rendered, basis=basis, keys_by_path=keys_by_path)


def _validate_rendered(path: Path, text: str) -> None:
    from memo.tui.config.session import coerce_value

    catalog = catalog_by_key()
    found_block = False
    for block in _TOML_BLOCK_RE.findall(text):
        found_block = True
        try:
            parsed = tomllib.loads(block)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigTransactionError(f"TOML parse error in staged {path}: {exc}") from exc
        for key, value in _flatten("", parsed).items():
            spec = catalog.get(key)
            if spec is None:
                raise ConfigTransactionError(f"unknown config key in staged {path}: {key}")
            if persistence_policy_for_key(key) is not PersistencePolicy.PERSISTENT:
                raise ConfigTransactionError(f"non-persistent config key in staged {path}: {key}")
            try:
                coerce_value(spec, value)
            except (TypeError, ValueError) as exc:
                raise ConfigTransactionError(f"invalid staged value for {key}: {exc}") from exc
    if not found_block:
        raise ConfigTransactionError(f"staged configuration has no TOML block: {path}")


def _write_manifest(path: Path, manifest: dict[str, Any], state: str) -> None:
    manifest["state"] = state
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class ConfigTransaction:
    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        self.env = dict(os.environ if env is None else env)
        self.home = config_home(self.env)

    def _safe_target(self, path: Path) -> Path:
        target = path.expanduser().resolve()
        if not target.is_relative_to(self.home):
            raise ConfigTransactionError(f"configuration target escapes config home: {path}")
        return target

    def commit(
        self,
        rendered: Mapping[Path, str],
        snapshot: SourceSnapshot,
    ) -> TransactionReceipt:
        transaction_id = uuid.uuid4().hex
        if not rendered:
            return TransactionReceipt(transaction_id, "complete", ())

        normalized = {self._safe_target(path): text for path, text in rendered.items()}
        basis = rendered.basis if isinstance(rendered, _RenderedDraft) else snapshot.files
        keys_by_path = rendered.keys_by_path if isinstance(rendered, _RenderedDraft) else {}
        for path in normalized:
            _, current = _read_source(path)
            expected = basis.get(path, _missing_fingerprint())
            if current != expected:
                keys = keys_by_path.get(path) or (str(path.relative_to(self.home)),)
                raise ConfigConflictError(tuple(sorted(keys)))

        transaction_dir = self.home / ".transactions" / transaction_id
        backup_root = transaction_dir / "backup"
        manifest_path = transaction_dir / "manifest.json"
        transaction_dir.mkdir(parents=True, exist_ok=False)
        staged: dict[Path, Path] = {}
        entries: list[dict[str, object]] = []
        manifest: dict[str, Any] = {"id": transaction_id, "files": entries}

        try:
            for path, text in normalized.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_name(f".{path.name}.memo-tmp-{transaction_id}")
                with temp.open("w", encoding="utf-8") as handle:
                    handle.write(text)
                    handle.flush()
                    os.fsync(handle.fileno())
                staged[path] = temp
                _validate_rendered(path, temp.read_text(encoding="utf-8"))

                relative = path.relative_to(self.home)
                backup = backup_root / relative
                existed = path.exists()
                if existed:
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, backup)
                entries.append(
                    {
                        "target": str(relative),
                        "backup": str(backup.relative_to(transaction_dir)),
                        "temp": str(temp.relative_to(self.home)),
                        "existed": existed,
                    }
                )

            _write_manifest(manifest_path, manifest, "prepared")
            _write_manifest(manifest_path, manifest, "committing")
            for path, temp in staged.items():
                os.replace(temp, path)
            _write_manifest(manifest_path, manifest, "complete")
        except Exception as exc:
            rollback_errors: list[str] = []
            for entry in entries:
                target = self.home / str(entry["target"])
                backup = transaction_dir / str(entry["backup"])
                try:
                    if bool(entry["existed"]):
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(backup, target)
                    else:
                        target.unlink(missing_ok=True)
                except OSError as rollback_exc:
                    rollback_errors.append(f"{target}: {rollback_exc}")
            for temp in staged.values():
                temp.unlink(missing_ok=True)
            state = "rollback_failed" if rollback_errors else "rolled_back"
            with suppress(OSError):
                _write_manifest(manifest_path, manifest, state)
            detail = f"; rollback failed: {'; '.join(rollback_errors)}" if rollback_errors else ""
            raise ConfigTransactionError(f"configuration commit failed: {exc}{detail}") from exc

        from memo.config_md import invalidate_cache

        invalidate_cache()
        return TransactionReceipt(
            transaction_id,
            "complete",
            tuple(sorted(normalized)),
            manifest_path,
        )


def recover_interrupted_transaction(home: Path) -> TransactionReceipt | None:
    resolved_home = home.expanduser().resolve()
    root = resolved_home / ".transactions"
    if not root.is_dir():
        return None
    for manifest_path in sorted(root.glob("*/manifest.json"), reverse=True):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("state") not in {"prepared", "committing", "rollback_failed"}:
            continue
        transaction_dir = manifest_path.parent
        restored: list[Path] = []
        try:
            for entry in manifest.get("files", []):
                target = (resolved_home / str(entry["target"])).resolve()
                if not target.is_relative_to(resolved_home):
                    raise ConfigTransactionError(f"unsafe recovery target: {target}")
                backup = transaction_dir / str(entry["backup"])
                if bool(entry["existed"]):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup, target)
                else:
                    target.unlink(missing_ok=True)
                temp = (resolved_home / str(entry["temp"])).resolve()
                if temp.is_relative_to(resolved_home):
                    temp.unlink(missing_ok=True)
                restored.append(target)
            _write_manifest(manifest_path, manifest, "recovered")
        except (OSError, KeyError, TypeError) as exc:
            raise ConfigTransactionError(f"failed to recover {manifest_path}: {exc}") from exc

        from memo.config_md import invalidate_cache

        invalidate_cache()
        return TransactionReceipt(
            str(manifest.get("id", transaction_dir.name)),
            "recovered",
            tuple(restored),
            manifest_path,
        )
    return None


def restore_transaction_backup(manifest_path: Path) -> TransactionReceipt:
    """Restore every original recorded by a selected transaction manifest."""
    manifest_path = manifest_path.expanduser().resolve()
    transaction_dir = manifest_path.parent
    home = transaction_dir.parent.parent.resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigTransactionError(f"failed to read backup manifest {manifest_path}: {exc}") from exc

    restored: list[Path] = []
    try:
        for entry in manifest.get("files", []):
            target = (home / str(entry["target"])).resolve()
            if not target.is_relative_to(home):
                raise ConfigTransactionError(f"unsafe backup target: {target}")
            backup = transaction_dir / str(entry["backup"])
            if bool(entry["existed"]):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, target)
            else:
                target.unlink(missing_ok=True)
            restored.append(target)
        manifest["restored_from_state"] = manifest.get("state")
        _write_manifest(manifest_path, manifest, "restored")
    except (OSError, KeyError, TypeError) as exc:
        raise ConfigTransactionError(f"failed to restore {manifest_path}: {exc}") from exc

    from memo.config_md import invalidate_cache

    invalidate_cache()
    return TransactionReceipt(
        str(manifest.get("id", transaction_dir.name)),
        "restored",
        tuple(restored),
        manifest_path,
    )


__all__ = [
    "ApplyPlan",
    "ConfigTransaction",
    "FileFingerprint",
    "PlannedChange",
    "SourceSnapshot",
    "TransactionReceipt",
    "recover_interrupted_transaction",
    "render_draft",
    "restore_transaction_backup",
    "snapshot_sources",
]
