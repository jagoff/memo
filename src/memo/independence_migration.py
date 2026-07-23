"""One-way migration from legacy integration metadata into Memo 4 contracts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import frontmatter

from memo.atomic_io import atomic_write_text
from memo.contracts import LEGACY_PROVENANCE_KEYS, PROVENANCE_KEYS, normalize_provenance

_REMOVED_FLAGS = {
    "MEMO_BRIEFING_SYNAPSE_DISABLE",
    "MEMO_EMIT_LEDGER",
    "MEMO_EMIT_RECEIPTS",
    "MEMO_MEMFLOW_BIN",
    "MEMO_RESPECT_SYNAPSE_FREEZE",
    "MEMO_SYNC_MEMFLOW_ENABLED",
    "MEMO_SYNAPSE_CLIENT_TIMEOUT",
    "MEMO_SYNAPSE_EXECUTABLE",
}


def _migrate_markdown(path: Path, *, write: bool) -> str:
    try:
        original = path.read_text(encoding="utf-8")
        post = frontmatter.loads(original)
    except (OSError, ValueError, TypeError):
        return "error"
    raw_extra = post.metadata.get("extra")
    if not isinstance(raw_extra, dict):
        return "unchanged"
    nested = raw_extra.get("provenance")
    nested_legacy = LEGACY_PROVENANCE_KEYS & nested.keys() if isinstance(nested, dict) else set()
    legacy = LEGACY_PROVENANCE_KEYS & raw_extra.keys()
    flat_native = PROVENANCE_KEYS & raw_extra.keys()
    if not legacy and not nested_legacy and not flat_native:
        return "unchanged"
    extra = dict(raw_extra)
    provenance = normalize_provenance(extra, preserve_legacy=True)
    for key in LEGACY_PROVENANCE_KEYS | PROVENANCE_KEYS:
        extra.pop(key, None)
    if provenance:
        extra["provenance"] = provenance
    post.metadata["extra"] = extra
    rendered = frontmatter.dumps(post)
    if write and rendered != original:
        atomic_write_text(path, rendered)
    return "migrated"


def _migrate_config(path: Path, *, write: bool) -> str:
    try:
        original = path.read_text(encoding="utf-8")
    except OSError:
        return "error"
    lines: list[str] = []
    changed = False
    for line in original.splitlines(keepends=True):
        if any(re.search(rf"\b{re.escape(flag)}\b", line) for flag in _REMOVED_FLAGS):
            changed = True
            continue
        updated = re.sub(
            r"""(MEMO_CACHE_BACKEND\s*(?:=|:)\s*)(["']?)memflow\b\2""",
            r"\1\2vault\2",
            line,
            flags=re.IGNORECASE,
        )
        changed = changed or updated != line
        lines.append(updated)
    if not changed:
        return "unchanged"
    if write:
        atomic_write_text(path, "".join(lines))
    return "migrated"


def _import_legacy_ledger(memory: Any, path: Path, *, write: bool) -> dict[str, int]:
    report = {"checked": 0, "imported": 0, "errors": 0, "skipped": 0}
    if not path.is_file():
        return report
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    source_id = _legacy_source_id(path)
    stamp_path = memory.cfg.state_dir / "independence-migration.json"
    stamp = _read_migration_stamp(stamp_path)
    if (
        stamp.get("legacy_ledger_sha256") == digest
        and stamp.get("legacy_ledger_source_id") == source_id
    ):
        report["skipped"] = 1
        return report
    parsed = _parse_legacy_ledger(path, report)
    if report["errors"]:
        return report

    existing_ids = {event.event_id for event in memory.operational.ledger.validated_events()}
    imported, skipped = _import_legacy_rows(
        memory,
        parsed,
        source_id=source_id,
        existing_ids=existing_ids,
        write=write,
    )
    report["imported"] = imported
    report["skipped"] = skipped
    if write:
        memory.operational.rebuild()
        _write_migration_stamp(
            stamp_path,
            stamp,
            digest=digest,
            source_id=source_id,
        )
    return report


def _read_migration_stamp(path: Path) -> dict[str, Any]:
    try:
        stamp = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return stamp if isinstance(stamp, dict) else {}


def _parse_legacy_ledger(
    path: Path,
    report: dict[str, int],
) -> list[tuple[int, dict[str, Any]]]:
    parsed: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        report["checked"] += 1
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError("legacy journal row is not an object")
            parsed.append((line_number, row))
        except (TypeError, ValueError, json.JSONDecodeError):
            report["errors"] += 1
    return parsed


def _import_legacy_rows(
    memory: Any,
    parsed: list[tuple[int, dict[str, Any]]],
    *,
    source_id: str,
    existing_ids: set[str],
    write: bool,
) -> tuple[int, int]:
    imported = 0
    skipped = 0
    occurrences: dict[str, int] = {}
    for _, row in parsed:
        canonical_row = _canonical_legacy_row(row)
        occurrence = occurrences.get(canonical_row, 0) + 1
        occurrences[canonical_row] = occurrence
        stable_id = _legacy_event_id(source_id, canonical_row, occurrence)
        if stable_id in existing_ids:
            skipped += 1
            continue
        if write:
            memory.operational.ledger.append(
                f"legacy.{row.get('op') or 'event'!s}",
                subject_uri=str(row.get("subject_uri") or "memo://migration/legacy"),
                trace_id=str(row.get("trace_id") or ""),
                payload={
                    "legacy_event": row,
                    "migrated_by": "memo.independence.v1",
                    "legacy_source_id": source_id,
                    "legacy_occurrence": occurrence,
                },
                event_id=stable_id,
            )
        imported += 1
    return imported, skipped


def _legacy_source_id(path: Path) -> str:
    source_uri = path.expanduser().resolve(strict=False).as_uri()
    return hashlib.sha256(f"memo-independence-source:{source_uri}".encode()).hexdigest()


def _canonical_legacy_row(row: dict[str, Any]) -> str:
    return json.dumps(
        row,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _legacy_event_id(source_id: str, canonical_row: str, occurrence: int) -> str:
    payload = json.dumps(
        ["memo-independence.v2", source_id, canonical_row, occurrence],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:32]


def _write_migration_stamp(
    path: Path,
    stamp: dict[str, Any],
    *,
    digest: str,
    source_id: str,
) -> None:
    updated = {
        **stamp,
        "legacy_ledger_sha256": digest,
        "legacy_ledger_source_id": source_id,
    }
    atomic_write_text(
        path,
        json.dumps(updated, ensure_ascii=False, indent=2, sort_keys=True),
    )


def migrate_independence(
    memory: Any,
    *,
    write: bool = False,
    legacy_ledger: Path | None = None,
    config_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Migrate provenance/config/journal data without contacting another system."""
    markdown = {"checked": 0, "migrated": 0, "unchanged": 0, "errors": 0}
    for path in sorted(memory.cfg.memory_dir.rglob("*.md")):
        markdown["checked"] += 1
        result = _migrate_markdown(path, write=write)
        key = "errors" if result == "error" else result
        markdown[key] += 1

    configs = {"checked": 0, "migrated": 0, "unchanged": 0, "errors": 0}
    if config_paths is None:
        from memo.config_md import config_dir, index_path

        discovered = [
            Path.cwd() / ".env",
            Path.home() / ".config" / "memo" / "config.toml",
            index_path(),
        ]
        root = config_dir()
        if root.is_dir():
            discovered.extend(sorted(root.glob("*.md")))
        config_paths = list(dict.fromkeys(discovered))
    for path in config_paths:
        if not path.is_file():
            continue
        configs["checked"] += 1
        result = _migrate_config(path, write=write)
        key = "errors" if result == "error" else result
        configs[key] += 1

    ledger = (
        _import_legacy_ledger(memory, legacy_ledger, write=write)
        if legacy_ledger is not None
        else {"checked": 0, "imported": 0, "errors": 0, "skipped": 0}
    )
    if write and markdown["migrated"]:
        memory.reindex()
    return {
        "schema": "memo.independence_migration.v1",
        "mode": "write" if write else "dry-run",
        "markdown": markdown,
        "config": configs,
        "legacy_ledger": ledger,
        "journal": memory.operational.ledger.verify(),
    }


__all__ = ["migrate_independence"]
