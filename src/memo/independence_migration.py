"""One-way migration from legacy integration metadata into Memo 4 contracts."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any

import frontmatter

from memo.atomic_io import atomic_write_text
from memo.config_md import _TOML_BLOCK_RE
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
_REMOVED_FLAG_ASSIGNMENT_RE = re.compile(
    rf"^\s*(?:export\s+)?(?:{'|'.join(sorted(map(re.escape, _REMOVED_FLAGS)))})\s*=",
    re.IGNORECASE,
)
_LEGACY_CACHE_ASSIGNMENT_RE = re.compile(
    r"""^(\s*(?:export\s+)?MEMO_CACHE_BACKEND\s*=\s*)(["']?)memflow\b\2""",
    re.IGNORECASE,
)
_SHELL_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_ENV_OPTIONS_WITH_VALUE = frozenset({"-C", "--chdir", "-S", "--split-string", "-u", "--unset"})
_ENV_OPTIONS_WITHOUT_VALUE = frozenset({"-0", "--ignore-environment", "--null", "-i"})
_SHELL_PUNCTUATION = ";&|<>()`"


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
        try:
            atomic_write_text(path, rendered)
        except (OSError, ValueError):
            return "error"
    return "migrated"


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    while index > backslashes and text[index - backslashes - 1] == "\\":
        backslashes += 1
    return backslashes % 2 == 1


def _toml_multiline_state(line: str, active: str | None) -> tuple[str | None, bool]:
    """Track TOML multiline strings so migration never edits their contents."""
    protected = active is not None
    quote: str | None = None
    index = 0
    while index < len(line):
        if active is not None:
            if line.startswith(active, index) and (active == "'''" or not _is_escaped(line, index)):
                active = None
                index += 3
            else:
                index += 1
            continue
        char = line[index]
        if quote is not None:
            if char == quote and (quote == "'" or not _is_escaped(line, index)):
                quote = None
            index += 1
            continue
        if char == "#":
            break
        delimiter = next(
            (candidate for candidate in ('"""', "'''") if line.startswith(candidate, index)),
            None,
        )
        if delimiter is not None:
            active = delimiter
            protected = True
            index += 3
            continue
        if char in {"'", '"'}:
            quote = char
        index += 1
    return active, protected


def _migrate_config_assignments(original: str, *, toml: bool = False) -> tuple[str, bool]:
    lines: list[str] = []
    changed = False
    multiline: str | None = None
    for line in original.splitlines(keepends=True):
        multiline, protected = _toml_multiline_state(line, multiline) if toml else (None, False)
        if not protected and _REMOVED_FLAG_ASSIGNMENT_RE.search(line):
            changed = True
            continue
        updated = (
            line
            if protected
            else _LEGACY_CACHE_ASSIGNMENT_RE.sub(
                r"\1\2vault\2",
                line,
            )
        )
        changed = changed or updated != line
        lines.append(updated)
    return "".join(lines), changed


def _migrate_markdown_config(original: str) -> tuple[str, bool]:
    parts: list[str] = []
    cursor = 0
    changed = False
    for match in _TOML_BLOCK_RE.finditer(original):
        raw_block = match.group(0)
        body_start = match.start(1) - match.start(0)
        closing_start = len(raw_block) - len("```")
        body = raw_block[body_start:closing_start]
        updated, block_changed = _migrate_config_assignments(body, toml=True)
        parts.append(original[cursor : match.start(0)])
        parts.append(raw_block[:body_start])
        parts.append(updated)
        parts.append(raw_block[closing_start:])
        changed = changed or block_changed
        cursor = match.end(0)
    parts.append(original[cursor:])
    return "".join(parts), changed


def _migrate_config(path: Path, *, write: bool) -> str:
    try:
        original = path.read_text(encoding="utf-8")
    except OSError:
        return "error"
    if path.suffix.lower() == ".md":
        rendered, changed = _migrate_markdown_config(original)
    else:
        rendered, changed = _migrate_config_assignments(
            original, toml=path.suffix.lower() == ".toml"
        )
    if not changed:
        return "unchanged"
    if write:
        try:
            atomic_write_text(path, rendered)
        except (OSError, ValueError):
            return "error"
    return "migrated"


def _is_legacy_memflow_startup_hook(value: object) -> bool:
    if not isinstance(value, dict) or value.get("type") not in (None, "command"):
        return False
    command = value.get("command")
    if not isinstance(command, str):
        return False
    argv = _hook_command_argv(command)
    return (
        len(argv) >= 2
        and _executable_name(argv[0]) == "memflow"
        and argv[1].lower() == "startup-banner"
    )


def _is_memo_briefing_hook(value: object) -> bool:
    if not isinstance(value, dict) or value.get("type") not in (None, "command"):
        return False
    command = value.get("command")
    if not isinstance(command, str):
        return False
    argv = _hook_command_argv(command)
    return (
        len(argv) >= 3
        and _executable_name(argv[0]) == "memo"
        and argv[1:3] == ["briefing", "--compact"]
    )


def _executable_name(token: str) -> str:
    name = token.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return name.removesuffix(".exe")


def _simple_shell_argv(command: str) -> list[str]:
    if "\r" in command or "\n" in command or "$(" in command or "`" in command:
        return []
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=_SHELL_PUNCTUATION)
        lexer.whitespace_split = True
        lexer.commenters = ""
        argv = list(lexer)
    except ValueError:
        return []
    if any(token and set(token) <= set(_SHELL_PUNCTUATION) for token in argv):
        return []
    return argv


def _hook_command_argv(command: str) -> list[str]:
    """Return the real executable and args for a simple hook command.

    Initial POSIX assignments and the standard ``env`` wrapper are supported.
    Shell pipelines/wrappers and compound commands intentionally fail closed:
    migration must neither classify a later argument such as
    ``printf memo briefing --compact`` as the executable nor replace a legacy
    command while silently deleting another action chained to it.
    """
    argv = _simple_shell_argv(command)
    index = 0
    while index < len(argv) and _SHELL_ASSIGNMENT_RE.match(argv[index]):
        index += 1
    if index >= len(argv) or _executable_name(argv[index]) != "env":
        return argv[index:]

    index += 1
    while index < len(argv):
        arg = argv[index]
        if arg == "--":
            index += 1
            break
        if _SHELL_ASSIGNMENT_RE.match(arg) or arg in _ENV_OPTIONS_WITHOUT_VALUE:
            index += 1
            continue
        if arg in _ENV_OPTIONS_WITH_VALUE:
            index += 2
            continue
        if arg.startswith(("--chdir=", "--split-string=", "--unset=")):
            index += 1
            continue
        if arg.startswith("-"):
            return []
        break
    return argv[index:]


def _load_codex_session_start(
    path: Path,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None, list[object] | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "error", None, None, None
    if not isinstance(payload, dict):
        return "error", None, None, None
    hooks = payload.get("hooks")
    if not isinstance(hooks, dict):
        return "unchanged", None, None, None
    session_start = hooks.get("SessionStart")
    if not isinstance(session_start, list):
        return "unchanged", None, None, None
    return "ready", payload, hooks, session_start


def _has_memo_briefing(session_start: list[object]) -> bool:
    for group in session_start:
        if not isinstance(group, dict):
            continue
        group_hooks = group.get("hooks")
        if isinstance(group_hooks, list) and any(
            _is_memo_briefing_hook(hook) for hook in group_hooks
        ):
            return True
    return False


def _migrate_codex_hook_group(
    group: object,
    *,
    has_memo_briefing: bool,
    memo_bin: str,
) -> tuple[object | None, bool, bool]:
    if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
        return group, False, has_memo_briefing

    original_hooks = group["hooks"]
    migrated = False
    migrated_hooks: list[object] = []
    for hook in original_hooks:
        if not _is_legacy_memflow_startup_hook(hook):
            migrated_hooks.append(hook)
            continue
        migrated = True
        if has_memo_briefing:
            continue
        replacement = dict(hook)
        replacement.update(
            {
                "type": "command",
                "command": (f"MEMO_NONINTERACTIVE=1 {shlex.quote(memo_bin)} briefing --compact"),
                "timeout": 5,
                "statusMessage": "Loading memo briefing",
            }
        )
        migrated_hooks.append(replacement)
        has_memo_briefing = True

    if not migrated_hooks and original_hooks:
        return None, migrated, has_memo_briefing
    migrated_group = dict(group)
    migrated_group["hooks"] = migrated_hooks
    return migrated_group, migrated, has_memo_briefing


def _migrate_codex_hooks(path: Path, *, write: bool, memo_bin: str) -> str:
    """Replace the retired Memflow SessionStart hook without touching foreign hooks."""
    status, payload, hooks, session_start = _load_codex_session_start(path)
    if status != "ready":
        return status
    assert payload is not None and hooks is not None and session_start is not None

    has_memo_briefing = _has_memo_briefing(session_start)
    migrated = False
    migrated_groups: list[object] = []
    for group in session_start:
        migrated_group, group_migrated, has_memo_briefing = _migrate_codex_hook_group(
            group,
            has_memo_briefing=has_memo_briefing,
            memo_bin=memo_bin,
        )
        migrated = migrated or group_migrated
        if migrated_group is not None:
            migrated_groups.append(migrated_group)

    if not migrated:
        return "unchanged"
    hooks["SessionStart"] = migrated_groups
    if write:
        try:
            atomic_write_text(
                path,
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            )
        except (OSError, ValueError):
            return "error"
    return "migrated"


def _import_legacy_ledger(memory: Any, path: Path, *, write: bool) -> dict[str, int]:
    report = {"checked": 0, "imported": 0, "errors": 0, "skipped": 0}
    if not path.is_file():
        return report
    try:
        raw_ledger = path.read_bytes()
        source_id = _legacy_source_id(path)
    except OSError:
        report["errors"] = 1
        return report
    digest = hashlib.sha256(raw_ledger).hexdigest()
    stamp_path = memory.cfg.state_dir / "independence-migration.json"
    stamp = _read_migration_stamp(stamp_path)
    if (
        stamp.get("legacy_ledger_sha256") == digest
        and stamp.get("legacy_ledger_source_id") == source_id
    ):
        report["skipped"] = 1
        return report
    parsed = _parse_legacy_ledger(raw_ledger, report)
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
    raw_ledger: bytes,
    report: dict[str, int],
) -> list[tuple[int, dict[str, Any]]]:
    parsed: list[tuple[int, dict[str, Any]]] = []
    try:
        lines = raw_ledger.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        report["errors"] += 1
        return parsed
    for line_number, line in enumerate(lines, start=1):
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
            memory.operational.import_legacy_event(
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
    codex_hooks_path: Path | None = None,
    memo_bin: str = "memo",
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

    integrations = {"checked": 0, "migrated": 0, "unchanged": 0, "errors": 0}
    if codex_hooks_path is not None and codex_hooks_path.is_file():
        integrations["checked"] = 1
        result = _migrate_codex_hooks(codex_hooks_path, write=write, memo_bin=memo_bin)
        key = "errors" if result == "error" else result
        integrations[key] += 1

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
        "agent_integrations": integrations,
        "legacy_ledger": ledger,
        "journal": memory.operational.ledger.verify(),
    }


__all__ = ["migrate_independence"]
