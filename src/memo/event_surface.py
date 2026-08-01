"""Persistent terminal/conversation events owned by Memo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memo.atomic_io import authority_write_lock, open_secure_directory
from memo.config import Config
from memo.errors import StorageError, ValidationError

SCHEMA = "memo.terminal_event.v1"


def _paths(state_dir: Path) -> tuple[Path, Path]:
    root = state_dir / "events"
    return root / "terminal-conversation.jsonl", root / "context.json"


def _context(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"epoch": 0, "context_id": None}
    except (OSError, json.JSONDecodeError) as exc:
        raise StorageError(f"could not read terminal event context: {path}") from exc
    epoch = value.get("epoch") if isinstance(value, dict) else None
    context_id = value.get("context_id") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
        or (context_id is not None and not isinstance(context_id, str))
    ):
        raise StorageError(f"terminal event context is invalid: {path}")
    return value


def _records(path: Path) -> list[dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise StorageError(f"could not read terminal event store: {path}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StorageError(
                f"terminal event store contains invalid JSON at line {line_number}: {path}"
            ) from exc
        if not isinstance(item, dict):
            raise StorageError(
                f"terminal event store contains a non-object at line {line_number}: {path}"
            )
        records.append(item)
    return records


def ingest_event(
    event: dict[str, Any],
    *,
    state_dir: Path | None = None,
    expected_epoch: int | None = None,
) -> dict[str, Any]:
    if (
        not isinstance(event, dict)
        or not isinstance(event.get("event_id"), str)
        or not event["event_id"]
    ):
        raise ValidationError("event_id is required")
    kind = event.get("kind") or event.get("type")
    if kind not in {"terminal", "conversation", "agent"}:
        raise ValidationError("kind must be terminal, conversation, or agent")
    data_path, context_path = _paths(state_dir or Config.from_env().state_dir)
    record = dict(event)
    record.update(schema=SCHEMA, kind=kind)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with authority_write_lock(data_path):
        context = _context(context_path)
        if expected_epoch is not None and not context_path.exists():
            raise ValidationError("terminal event context is missing")
        if expected_epoch is not None and expected_epoch != int(context.get("epoch", 0)):
            raise ValidationError("terminal event context epoch is stale")
        for old in _records(data_path):
            if old.get("event_id") != record["event_id"]:
                continue
            if old != record:
                raise ValidationError("event_id already exists with different payload")
            return {
                "accepted": False,
                "duplicate": True,
                "event": old,
                "epoch": context.get("epoch", 0),
            }
        encoded = json.dumps(record, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
        with open_secure_directory(data_path.parent, create=True) as directory:
            directory.append_bytes(data_path.name, encoded)
        return {
            "accepted": True,
            "duplicate": False,
            "event": record,
            "epoch": context.get("epoch", 0),
        }


def list_events(
    *,
    state_dir: Path | None = None,
    kind: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    data_path, _ = _paths(state_dir or Config.from_env().state_dir)
    selected = [item for item in _records(data_path) if kind is None or item.get("kind") == kind]
    return selected[-max(0, limit) :]
