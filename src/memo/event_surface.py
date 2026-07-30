"""Persistent terminal/conversation events owned by Memo."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .config import Config

SCHEMA = "memo.terminal_event.v1"
def _paths(state_dir: Path) -> tuple[Path, Path]:
    root = state_dir / "events"
    return root / "terminal-conversation.jsonl", root / "context.json"
def _context(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {"epoch": 0, "context_id": None}
def ingest_event(event: dict[str, Any], *, state_dir: Path | None = None, expected_epoch: int | None = None) -> dict[str, Any]:
    if not isinstance(event, dict) or not isinstance(event.get("event_id"), str) or not event["event_id"]:
        raise ValueError("event_id is required")
    kind = event.get("kind") or event.get("type")
    if kind not in {"terminal", "conversation", "agent"}:
        raise ValueError("kind must be terminal, conversation, or agent")
    data_path, context_path = _paths(state_dir or Config.from_env().state_dir)
    context = _context(context_path)
    if expected_epoch is not None and expected_epoch != int(context.get("epoch", 0)):
        raise RuntimeError("stale context epoch")
    record = dict(event); record.update(schema=SCHEMA, kind=kind)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if data_path.exists():
        for line in data_path.read_text(encoding="utf-8").splitlines():
            try:
                old = json.loads(line)
                if old.get("event_id") == record["event_id"]:
                    if old != record: raise ValueError("event_id already exists with different payload")
                    return {"accepted": False, "duplicate": True, "event": old, "epoch": context.get("epoch", 0)}
            except json.JSONDecodeError: continue
    with data_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
    return {"accepted": True, "duplicate": False, "event": record, "epoch": context.get("epoch", 0)}
def list_events(*, state_dir: Path | None = None, kind: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    data_path, _ = _paths(state_dir or Config.from_env().state_dir)
    if not data_path.exists(): return []
    out = []
    for line in data_path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            if kind is None or item.get("kind") == kind: out.append(item)
        except json.JSONDecodeError: continue
    return out[-max(0, limit):]
