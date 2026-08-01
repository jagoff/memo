"""Persistent chat-session continuity CLI."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

import click

from memo.atomic_io import atomic_write_text, authority_write_lock
from memo.config import Config
from memo.errors import StorageError

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ROLES = {"user", "assistant", "system"}


def _valid(value: str, name: str) -> str:
    if not value or not _ID.fullmatch(value):
        raise click.ClickException(f"invalid {name}")
    return value


def _path() -> Path:
    directory = Config.from_env().state_dir
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "chat_sessions.json"


def _load(path: Path | None = None) -> dict[str, Any]:
    source = path or _path()
    try:
        raw = source.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"sessions": {}}
    except OSError as exc:
        raise StorageError(f"could not read chat sessions: {source}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StorageError(f"chat session store is invalid JSON: {source}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("sessions"), dict):
        raise StorageError(f"chat session store has an invalid shape: {source}")
    return data


def _save(path: Path, data: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
    )


def _start_session(*, session_id: str, client: str) -> dict[str, Any]:
    path = _path()
    with authority_write_lock(path):
        data = _load(path)
        existing = data["sessions"].get(session_id)
        if existing is None:
            existing = {
                "session_id": session_id,
                "client": client,
                "created_at": time.time(),
                "turns": [],
            }
            data["sessions"][session_id] = existing
            _save(path, data)
        elif existing.get("client") != client:
            raise click.ClickException("session_id already exists for a different client")
        return dict(existing)


def _append_turn(
    *,
    session_id: str,
    question: str,
    answer: str,
    client: str,
    turn_id: str,
    role: str,
) -> dict[str, Any]:
    path = _path()
    with authority_write_lock(path):
        data = _load(path)
        session = data["sessions"].get(session_id)
        if session is None:
            raise click.ClickException("session not found")
        turn: dict[str, Any] = {
            "turn_id": turn_id,
            "role": role,
            "question": question,
            "answer": answer,
            "client": client,
        }
        found = next(
            (item for item in session["turns"] if item.get("turn_id") == turn_id),
            None,
        )
        if found is not None:
            if any(found.get(key) != value for key, value in turn.items()):
                raise click.ClickException(
                    "turn_id already exists with a different payload"
                )
            return dict(found)
        turn["created_at"] = time.time()
        session["turns"].append(turn)
        _save(path, data)
        return turn


@click.group(name="chat-session")
def chat_session_group() -> None:
    """Local diagnostic chats; not replicated. Use `memo mesh` for peers."""


@chat_session_group.command()
@click.option("--session-id", default="")
@click.option("--client", default="memo-chat", show_default=True)
@click.option("--json", "as_json", is_flag=True)
def start(session_id: str, client: str, as_json: bool) -> None:
    sid = _valid(session_id, "session_id") if session_id else "cs-" + uuid.uuid4().hex
    out = _start_session(session_id=sid, client=client)
    click.echo(json.dumps(out, ensure_ascii=False) if as_json else sid)


@chat_session_group.command()
@click.argument("session_id")
@click.argument("question")
@click.option("--answer", default="")
@click.option("--client", default="memo-chat")
@click.option("--turn-id", default="")
@click.option("--role", default="user", type=click.Choice(sorted(_ROLES)))
@click.option("--json", "as_json", is_flag=True)
def append(
    session_id: str,
    question: str,
    answer: str,
    client: str,
    turn_id: str,
    role: str,
    as_json: bool,
) -> None:
    sid = _valid(session_id, "session_id")
    if not question.strip():
        raise click.ClickException("question required")
    tid = _valid(turn_id, "turn_id") if turn_id else "t-" + uuid.uuid4().hex
    found = _append_turn(
        session_id=sid,
        question=question,
        answer=answer,
        client=client,
        turn_id=tid,
        role=role,
    )
    click.echo(json.dumps(found, ensure_ascii=False) if as_json else tid)


@chat_session_group.command()
@click.argument("session_id")
@click.option("--json", "as_json", is_flag=True)
def get(session_id: str, as_json: bool) -> None:
    sid = _valid(session_id, "session_id")
    obj = _load()["sessions"].get(sid)
    if obj is None:
        raise click.ClickException("session not found")
    click.echo(json.dumps(obj, ensure_ascii=False) if as_json else sid)


@chat_session_group.command(name="list")
@click.option("--limit", default=10, type=click.IntRange(1, 1000))
@click.option("--json", "as_json", is_flag=True)
def list_sessions(limit: int, as_json: bool) -> None:
    rows = sorted(
        _load()["sessions"].values(),
        key=lambda item: item.get("created_at", 0),
        reverse=True,
    )[:limit]
    click.echo(
        json.dumps({"sessions": rows}, ensure_ascii=False)
        if as_json
        else "\n".join(item["session_id"] for item in rows)
    )
