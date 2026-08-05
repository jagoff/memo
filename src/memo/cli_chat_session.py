"""Persistent chat-session continuity CLI."""

from __future__ import annotations

import json
import re
import time
import uuid
from bisect import bisect_right
from pathlib import Path

import click

from memo.atomic_io import atomic_write_text, authority_write_lock
from memo.config import Config

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ROLES = {"user", "assistant", "system"}


def _valid(value: str, name: str) -> str:
    if not value or not _ID.fullmatch(value):
        raise click.ClickException(f"invalid {name}")
    return value


def _path() -> Path:
    d = Config.from_env().state_dir
    d.mkdir(parents=True, exist_ok=True)
    return d / "chat_sessions.json"


def _load() -> dict:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"sessions": {}}
    except OSError as exc:
        raise click.ClickException(f"cannot read chat session store: {exc}") from exc
    except ValueError as exc:
        raise click.ClickException("chat session store is corrupt") from exc

    sessions = data.get("sessions") if isinstance(data, dict) else None
    if not isinstance(sessions, dict) or any(
        not isinstance(session_id, str) or not isinstance(session, dict)
        for session_id, session in sessions.items()
    ):
        raise click.ClickException("chat session store is corrupt")
    return data


def _save(data: dict) -> None:
    p = _path()
    try:
        atomic_write_text(
            p,
            json.dumps(data, ensure_ascii=False, indent=2),
        )
    except OSError as exc:
        raise click.ClickException(f"cannot write chat session store: {exc}") from exc


def _snapshot(session: dict) -> dict:
    """Return a backward-compatible session with an activity watermark."""
    out = dict(session)
    out["updated_at"] = session.get("updated_at", session.get("created_at"))
    out.setdefault("source", "cli")
    return out


def _iso_to_epoch(iso: str) -> float:
    return time.mktime(time.strptime(iso, "%Y-%m-%dT%H:%M:%S"))


def _http_sessions_root() -> Path:
    return Config.from_env().state_dir / "chat" / "sessions"


def _http_sessions() -> dict[str, dict]:
    """Sessions created via `memo chat serve`'s HTTP/SSE API.

    Those live in a separate per-session JSONL store (memo.chat.sessions.SessionStore,
    rooted at state_dir/chat/sessions) that this CLI's own start/append/get/list never
    wrote to and never read from — surfaced here so a session started through the chat
    UI is visible to `memo chat-session get/list` too.
    """
    from memo.chat.sessions import SessionStore

    store = SessionStore(_http_sessions_root())
    out: dict[str, dict] = {}
    for entry in store.list_sessions(limit=1000):
        sid = entry["session_id"]
        out[sid] = {
            "session_id": sid,
            "client": "memo-chat-http",
            "source": "http",
            "created_at": _iso_to_epoch(entry["first_ts"]),
            "updated_at": _iso_to_epoch(entry["last_ts"]),
            "turn_count": entry["turn_count"],
            "label": entry["label"],
        }
    return out


def _http_session_turns(session_id: str) -> list[dict] | None:
    from memo.chat.sessions import SessionStore

    store = SessionStore(_http_sessions_root())
    try:
        turns = store.get(session_id)
    except ValueError:
        return None
    return turns or None


@click.group(name="chat-session")
def chat_session_group():
    """Persisted conversational sessions."""


@chat_session_group.command()
@click.option("--session-id", default="")
@click.option("--client", default="memo-chat", show_default=True)
@click.option("--json", "as_json", is_flag=True)
def start(session_id, client, as_json):
    sid = _valid(session_id, "session_id") if session_id else "cs-" + uuid.uuid4().hex
    with authority_write_lock(_path()):
        data = _load()
        existing = data["sessions"].get(sid)
        if existing is None:
            now = time.time()
            existing = {
                "session_id": sid,
                "client": client,
                "created_at": now,
                "updated_at": now,
                "turns": [],
            }
            data["sessions"][sid] = existing
            _save(data)
    out = _snapshot(existing)
    click.echo(json.dumps(out, ensure_ascii=False) if as_json else sid)


@chat_session_group.command()
@click.argument("session_id")
@click.argument("question")
@click.option("--answer", default="")
@click.option("--client", default="memo-chat")
@click.option("--turn-id", default="")
@click.option("--role", default="user", type=click.Choice(sorted(_ROLES)))
@click.option("--json", "as_json", is_flag=True)
def append(session_id, question, answer, client, turn_id, role, as_json):
    sid = _valid(session_id, "session_id")
    if not question.strip():
        raise click.ClickException("question required")
    tid = _valid(turn_id, "turn_id") if turn_id else "t-" + uuid.uuid4().hex
    with authority_write_lock(_path()):
        data = _load()
        sess = data["sessions"].get(sid)
        if sess is None:
            raise click.ClickException("session not found")
        turns = sess.get("turns")
        if not isinstance(turns, list) or any(not isinstance(turn, dict) for turn in turns):
            raise click.ClickException("chat session store is corrupt")
        found = next((turn for turn in turns if turn.get("turn_id") == tid), None)
        if found is None:
            now = time.time()
            found = {
                "turn_id": tid,
                "role": role,
                "question": question,
                "answer": answer,
                "client": client,
                "created_at": now,
            }
            turns.append(found)
            sess["updated_at"] = now
            _save(data)
    click.echo(json.dumps(found, ensure_ascii=False) if as_json else tid)


@chat_session_group.command()
@click.argument("session_id")
@click.option("--json", "as_json", is_flag=True)
def get(session_id, as_json):
    sid = _valid(session_id, "session_id")
    obj = _load()["sessions"].get(sid)
    if obj is not None:
        obj = _snapshot(obj)
    else:
        turns = _http_session_turns(sid)
        if turns is None:
            raise click.ClickException("session not found")
        obj = {
            "session_id": sid,
            "client": "memo-chat-http",
            "source": "http",
            "created_at": turns[0]["ts"],
            "updated_at": turns[-1]["ts"],
            "turns": turns,
        }
    click.echo(json.dumps(obj, ensure_ascii=False) if as_json else sid)


@chat_session_group.command(name="list")
@click.option("--limit", default=10, type=click.IntRange(1, 1000))
@click.option(
    "--cursor",
    default=None,
    help=(
        "Stable session-id cursor. Supplying it (including an empty first "
        "cursor) enables exhaustive pagination."
    ),
)
@click.option("--json", "as_json", is_flag=True)
def list_sessions(limit, cursor, as_json):
    sessions = {sid: _snapshot(row) for sid, row in _load()["sessions"].items()}
    sessions.update(_http_sessions())
    if cursor is None:
        rows = sorted(
            sessions.values(),
            key=lambda x: x.get("created_at", 0),
            reverse=True,
        )[:limit]
        payload = {"sessions": rows}
    else:
        session_ids = sorted(str(session_id) for session_id in sessions)
        start_at = bisect_right(session_ids, cursor)
        page_ids = session_ids[start_at : start_at + limit]
        rows = [sessions[session_id] for session_id in page_ids]
        has_more = start_at + len(page_ids) < len(session_ids)
        payload = {
            "sessions": rows,
            "next_cursor": page_ids[-1] if has_more and page_ids else None,
            "has_more": has_more,
        }
    click.echo(
        json.dumps(payload, ensure_ascii=False)
        if as_json
        else "\n".join(x["session_id"] for x in rows)
    )
