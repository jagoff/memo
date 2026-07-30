"""Persistent chat-session continuity CLI."""
from __future__ import annotations
import json, os, re, tempfile, time, uuid
from pathlib import Path
import click
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
        return data if isinstance(data, dict) and isinstance(data.get("sessions"), dict) else {"sessions": {}}
    except (OSError, ValueError):
        return {"sessions": {}}

def _save(data: dict) -> None:
    p = _path(); fd, tmp = tempfile.mkstemp(prefix=".chat-sessions.", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

@click.group(name="chat-session")
def chat_session_group():
    """Persisted conversational sessions."""

@chat_session_group.command()
@click.option("--session-id", default="")
@click.option("--client", default="memo-chat", show_default=True)
@click.option("--json", "as_json", is_flag=True)
def start(session_id, client, as_json):
    sid = _valid(session_id, "session_id") if session_id else "cs-" + uuid.uuid4().hex
    data = _load(); existing = data["sessions"].get(sid)
    if existing is None:
        existing = {"session_id": sid, "client": client, "created_at": time.time(), "turns": []}
        data["sessions"][sid] = existing; _save(data)
    out = dict(existing)
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
    if not question.strip(): raise click.ClickException("question required")
    data = _load(); sess = data["sessions"].get(sid)
    if sess is None: raise click.ClickException("session not found")
    tid = _valid(turn_id, "turn_id") if turn_id else "t-" + uuid.uuid4().hex
    found = next((t for t in sess["turns"] if t.get("turn_id") == tid), None)
    if found is None:
        found = {"turn_id": tid, "role": role, "question": question, "answer": answer, "client": client, "created_at": time.time()}
        sess["turns"].append(found); _save(data)
    click.echo(json.dumps(found, ensure_ascii=False) if as_json else tid)

@chat_session_group.command()
@click.argument("session_id")
@click.option("--json", "as_json", is_flag=True)
def get(session_id, as_json):
    sid = _valid(session_id, "session_id"); obj = _load()["sessions"].get(sid)
    if obj is None: raise click.ClickException("session not found")
    click.echo(json.dumps(obj, ensure_ascii=False) if as_json else sid)

@chat_session_group.command(name="list")
@click.option("--limit", default=10, type=click.IntRange(1, 1000))
@click.option("--json", "as_json", is_flag=True)
def list_sessions(limit, as_json):
    rows = sorted(_load()["sessions"].values(), key=lambda x: x.get("created_at", 0), reverse=True)[:limit]
    click.echo(json.dumps({"sessions": rows}, ensure_ascii=False) if as_json else "\n".join(x["session_id"] for x in rows))
