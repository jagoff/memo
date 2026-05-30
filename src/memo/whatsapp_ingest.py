"""WhatsApp → notas legibles en el vault → índice de Memo.

Lee los mensajes del bridge local de `whatsapp-mcp` y escribe **una nota
Markdown legible por contacto** en `Obsidian/Whatsapp/<contacto>.md` (transcript
agrupado por fecha, sin `id:` en el frontmatter). Esas notas se indexan en Memo
vía `memo ingest` (source=vault-ingest), así quedan buscables por `memo search`
/ `ask` y por el chat de synapse en :8765 — y a la vez son navegables en
Obsidian junto a los dossiers de `Obsidian/Contacts`.

Sin dependencia de `rag`. El scope es opt-in (`include_chats` o `all_chats`).
Las notas se regeneran completas en cada corrida (idempotente): re-ejecutar
actualiza el archivo del contacto con todo lo nuevo. El `memo ingest` posterior
re-embebe sólo lo que cambió (body_hash).

Exclusiones fijas: `status@broadcast`, bot JID (`WHATSAPP_BOT_JID`), notes-inbox
(`WA_LISTENER_NOTES_CHAT_JID`) y mensajes con prefijo U+200B (output del bot).
"""
from __future__ import annotations

import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Config ─────────────────────────────────────────────────────────────────

DEFAULT_BRIDGE_DB = Path(
    os.environ.get(
        "MEMO_WHATSAPP_DB",
        "~/repos/whatsapp-mcp/whatsapp-bridge/store/messages.db",
    )
).expanduser()

DEFAULT_RETENTION_DAYS = 180

_BOT_JID = os.environ.get("WHATSAPP_BOT_JID", "120363426178035051@g.us")
_LISTENER_NOTES_CHAT_JID = os.environ.get(
    "WA_LISTENER_NOTES_CHAT_JID", "5493425153999-1539438783@g.us",
)
HARDCODED_EXCLUDE_JIDS = frozenset({
    "status@broadcast",
    _BOT_JID,
    _LISTENER_NOTES_CHAT_JID,
})

_ANTILOOP_MARKER = "​"  # U+200B
_PHONE_DIGITS = re.compile(r"\d+")
_FNAME_BAD = re.compile(r"[\\/:\*\?\"<>\|]+")


@dataclass(frozen=True)
class WAMessage:
    id: str
    chat_jid: str
    chat_name: str
    sender: str
    content: str
    timestamp: float              # epoch seconds
    is_from_me: bool
    media_type: str | None


# ── Reader ───────────────────────────────────────────────────────────────────

def _parse_bridge_ts(raw: object) -> float | None:
    """Parsea el `timestamp` del bridge (RFC3339 string o numérico) a epoch."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if not s:
        return None
    if "." in s:
        head, sep, rest = s.partition(".")
        frac = rest
        tz = ""
        for i, ch in enumerate(rest):
            if ch in "+-Z":
                frac = rest[:i]
                tz = rest[i:]
                break
        frac = frac[:6]
        s = f"{head}{sep}{frac}{tz}"
    s = s.replace("Z", "+00:00") if s.endswith("Z") else s
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt.timestamp()


def read_messages(
    bridge_db: Path,
    *,
    since_ts: float = 0.0,
    chat_jid: str | None = None,
    exclude_jids: frozenset[str] = HARDCODED_EXCLUDE_JIDS,
) -> list[WAMessage]:
    """Carga mensajes del bridge (read-only). `since_ts` filtra por epoch
    (exclusivo). Salta vacíos, chats excluidos y output del bot."""
    if not bridge_db.is_file():
        return []
    conn = sqlite3.connect(f"file:{bridge_db}?mode=ro&immutable=1", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        q = (
            "SELECT m.id, m.chat_jid, m.sender, m.content, m.timestamp, "
            " m.is_from_me, m.media_type, COALESCE(c.name, '') AS chat_name "
            "FROM messages m LEFT JOIN chats c ON c.jid = m.chat_jid "
            "WHERE m.content IS NOT NULL AND m.content != '' "
        )
        params: list[Any] = []
        if chat_jid:
            q += " AND m.chat_jid = ? "
            params.append(chat_jid)
        q += " ORDER BY m.chat_jid, m.timestamp"
        rows = conn.execute(q, params).fetchall()
    finally:
        conn.close()

    out: list[WAMessage] = []
    for r in rows:
        jid = r["chat_jid"]
        if jid in exclude_jids:
            continue
        content = str(r["content"] or "")
        if content.startswith(_ANTILOOP_MARKER):
            continue
        ts = _parse_bridge_ts(r["timestamp"])
        if ts is None or ts <= since_ts:
            continue
        chat_name = str(r["chat_name"] or "")
        out.append(WAMessage(
            id=str(r["id"]),
            chat_jid=jid,
            chat_name=chat_name or jid,
            sender=str(r["sender"] or ""),
            content=content,
            timestamp=ts,
            is_from_me=bool(r["is_from_me"]),
            media_type=r["media_type"],
        ))
    return out


# ── Render de notas ───────────────────────────────────────────────────────────

def _mask_phone(sender: str) -> str:
    digits = "".join(_PHONE_DIGITS.findall(sender))
    return f"…{digits[-4:]}" if len(digits) >= 4 else "?"


def _speaker_label(msg: WAMessage) -> str:
    """yo / nombre del contacto (1:1) / últimos 4 dígitos (grupo)."""
    if msg.is_from_me:
        return "yo"
    is_group = msg.chat_jid.endswith("@g.us")
    if not is_group and msg.chat_name and msg.chat_name != msg.chat_jid:
        return msg.chat_name
    return _mask_phone(msg.sender)


def _safe_filename(chat_name: str, chat_jid: str) -> str:
    base = _FNAME_BAD.sub("-", chat_name).strip() or chat_jid.split("@", 1)[0]
    return base[:120]


def render_chat_note(chat_jid: str, chat_name: str, msgs: list[WAMessage]) -> str:
    """Una nota legible: frontmatter (sin `id:`) + transcript agrupado por día."""
    msgs = sorted(msgs, key=lambda m: m.timestamp)
    first_iso = datetime.fromtimestamp(msgs[0].timestamp).isoformat(timespec="seconds")
    last_iso = datetime.fromtimestamp(msgs[-1].timestamp).isoformat(timespec="seconds")
    is_group = chat_jid.endswith("@g.us")

    lines: list[str] = []
    lines.append("---")
    lines.append(f"title: WhatsApp · {chat_name}")
    lines.append("source: whatsapp")
    lines.append(f"chat_jid: {chat_jid}")
    lines.append(f"chat_name: {chat_name}")
    lines.append("tags: [whatsapp, chat]")
    lines.append(f"first_message: {first_iso}")
    lines.append(f"last_message: {last_iso}")
    lines.append(f"messages: {len(msgs)}")
    lines.append("---")
    lines.append("")
    lines.append(f"# WhatsApp · {chat_name}")
    lines.append("")
    if not is_group:
        # link blando al dossier en Obsidian/Contacts (no rompe si no existe)
        lines.append(f"> Contacto: [[{chat_name}]] · `{chat_jid}`")
    else:
        lines.append(f"> Grupo · `{chat_jid}`")
    lines.append("")

    cur_day = ""
    for m in msgs:
        dt = datetime.fromtimestamp(m.timestamp)
        day = dt.strftime("%Y-%m-%d")
        if day != cur_day:
            cur_day = day
            lines.append(f"## {day}")
            lines.append("")
        speaker = _speaker_label(m)
        text = m.content.strip().replace("\n", " ")
        lines.append(f"- **{speaker}** ({dt.strftime('%H:%M')}): {text}")
    lines.append("")
    return "\n".join(lines)


# ── Orquestación ──────────────────────────────────────────────────────────────

def resolve_notes_dir(mem: Any) -> Path:
    """`Obsidian/Whatsapp` en la raíz del vault. Override: MEMO_WHATSAPP_NOTES_DIR.

    La raíz del vault se deriva de `data_dir` (…/<vault>/Obsidian/AI/memory
    → <vault>). Si la estructura difiere, usar el env var.
    """
    env = os.environ.get("MEMO_WHATSAPP_NOTES_DIR")
    if env:
        return Path(env).expanduser()
    data_dir = Path(mem.cfg.data_dir)
    # …/Notes/Obsidian/AI/memory  → parents[2] = …/Notes
    try:
        vault_root = data_dir.parents[2]
    except IndexError:
        vault_root = data_dir
    return vault_root / "Obsidian" / "Whatsapp"


def run(
    mem: Any,
    *,
    bridge_db: Path | None = None,
    since: str | None = None,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    include_chats: tuple[str, ...] = (),
    exclude_chats: tuple[str, ...] = (),
    all_chats: bool = False,
    notes_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Escribe una nota legible por chat en `Obsidian/Whatsapp`. Devuelve resumen.
    El indexado en Memo (`memo ingest`) lo dispara la capa CLI."""
    if not include_chats and not all_chats:
        raise ValueError(
            "scope requerido: pasá include_chats=(...) o all_chats=True "
            "(no ingesto todos los chats por defecto)"
        )

    db = bridge_db or DEFAULT_BRIDGE_DB
    if not db.is_file():
        raise FileNotFoundError(f"bridge DB no encontrada: {db}")

    out_dir = notes_dir or resolve_notes_dir(mem)

    since_ts = time.time() - retention_days * 86400
    if since:
        try:
            since_ts = max(since_ts, datetime.strptime(since, "%Y-%m-%d").timestamp())
        except ValueError as exc:
            raise ValueError(f"--since debe ser YYYY-MM-DD, got {since!r}") from exc

    messages = read_messages(db, since_ts=since_ts, exclude_jids=HARDCODED_EXCLUDE_JIDS | set(exclude_chats))

    include = set(include_chats)
    by_chat: dict[str, list[WAMessage]] = {}
    for m in messages:
        if include and m.chat_jid not in include:
            continue
        by_chat.setdefault(m.chat_jid, []).append(m)

    summary: dict[str, Any] = {
        "bridge_db": str(db),
        "notes_dir": str(out_dir),
        "messages_read": len(messages),
        "chats": sorted(by_chat.keys()),
        "notes_written": 0,
        "files": [],
        "dry_run": dry_run,
    }

    if dry_run or not by_chat:
        return summary

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    files: list[str] = []
    for jid, msgs in by_chat.items():
        chat_name = msgs[0].chat_name
        note = render_chat_note(jid, chat_name, msgs)
        fname = _safe_filename(chat_name, jid) + ".md"
        path = out_dir / fname
        path.write_text(note, encoding="utf-8")
        written += 1
        files.append(str(path))

    summary["notes_written"] = written
    summary["files"] = files
    return summary
