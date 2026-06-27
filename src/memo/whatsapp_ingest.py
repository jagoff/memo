"""WhatsApp → human-readable notes in the vault → Memo index.

Reads messages from the local `whatsapp-mcp` bridge and writes **one
human-readable Markdown note per contact** in `Obsidian/Whatsapp/<contact>.md`
(transcript grouped by date, no `id:` in the frontmatter). Those notes are
indexed in Memo via `memo ingest` (source=vault-ingest), so they become
searchable by `memo search` / `ask` and by the synapse chat on :8765 — and at
the same time browsable in Obsidian alongside the dossiers in `Obsidian/Contacts`.

No dependency on `rag`. The scope is opt-in (`include_chats` or `all_chats`).
The notes are fully regenerated on every run (idempotent): re-running updates
the contact's file with everything new. The subsequent `memo ingest` re-embeds
only what changed (body_hash).

Fixed exclusions: `status@broadcast`, bot JID (`WHATSAPP_BOT_JID`), notes-inbox
(`WA_LISTENER_NOTES_CHAT_JID`) and messages prefixed with U+200B (bot output).
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Config ─────────────────────────────────────────────────────────────────

_DEFAULT_BRIDGE_DB_RAW = "~/repos/whatsapp-mcp/whatsapp-bridge/store/messages.db"


def _resolve_bridge_db() -> Path:
    """Return the WhatsApp bridge DB path from MEMO_WHATSAPP_DB flag or the
    compiled-in default. Reading through the flags registry ensures the path
    appears in `memo config validate` and `memo config flags`."""
    from memo.flags import flag_str

    raw = flag_str("MEMO_WHATSAPP_DB") or _DEFAULT_BRIDGE_DB_RAW
    return Path(raw).expanduser()


DEFAULT_BRIDGE_DB = _resolve_bridge_db()

DEFAULT_RETENTION_DAYS = 180

# Your own bot/listener group JIDs to exclude from ingest. No personal
# identifiers are baked into source — set these env vars to your own JIDs.
def _get_whatsapp_jids() -> tuple[str, str]:
    try:
        from memo.flags import flag_str

        return flag_str("WHATSAPP_BOT_JID") or "", flag_str("WA_LISTENER_NOTES_CHAT_JID") or ""
    except Exception:
        return "", ""


_BOT_JID, _LISTENER_NOTES_CHAT_JID = _get_whatsapp_jids()
HARDCODED_EXCLUDE_JIDS = frozenset(
    jid for jid in {"status@broadcast", _BOT_JID, _LISTENER_NOTES_CHAT_JID} if jid
)

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
    timestamp: float  # epoch seconds
    is_from_me: bool
    media_type: str | None


# ── Reader ───────────────────────────────────────────────────────────────────


def _parse_bridge_ts(raw: object) -> float | None:
    """Parse the bridge `timestamp` (RFC3339 string or numeric) to epoch."""
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
    """Load messages from the bridge (read-only). `since_ts` filters by epoch
    (exclusive). Skips empties, excluded chats, and bot output."""
    if not bridge_db.is_file():
        return []
    # mode=ro (NOT immutable): the bridge writes to WAL and keeps recent messages
    # there until the next checkpoint. `immutable=1` promises SQLite the file
    # doesn't change and makes it IGNORE the -wal, so the newest tail of each chat
    # stays invisible until a checkpoint → stale notes/index.
    conn = sqlite3.connect(f"file:{bridge_db}?mode=ro", uri=True)
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
        out.append(
            WAMessage(
                id=str(r["id"]),
                chat_jid=jid,
                chat_name=chat_name or jid,
                sender=str(r["sender"] or ""),
                content=content,
                timestamp=ts,
                is_from_me=bool(r["is_from_me"]),
                media_type=r["media_type"],
            )
        )
    return out


# ── Note rendering ────────────────────────────────────────────────────────────


def _mask_phone(sender: str) -> str:
    digits = "".join(_PHONE_DIGITS.findall(sender))
    return f"…{digits[-4:]}" if len(digits) >= 4 else "?"


def _speaker_label(msg: WAMessage) -> str:
    """me / contact name (1:1) / last 4 digits (group)."""
    if msg.is_from_me:
        return "me"
    is_group = msg.chat_jid.endswith("@g.us")
    if not is_group and msg.chat_name and msg.chat_name != msg.chat_jid:
        return msg.chat_name
    return _mask_phone(msg.sender)


def _safe_filename(chat_name: str, chat_jid: str) -> str:
    base = _FNAME_BAD.sub("-", chat_name).strip() or chat_jid.split("@", 1)[0]
    return base[:120]


def render_chat_note(chat_jid: str, chat_name: str, msgs: list[WAMessage]) -> str:
    """A human-readable note: frontmatter (no `id:`) + transcript grouped by day."""
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
        # soft link to the dossier in Obsidian/Contacts (doesn't break if absent)
        lines.append(f"> Contact: [[{chat_name}]] · `{chat_jid}`")
    else:
        lines.append(f"> Group · `{chat_jid}`")
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


# ── Orchestration ─────────────────────────────────────────────────────────────


def resolve_notes_dir(mem: Any) -> Path:
    """`<SYSTEM_DIR>/Whatsapp` at the vault root. Override: MEMO_WHATSAPP_NOTES_DIR.

    The vault root is derived from `data_dir` by finding the `<SYSTEM_DIR>`
    ancestor (e.g. `Obsidian`) and taking its parent — robust to the depth of
    the memories subdir (`<SYSTEM_DIR>/Memory`, `<SYSTEM_DIR>/AI/memory`, …).
    If the structure doesn't contain `<SYSTEM_DIR>`, use the env var.
    """
    from memo.config import SYSTEM_DIR
    from memo.flags import flag_str

    env = flag_str("MEMO_WHATSAPP_NOTES_DIR")
    if env:
        return Path(env).expanduser()
    data_dir = Path(mem.cfg.data_dir)
    # …/Notes/<SYSTEM_DIR>/Memory  → vault_root = …/Notes (parent of SYSTEM_DIR).
    vault_root = data_dir
    for anc in data_dir.parents:
        if anc.name == SYSTEM_DIR:
            vault_root = anc.parent
            break
    else:
        # Unexpected layout (no SYSTEM_DIR in the path): fall back to the previous one.
        vault_root = data_dir.parents[2] if len(data_dir.parents) > 2 else data_dir
    return vault_root / SYSTEM_DIR / "Whatsapp"


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
    """Write a human-readable note per chat in `Obsidian/Whatsapp`. Returns a summary.
    Indexing into Memo (`memo ingest`) is triggered by the CLI layer."""
    if not include_chats and not all_chats:
        raise ValueError(
            "scope required: pass include_chats=(...) or all_chats=True "
            "(I don't ingest all chats by default)"
        )

    from memo.flags import flag_str

    db = bridge_db or _resolve_bridge_db()
    if not db.is_file():
        if bridge_db is None and flag_str("MEMO_WHATSAPP_DB") == "":
            raise ValueError(
                f"WhatsApp DB not found: {db}. "
                "Set MEMO_WHATSAPP_DB env var to the path of your whatsapp-mcp messages.db."
            )
        raise FileNotFoundError(f"bridge DB not found: {db}")

    out_dir = notes_dir or resolve_notes_dir(mem)

    since_ts = time.time() - retention_days * 86400
    if since:
        try:
            since_ts = max(since_ts, datetime.strptime(since, "%Y-%m-%d").timestamp())
        except ValueError as exc:
            raise ValueError(f"--since must be YYYY-MM-DD, got {since!r}") from exc

    messages = read_messages(
        db, since_ts=since_ts, exclude_jids=HARDCODED_EXCLUDE_JIDS | set(exclude_chats)
    )

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
