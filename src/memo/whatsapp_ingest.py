"""WhatsApp → memo ingest.

Reads messages from the local `whatsapp-mcp` bridge SQLite store, groups
contiguous same-speaker messages into conversational chunks, and saves each
chunk as a memo memoria with `type="reference"` and `extra.source="whatsapp"`.

Reference-typed records are searchable by `memo search`/`ask` and by the
synapse chat at :8765 (which applies no type filter), but stay OUT of the
every-prompt recall hook when `MEMO_RECALL_EXCLUDE_REFERENCE=1` — so personal
chat history never pollutes recall as "authoritative facts".

The chunking heuristics (same-speaker 5-min window, 800-char cap, tiny-group
merge, ±10-message `parent` rerank window) are ported from
`rag/scripts/ingest_whatsapp.py` but this module has NO dependency on `rag`:
it owns a simplified sender-label resolver and a JSON cursor under
`MEMO_STATE_DIR` (not rag's sqlite-vec collection).

Idempotent re-runs: a per-chat cursor (`whatsapp_ingest_state.json`) tracks the
max ingested message timestamp per `chat_jid`; messages at or below the cursor
are skipped. memo has no content-level dedup (every save is a new UUID), so the
cursor is the dedup mechanism — a crash between save and cursor-write can
re-ingest at most one chat's most recent window on the next run.

Invoked via `memo import whatsapp` (see `cli_import.py`).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

# ── Config ─────────────────────────────────────────────────────────────────

# Bridge DB path. Override via env for non-default checkouts / tests.
DEFAULT_BRIDGE_DB = Path(
    os.environ.get(
        "MEMO_WHATSAPP_DB",
        "~/repos/whatsapp-mcp/whatsapp-bridge/store/messages.db",
    )
).expanduser()

# Conversational chunk boundaries (match rag/vault conventions).
CHUNK_SAME_SPEAKER_WINDOW_S = 300     # 5 min — messages closer than this merge
CHUNK_MIN_MERGE_CHARS = 150           # target lower bound; tinier groups merge up
CHUNK_MAX_CHARS = 800                 # hard cap matches vault chunk size

PARENT_WINDOW_MESSAGES = 10           # ±N messages around chunk for `parent`
PARENT_MAX_CHARS = 1200               # cap (vault convention)

DEFAULT_RETENTION_DAYS = 180

# Chats never indexed regardless of opt-in. status@broadcast = WhatsApp's
# internal story feed (not conversational). The bot JID + notes-inbox JID are
# pulled from the same env vars the listener/rag use, so all WA paths agree.
_BOT_JID = os.environ.get("WHATSAPP_BOT_JID", "120363426178035051@g.us")
_LISTENER_NOTES_CHAT_JID = os.environ.get(
    "WA_LISTENER_NOTES_CHAT_JID", "5493425153999-1539438783@g.us",
)
HARDCODED_EXCLUDE_JIDS = frozenset({
    "status@broadcast",
    _BOT_JID,
    _LISTENER_NOTES_CHAT_JID,
})

# U+200B (zero-width space): the listener bot prefixes every outbound message
# with it. Any bridge row starting with it is bot output, not human content.
_ANTILOOP_MARKER = "​"

_PHONE_DIGITS = re.compile(r"\d+")
_SLUG_NON_WORD = re.compile(r"[^\w\s-]")
_SLUG_WS = re.compile(r"[\s_]+")


# ── Data types ─────────────────────────────────────────────────────────────

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


@dataclass(frozen=True)
class WAChunk:
    chat_jid: str
    chat_name: str
    sender: str                   # dominant speaker label for the chunk
    is_from_me: bool              # whether the chunk's first message is self-sent
    first_msg_id: str
    last_msg_id: str
    first_ts: float
    last_ts: float
    body: str                     # canonical display text (Speaker: line format)
    parent: str                   # ±N-message window for rerank context


# ── Reader ─────────────────────────────────────────────────────────────────

def _parse_bridge_ts(raw: object) -> float | None:
    """Parse the bridge's `timestamp` into epoch seconds. The Go bridge writes
    RFC3339 (string); some older rows are numeric. Returns None on garbage."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if not s:
        return None
    # Trim fractional seconds beyond microseconds (Go writes nanoseconds).
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
    """Load messages from the bridge DB (read-only). `since_ts` filters by epoch
    (exclusive). `chat_jid` restricts to one chat (mainly for tests). Skips
    empty-content rows, excluded chats, and bot anti-loop output. Sorted by
    (chat_jid, timestamp)."""
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


# ── Chunker ────────────────────────────────────────────────────────────────

def _mask_phone(sender: str) -> str:
    """Mask a sender JID to its last-4 digits to avoid leaking full numbers."""
    digits = "".join(_PHONE_DIGITS.findall(sender))
    if len(digits) >= 4:
        return f"…{digits[-4:]}"
    return "?"


def _speaker_label(msg: WAMessage) -> str:
    """Human label for a message author. No rag/dossier dependency:
      - self     → "yo"
      - 1:1 chat → contact name from `chats.name` (the chat_name)
      - group    → masked last-4 of the sender's phone
      - fallback → "?"
    """
    if msg.is_from_me:
        return "yo"
    # 1:1 chats are anything that isn't a group (`@g.us`). status@broadcast is
    # already excluded upstream. For DMs the chat's display name is the contact.
    is_group = msg.chat_jid.endswith("@g.us")
    if not is_group and msg.chat_name and msg.chat_name != msg.chat_jid:
        return msg.chat_name
    return _mask_phone(msg.sender)


def _render_window(messages: list[WAMessage]) -> str:
    """Format messages as a `Speaker: content` transcript."""
    return "\n".join(f"{_speaker_label(m)}: {m.content.strip()}" for m in messages)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n", 0, max_chars)
    if cut < max_chars // 2:
        cut = max_chars
    return text[:cut].rstrip()


def _merge_tiny_groups(
    groups: list[list[WAMessage]],
    min_chars: int,
    window_s: float,
    max_chars: int,
) -> list[list[WAMessage]]:
    """Merge groups whose body is < min_chars into the closest neighbor within
    `window_s` when the combined body stays under `max_chars`. Prefers the
    previous (time-causal) group."""
    if not groups:
        return []
    out = list(groups)
    i = 0
    while i < len(out):
        body_len = sum(len(m.content) + 10 for m in out[i])
        if body_len >= min_chars or len(out[i]) == 0:
            i += 1
            continue
        prev = out[i - 1] if i > 0 else None
        nxt = out[i + 1] if i + 1 < len(out) else None
        if prev is not None:
            gap = out[i][0].timestamp - prev[-1].timestamp
            combined = sum(len(m.content) + 10 for m in prev) + body_len
            if gap <= window_s and combined <= max_chars:
                prev.extend(out[i])
                out.pop(i)
                continue
        if nxt is not None:
            gap = nxt[0].timestamp - out[i][-1].timestamp
            combined = body_len + sum(len(m.content) + 10 for m in nxt)
            if gap <= window_s and combined <= max_chars:
                out[i].extend(nxt)
                out.pop(i + 1)
                i += 1
                continue
        i += 1
    return out


def chunk_conversation(
    messages: Iterable[WAMessage],
    *,
    same_speaker_window_s: float = CHUNK_SAME_SPEAKER_WINDOW_S,
    min_merge_chars: int = CHUNK_MIN_MERGE_CHARS,
    max_chars: int = CHUNK_MAX_CHARS,
) -> list[list[WAMessage]]:
    """Group messages within each chat into chunks. Start a new group on speaker
    change, a gap ≥ window, or an overflow past max_chars; then merge tiny
    groups. Grouping is strictly intra-chat."""
    by_chat: dict[str, list[WAMessage]] = {}
    for m in messages:
        by_chat.setdefault(m.chat_jid, []).append(m)

    groups: list[list[WAMessage]] = []
    for _jid, msgs in by_chat.items():
        msgs.sort(key=lambda m: m.timestamp)
        local: list[list[WAMessage]] = []
        current: list[WAMessage] = []
        current_len = 0
        for m in msgs:
            start_new = False
            if current:
                prev = current[-1]
                gap = m.timestamp - prev.timestamp
                if gap >= same_speaker_window_s:
                    start_new = True
                elif m.sender != prev.sender or m.is_from_me != prev.is_from_me:
                    start_new = True
                elif current_len + len(m.content) + 10 > max_chars:
                    start_new = True
            if start_new:
                local.append(current)
                current = [m]
                current_len = len(m.content) + 10
            else:
                current.append(m)
                current_len += len(m.content) + 10
        if current:
            local.append(current)
        groups.extend(
            _merge_tiny_groups(local, min_merge_chars, same_speaker_window_s, max_chars)
        )
    return groups


def build_chunks(
    messages: list[WAMessage],
    *,
    same_speaker_window_s: float = CHUNK_SAME_SPEAKER_WINDOW_S,
    min_merge_chars: int = CHUNK_MIN_MERGE_CHARS,
    max_chars: int = CHUNK_MAX_CHARS,
    parent_window: int = PARENT_WINDOW_MESSAGES,
    parent_max_chars: int = PARENT_MAX_CHARS,
) -> list[WAChunk]:
    """group → render body + ±N parent window → WAChunk list. `parent` uses the
    surrounding messages (not just the chunk's own) so the reranker has context
    for short messages like "dale" / "ok mañana"."""
    if not messages:
        return []
    by_chat_sorted: dict[str, list[WAMessage]] = {}
    for m in messages:
        by_chat_sorted.setdefault(m.chat_jid, []).append(m)
    for lst in by_chat_sorted.values():
        lst.sort(key=lambda x: x.timestamp)
    id_to_pos: dict[tuple[str, str], int] = {}
    for jid, lst in by_chat_sorted.items():
        for i, m in enumerate(lst):
            id_to_pos[(jid, m.id)] = i

    groups = chunk_conversation(
        messages, same_speaker_window_s=same_speaker_window_s,
        min_merge_chars=min_merge_chars, max_chars=max_chars,
    )

    out: list[WAChunk] = []
    for grp in groups:
        if not grp:
            continue
        first, last = grp[0], grp[-1]
        body = _truncate(_render_window(grp), max_chars)

        lst = by_chat_sorted[first.chat_jid]
        pos = id_to_pos[(first.chat_jid, first.id)]
        lo = max(0, pos - parent_window)
        hi = min(len(lst), pos + parent_window + 1)
        parent = _truncate(_render_window(lst[lo:hi]), parent_max_chars)

        out.append(WAChunk(
            chat_jid=first.chat_jid,
            chat_name=first.chat_name,
            sender=_speaker_label(first),
            is_from_me=first.is_from_me,
            first_msg_id=first.id,
            last_msg_id=last.id,
            first_ts=first.timestamp,
            last_ts=last.timestamp,
            body=body,
            parent=parent,
        ))
    return out


# ── Cursor (incremental dedup) ───────────────────────────────────────────────

_STATE_FILENAME = "whatsapp_ingest_state.json"


def _state_path(state_dir: Path) -> Path:
    return state_dir / _STATE_FILENAME


def _load_cursors(state_dir: Path) -> dict[str, dict[str, Any]]:
    p = _state_path(state_dir)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        cursors = data.get("cursors", {})
        return cursors if isinstance(cursors, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_cursors(state_dir: Path, cursors: dict[str, dict[str, Any]]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "cursors": cursors,
        "updated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
    }
    tmp = _state_path(state_dir).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, _state_path(state_dir))


def _reset_cursors(state_dir: Path) -> None:
    p = _state_path(state_dir)
    if p.is_file():
        p.unlink()


# ── Orchestration ────────────────────────────────────────────────────────────

def _slug(s: str) -> str:
    s = s.lower().strip()
    s = _SLUG_NON_WORD.sub("", s)
    s = _SLUG_WS.sub("-", s)
    return s.strip("-") or "chat"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat(timespec="milliseconds")


def run(
    mem: Any,
    *,
    bridge_db: Path | None = None,
    since: str | None = None,
    reset: bool = False,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    include_chats: tuple[str, ...] = (),
    exclude_chats: tuple[str, ...] = (),
    all_chats: bool = False,
    min_chars: int = 1,
    max_chats: int | None = None,
    max_messages: int | None = None,
    dry_run: bool = False,
    preserve_timestamps: bool = False,
    reindex: bool = True,
) -> dict[str, Any]:
    """Ingest WhatsApp chunks into memo. Returns a summary dict.

    Scope is opt-in: pass `include_chats` (allowlist) or `all_chats=True`.
    Refuses to run with neither, to avoid an accidental full-corpus dump.
    """
    if not include_chats and not all_chats:
        raise ValueError(
            "scope required: pass include_chats=(...) or all_chats=True "
            "(refusing to ingest every chat by default)"
        )

    db = bridge_db or DEFAULT_BRIDGE_DB
    if not db.is_file():
        raise FileNotFoundError(f"bridge DB not found: {db}")

    state_dir = Path(mem.cfg.state_dir)
    if reset:
        _reset_cursors(state_dir)
    cursors = _load_cursors(state_dir)

    exclude = HARDCODED_EXCLUDE_JIDS | set(exclude_chats)
    include = set(include_chats)

    # Global floor: max(retention window, explicit --since).
    retention_floor = time.time() - retention_days * 86400
    since_ts = retention_floor
    if since:
        try:
            since_dt = datetime.strptime(since, "%Y-%m-%d")
            since_ts = max(since_ts, since_dt.timestamp())
        except ValueError as exc:
            raise ValueError(f"--since must be YYYY-MM-DD, got {since!r}") from exc

    messages = read_messages(db, since_ts=since_ts, exclude_jids=frozenset(exclude))

    # Allowlist + per-chat cursor + cap filters.
    seen_chats: list[str] = []
    filtered: list[WAMessage] = []
    for m in messages:
        if include and m.chat_jid not in include:
            continue
        cur = cursors.get(m.chat_jid, {})
        if m.timestamp <= float(cur.get("last_ts", 0.0)):
            continue
        if max_chats is not None and m.chat_jid not in seen_chats:
            if len(seen_chats) >= max_chats:
                continue
            seen_chats.append(m.chat_jid)
        filtered.append(m)
        if max_messages is not None and len(filtered) >= max_messages:
            break

    chunks = build_chunks(filtered)
    chunks = [c for c in chunks if len(c.body) >= min_chars]

    summary: dict[str, Any] = {
        "bridge_db": str(db),
        "messages_read": len(messages),
        "messages_after_filter": len(filtered),
        "chunks_built": len(chunks),
        "chunks_saved": 0,
        "chats": sorted({c.chat_jid for c in chunks}),
        "dry_run": dry_run,
        "reindexed": 0,
    }

    if dry_run or not chunks:
        return summary

    # Save chunks, tracking the max ingested ts per chat for cursor advancement.
    latest: dict[str, tuple[float, str]] = {}
    saved = 0
    for c in chunks:
        date_str = _iso(c.first_ts)[:10]
        extra = {
            "source": "whatsapp",
            "chat_jid": c.chat_jid,
            "chat_name": c.chat_name,
            "sender": c.sender,
            "is_from_me": c.is_from_me,
            "first_msg_id": c.first_msg_id,
            "last_msg_id": c.last_msg_id,
            "wa_first_ts": c.first_ts,
            "wa_last_ts": c.last_ts,
            "wa_first_iso": _iso(c.first_ts),
            "parent": c.parent,
        }
        mem.save(
            content=c.body,
            title=f"WhatsApp · {c.chat_name} · {date_str}",
            type_="reference",
            tags=["whatsapp", "chunk", f"wa-chat:{_slug(c.chat_name)}"],
            extra=extra,
            created=_iso(c.first_ts) if preserve_timestamps else None,
            defer_embed=True,
            auto_project=False,
            skip_memflow_receipt=True,
        )
        saved += 1
        prev = latest.get(c.chat_jid)
        if prev is None or c.last_ts > prev[0]:
            latest[c.chat_jid] = (c.last_ts, c.last_msg_id)

    # Advance cursors only for chats that actually saved chunks.
    for jid, (ts, msg_id) in latest.items():
        cursors[jid] = {"last_ts": ts, "last_msg_id": msg_id}
    _save_cursors(state_dir, cursors)
    summary["chunks_saved"] = saved

    if reindex:
        try:
            stats = mem.reindex()
            summary["reindexed"] = int((stats or {}).get("reindexed", 0))
            summary["reindex_stats"] = stats
        except Exception as exc:  # noqa: BLE001 — reindex is best-effort here
            summary["reindex_error"] = str(exc)

    return summary
