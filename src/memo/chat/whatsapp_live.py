"""Live WhatsApp bridge read-through for recency-intent chat queries.

The semantic memo index stores WhatsApp transcripts as ingested chunks and is
only as fresh as the last ingest run — structurally unable to answer "el
último mensaje con X" (it can't return a single message, can't guarantee time
order, and lags the live bridge DB). This module is the read-through
fallback: for a recency/last-message query about a WhatsApp contact, it reads
the bridge SQLite DB directly (``ORDER BY timestamp DESC LIMIT N``) so the
answer is message-granular and always current.

Read-only and WAL-visible — connections use ``mode=ro`` (NEVER
``immutable=1``, which hides freshly-WAL-written rows). Every public function
is fail-soft: any exception degrades to an empty result / ``False`` rather
than propagating, so a missing or locked bridge DB falls back to the normal
semantic path instead of breaking the chat response.
"""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .contacts_alias import resolve_jid

_DEFAULT_DB = "~/repos/whatsapp-mcp/whatsapp-bridge/store/messages.db"

# Generic words in a chat's display name that must not, on their own, count
# as a match trigger against a query (e.g. a chat named "Family Chat" should
# not match every query that happens to say "chat").
_GENERIC_NAME_TOKENS = {"group", "grupo", "chat", "family", "familia", "the", "los", "las"}

# Connectives / query words that must never be treated as a name-match
# trigger, so a chat whose name happens to contain one of these (e.g.
# "Mensajes Colegio") doesn't wrongly match every query that talks ABOUT
# messages/recency rather than naming that chat (e.g. "los últimos mensajes
# de Ana"). Ported verbatim from synapse's `_MATCH_STOPWORDS`.
_MATCH_STOPWORDS = {
    "con",
    "sin",
    "por",
    "para",
    "del",
    "una",
    "uno",
    "unos",
    "unas",
    "que",
    "cual",
    "cuales",
    "son",
    "mis",
    "esta",
    "este",
    "esa",
    "ese",
    "mas",
    "hoy",
    "today",
    "with",
    "and",
    "for",
    "the",
    "mensaje",
    "mensajes",
    "chats",
    "ultimo",
    "ultima",
    "ultimos",
    "ultimas",
    "last",
    "latest",
    "reciente",
    "recientes",
    "conversacion",
    "conversaciones",
    "conversation",
    "whatsapp",
    "dijo",
    "escribio",
}

_RECENCY_RE = re.compile(
    r"(últim[oa]s? (mensajes?|conversaci[oó]n)|qué (me )?dijo|"
    r"conversaci[oó]n con|last messages?|what did .{1,40} say)",
    re.IGNORECASE,
)
_TODAY_RE = re.compile(r"\b(hoy|today)\b", re.IGNORECASE)
_PLURAL_TOKENS = ("mensajes", "messages", "últimos", "ultimos", "últimas", "ultimas")
_SINGULAR_TOKENS = ("último mensaje", "ultimo mensaje", "last message")


def _name_tokens(name: str) -> list[str]:
    """Significant lowercase tokens of a chat name usable for query matching."""
    return [
        t
        for t in re.findall(r"[a-záéíóúñ]+", name.lower())
        if len(t) >= 3 and t not in _GENERIC_NAME_TOKENS and t not in _MATCH_STOPWORDS
    ]


def bridge_db_path() -> Path:
    """Resolve the bridge ``messages.db`` path (``MEMO_WHATSAPP_DB`` override)."""
    raw = os.environ.get("MEMO_WHATSAPP_DB", "").strip() or _DEFAULT_DB
    return Path(raw).expanduser()


@contextmanager
def _connect_ro(path: Path) -> Iterator[sqlite3.Connection]:
    # mode=ro (NOT immutable) so committed WAL rows stay visible.
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _chat_name_for_jid(conn: sqlite3.Connection, jid: str) -> str:
    row = conn.execute("SELECT name FROM chats WHERE jid = ? LIMIT 1", (jid,)).fetchone()
    return str(row["name"]) if row and row["name"] else jid.split("@")[0]


def _distinct_other_senders(conn: sqlite3.Connection, jid: str) -> int:
    """Distinct non-me senders in a chat (<=1 means effectively 1:1, even if a group)."""
    row = conn.execute(
        "SELECT COUNT(DISTINCT sender) AS n FROM messages "
        "WHERE chat_jid = ? AND is_from_me = 0 "
        "AND sender IS NOT NULL AND TRIM(sender) != ''",
        (jid,),
    ).fetchone()
    return int(row["n"]) if row else 0


def resolve_chats(query: str, db: Path, contacts_index: dict[str, str]) -> list[tuple[str, str]]:
    """Resolve chats matching a person/name in ``query`` to ``(jid, label)``.

    Priority: an exact contact match via ``contacts_alias.resolve_jid`` (the
    authoritative, hand-written contact book) wins outright and returns that
    one chat. Otherwise, every chat whose significant name token appears as a
    whole word in the query is a candidate; a ``@g.us`` group is only kept
    when it has at most one distinct non-me sender (otherwise a busy group
    would pollute a "conversation with X" query). Returns ``[]`` on no match,
    a missing DB, or any sqlite error.
    """
    q = (query or "").strip()
    if len(q) < 2:
        return []
    if not db.exists():
        return []
    try:
        with _connect_ro(db) as conn:
            jid = resolve_jid(query, contacts_index)
            if jid:
                return [(jid, _chat_name_for_jid(conn, jid))]
            ql = f" {q.lower()} "
            rows = conn.execute(
                "SELECT jid, name FROM chats WHERE name IS NOT NULL AND TRIM(name) != ''"
            ).fetchall()
            matched = [
                (str(r["jid"]), str(r["name"]))
                for r in rows
                if any(re.search(rf"\b{re.escape(t)}\b", ql) for t in _name_tokens(str(r["name"])))
            ]
            return [
                (jid, name)
                for jid, name in matched
                if not jid.endswith("@g.us") or _distinct_other_senders(conn, jid) <= 1
            ]
    except sqlite3.Error:
        return []


def last_messages(
    db: Path, chat_jid: str, *, limit: int = 10, today_only: bool = False
) -> list[dict[str, Any]]:
    """Newest ``limit`` non-empty messages for ``chat_jid``, oldest-to-newest.

    ``limit`` is clamped to ``[1, 200]`` when ``today_only`` else ``[1, 100]``.
    Returns ``[]`` on a missing DB or any sqlite error.
    """
    limit = max(1, min(int(limit), 200 if today_only else 100))
    if not db.exists():
        return []
    day_filter = "AND date(timestamp, 'localtime') = date('now', 'localtime')" if today_only else ""
    try:
        with _connect_ro(db) as conn:
            rows = conn.execute(
                f"""
                SELECT datetime(timestamp, 'localtime') AS ts,
                       is_from_me,
                       content
                  FROM messages
                 WHERE chat_jid = ?
                   AND content IS NOT NULL AND TRIM(content) != ''
                   {day_filter}
                 ORDER BY timestamp DESC
                 LIMIT ?
                """,  # noqa: S608 — day_filter is one of two hardcoded literals, no user input
                (chat_jid, limit),
            ).fetchall()
    except sqlite3.Error:
        return []
    msgs = [
        {
            "ts": str(r["ts"] or ""),
            "is_from_me": bool(r["is_from_me"]),
            "content": str(r["content"] or ""),
        }
        for r in rows
    ]
    msgs.reverse()  # chronological for natural reading / synthesis
    return msgs


def format_transcript(label: str, msgs: list[dict[str, Any]]) -> str:
    """Render ``msgs`` (as returned by :func:`last_messages`) as ``[ts] sender: content`` lines."""
    try:
        lines = [
            f"[{m.get('ts', '')}] {'yo' if m.get('is_from_me') else label}: {m.get('content', '')}"
            for m in msgs
        ]
    except (TypeError, AttributeError):
        return ""
    return "\n".join(lines)


def recency_conversation_intent(q: str) -> bool:
    """True when the query asks for recent messages / what someone said."""
    return bool(_RECENCY_RE.search(q or ""))


def singular_last_intent(q: str) -> bool:
    """True when the user asked for THE last message (singular), not a batch.

    "último mensaje" / "last message" signals singular; any plural marker
    ("mensajes", "últimos", ...) cancels it.
    """
    ql = (q or "").lower()
    if any(t in ql for t in _PLURAL_TOKENS):
        return False
    return any(t in ql for t in _SINGULAR_TOKENS)


def today_only_intent(q: str) -> bool:
    """True when the user scopes the query to *today* ("hoy" / "today")."""
    return bool(_TODAY_RE.search(q or ""))
