"""Cross-reference and backlink system for memories (typed edges + backlinks).

Detects and manages wikilinks between memories, enabling:
- Automatic detection of [[wikilinks]] in memory content
- Typed `- relation_type [[target]]` edges (basic-memory style grammar)
- Backlink queries (what memories reference this one)
- Prefix-aware reverse traversal (hand-authored short prefixes match full IDs)
- Link suggestions when saving (suggest linking to related memories)

## Typed Link Grammar

In addition to bare ``[[wikilinks]]``, content may contain typed list items::

    - supersedes [[aaaaaaaa1111000000000000000000ff]]
    - caused_by [[bbbbbbbb2222000000000000000000ff|the OOM bug]]

``relation_type`` must be a lowercase snake_case word (2–32 chars, starting with
a letter).  Typed lines win the de-dup: the ``[[target]]`` inside them is NOT
double-counted as a bare ``wikilink``.  Hand-edited typed edges in the markdown
survive a reindex because ``index_source()`` does a delete-then-insert on the
source's rows.

## Backlink Index

A separate table tracks:
- source_id: memory that contains the link
- target_id: memory being referenced (stored as-is, may be a short prefix)
- link_type: 'wikilink' or a typed relation (supersedes, caused_by, …)

## Link Suggestions

When saving a memory, suggests linking to existing memories based on:
- Semantic similarity (high similarity = potential duplicate or related)
- Shared entities
- Recent context (if available)
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_WIKILINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")

# Typed link grammar (basic-memory style): a list item whose whole content is
# `- relation_type [[target]]`. relation_type: lowercase snake_case, 2-32 chars.
_TYPED_LINK_PATTERN = re.compile(
    r"^\s{0,3}[-*]\s+([a-z][a-z0-9_]{1,31})\s+\[\[([^\]]+)\]\]\s*$",
    re.MULTILINE,
)


@dataclass
class Wikilink:
    """A detected wikilink in memory content."""

    target: str  # The memory ID or name being linked to
    alias: str | None  # Optional display alias
    position: int  # Character position in content
    link_type: str = "wikilink"  # 'wikilink' or a typed relation (supersedes, caused_by, ...)


@dataclass
class Backlink:
    """A backlink to a memory."""

    source_id: str
    source_title: str
    target_id: str
    link_type: str  # wikilink, entity, etc.
    context: str  # Snippet of context around the link


@dataclass
class LinkSuggestion:
    """A suggestion to link to an existing memory."""

    memory_id: str
    title: str
    similarity: float
    reason: str  # Why this link is suggested


def _split_alias(target_raw: str) -> tuple[str, str | None]:
    if "|" in target_raw:
        target, alias = target_raw.split("|", 1)
        return target.strip(), alias.strip()
    return target_raw.strip(), None


def parse_links(content: str) -> list[Wikilink]:
    """Parse typed `- relation_type [[target]]` lines plus bare [[wikilinks]].

    Typed lines win: their [[target]] is not double-counted as a bare link.
    Hand-edited edges in the markdown survive reindex (markdown is truth).
    """
    links: list[Wikilink] = []
    typed_spans: list[tuple[int, int]] = []
    for m in _TYPED_LINK_PATTERN.finditer(content):
        target, alias = _split_alias(m.group(2))
        links.append(
            Wikilink(target=target, alias=alias, position=m.start(2), link_type=m.group(1))
        )
        typed_spans.append(m.span())
    for m in _WIKILINK_PATTERN.finditer(content):
        if any(start <= m.start() < end for start, end in typed_spans):
            continue
        target, alias = _split_alias(m.group(1))
        links.append(Wikilink(target=target, alias=alias, position=m.start()))
    return links


def source_titles_via(get: Callable[[str], Any]) -> Callable[[list[str]], dict[str, str]]:
    """Build a batched ``title_resolver`` for :meth:`get_backlinks` from a
    single-id ``get`` callable (e.g. ``Memory.get``). Dedups the ids so each is
    fetched once; a missing/``None`` record maps to ``""``."""

    def _resolve(ids: list[str]) -> dict[str, str]:
        titles: dict[str, str] = {}
        for sid in dict.fromkeys(ids):
            rec = get(sid)
            titles[sid] = (getattr(rec, "title", "") or "") if rec is not None else ""
        return titles

    return _resolve


class CrossReferenceIndex:
    """Index for cross-references and backlinks.

    Args:
        db_path: Path to the cross-reference SQLite database.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._tx_lock = threading.Lock()
        # Open one shared connection eagerly (check_same_thread=False so it
        # survives the FastMCP worker threadpool). Eager init + _tx_lock kills
        # the lazy-init race where two threads each opened a connection, and
        # serialises writes via BEGIN IMMEDIATE — matching GraphStore.
        self._conn = sqlite3.connect(str(db_path), timeout=10.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with suppress(sqlite3.Error):
            self._conn.execute("PRAGMA journal_mode=WAL")
            # Bound the WAL: long-lived readers (daemons, MCP sessions) pin
            # snapshots, so a passive checkpoint never truncates on its own
            # (graph.db-wal was found at 80MB against a 127MB database).
            self._conn.execute("PRAGMA journal_size_limit=16777216")
        try:
            self._init_schema()
        except Exception:
            self.close()
            raise

    def _get_conn(self) -> sqlite3.Connection:
        return self._conn

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        # One shared connection across the FastMCP threadpool: two threads
        # issuing BEGIN IMMEDIATE concurrently would raise "transaction within a
        # transaction", so serialise writes on _tx_lock (GraphStore pattern).
        with self._tx_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def _init_schema(self) -> None:
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backlinks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                link_type TEXT NOT NULL DEFAULT 'wikilink',
                context TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(source_id, target_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_backlinks_source ON backlinks(source_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_backlinks_target ON backlinks(target_id)")
        conn.commit()

    def index_wikilinks(self, memory_id: str, content: str) -> list[Wikilink]:
        """Detect and index wikilinks in memory content.

        Args:
            memory_id: The ID of the memory containing the links.
            content: The content to parse for wikilinks.

        Returns:
            List of detected Wikilink objects.
        """
        wikilinks = []
        for match in _WIKILINK_PATTERN.finditer(content):
            target_raw = match.group(1)
            # Parse [[target]] or [[target|alias]]
            if "|" in target_raw:
                target, alias = target_raw.split("|", 1)
            else:
                target = target_raw
                alias = None

            wikilinks.append(
                Wikilink(
                    target=target.strip(),
                    alias=alias.strip() if alias else None,
                    position=match.start(),
                )
            )

        # Store in index — one executemany in a single transaction (target is
        # stored as-is; it could be an ID or a title resolved later).
        with self._tx() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO backlinks
                (source_id, target_id, link_type, context, created_at)
                VALUES (?, ?, 'wikilink', ?, datetime('now'))
                """,
                [
                    (
                        memory_id,
                        link.target,
                        content[max(0, link.position - 50) : link.position + 50],
                    )
                    for link in wikilinks
                ],
            )

        return wikilinks

    def index_source(self, memory_id: str, content: str) -> list[Wikilink]:
        """Re-index all links for one source memory (delete-then-insert).

        Unlike `index_wikilinks` (append-only, legacy), this REPLACES the
        source's rows so links removed from the markdown disappear from the
        index — required by the save/update/reindex wiring.
        """
        links = parse_links(content)
        with self._tx() as conn:
            conn.execute("DELETE FROM backlinks WHERE source_id = ?", (memory_id,))
            conn.executemany(
                """
                INSERT OR REPLACE INTO backlinks
                (source_id, target_id, link_type, context, created_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                """,
                [
                    (
                        memory_id,
                        link.target,
                        link.link_type,
                        content[max(0, link.position - 50) : link.position + 50],
                    )
                    for link in links
                ],
            )
        return links

    def referencing_sources(self, memory_id: str) -> list[Backlink]:
        """Reverse traversal: sources whose stored target is this id or a
        >=8-char prefix of it (hand-authored links use short prefixes)."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT source_id, target_id, link_type, context FROM backlinks
            WHERE target_id = ?
               OR (length(target_id) >= 8 AND ? LIKE (target_id || '%'))
            """,
            (memory_id, memory_id),
        ).fetchall()
        return [
            Backlink(
                source_id=row["source_id"],
                source_title="",
                target_id=row["target_id"],
                link_type=row["link_type"],
                context=row["context"] or "",
            )
            for row in rows
        ]

    def get_backlinks(
        self,
        memory_id: str,
        *,
        title_resolver: Callable[[list[str]], dict[str, str]] | None = None,
    ) -> list[Backlink]:
        """Get all memories that reference this one.

        Args:
            memory_id: The memory ID to find backlinks for.
            title_resolver: Optional batched (source_ids -> {id: title}) lookup
                used to populate ``source_title`` (crossref stores only ids).
                Build one from ``Memory.get`` via :func:`source_titles_via`.

        Returns:
            List of Backlink objects.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT source_id, target_id, link_type, context FROM backlinks WHERE target_id = ?",
            (memory_id,),
        ).fetchall()

        source_ids = [row["source_id"] for row in rows]
        titles = title_resolver(source_ids) if (title_resolver and source_ids) else {}

        backlinks = []
        for row in rows:
            backlinks.append(
                Backlink(
                    source_id=row["source_id"],
                    source_title=titles.get(row["source_id"], ""),
                    target_id=row["target_id"],
                    link_type=row["link_type"],
                    context=row["context"] or "",
                )
            )

        return backlinks

    def get_outlinks(self, memory_id: str) -> list[Wikilink]:
        """Get all memories that this one references.

        Args:
            memory_id: The memory ID to find outlinks for.

        Returns:
            List of Wikilink objects.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT target_id, link_type, context FROM backlinks WHERE source_id = ?",
            (memory_id,),
        ).fetchall()

        outlinks = []
        for row in rows:
            outlinks.append(
                Wikilink(
                    target=row["target_id"],
                    alias=None,
                    position=0,
                    link_type=row["link_type"],
                )
            )

        return outlinks

    def remove_memoria(self, memory_id: str) -> None:
        """Remove all links for a memory (when deleted).

        Args:
            memory_id: The memory ID to remove.
        """
        with self._tx() as conn:
            conn.execute("DELETE FROM backlinks WHERE source_id = ?", (memory_id,))
            conn.execute("DELETE FROM backlinks WHERE target_id = ?", (memory_id,))

    def reset(self) -> None:
        """Clear the whole cross-reference index (all backlinks).

        Truncates the table rather than deleting the DB file — safe when the
        crossref tables share the main DB (`single_db`), where unlinking the
        file would destroy everything. Used by `memo links reindex`.
        """
        with self._tx() as conn:
            conn.execute("DELETE FROM backlinks")

    def close(self) -> None:
        """Close the database connection."""
        with suppress(BaseException):
            self._conn.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        with suppress(BaseException):
            self.close()


class LinkSuggester:
    """Suggests links to existing memories when saving.

    Args:
        memory: The Memory instance to search.
        crossref: The CrossReferenceIndex for existing links.
    """

    def __init__(self, memory: Any, crossref: CrossReferenceIndex) -> None:
        self.memory = memory
        self.crossref = crossref

    def suggest_links(
        self,
        content: str,
        title: str,
        tags: list[str],
        limit: int = 5,
    ) -> list[LinkSuggestion]:
        """Suggest links to existing memories based on content.

        Args:
            content: The memory content being saved.
            title: The memory title.
            tags: The memory tags.
            limit: Maximum suggestions to return.

        Returns:
            List of LinkSuggestion objects.
        """
        suggestions = []

        # Strategy 1: Semantic similarity search
        hits = self.memory.search(content, limit=limit * 2, mode="vec")
        for hit in hits[:limit]:
            if hit.score > 0.7:  # Only suggest high similarity
                suggestions.append(
                    LinkSuggestion(
                        memory_id=hit.id,
                        title=hit.title,
                        similarity=hit.score,
                        reason=f"High semantic similarity ({hit.score:.2f})",
                    )
                )

        # Deduplicate by memory_id
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s.memory_id not in seen:
                seen.add(s.memory_id)
                unique_suggestions.append(s)

        return unique_suggestions[:limit]

    def format_wikilink(self, memory_id: str, title: str | None = None) -> str:
        """Format a memory ID as a wikilink.

        Args:
            memory_id: The memory ID to link to.
            title: Optional display title (defaults to memory_id).

        Returns:
            Wikilink string like [[memory-id]] or [[memory-id|Title]].
        """
        if title and title != memory_id:
            return f"[[{memory_id}|{title}]]"
        return f"[[{memory_id}]]"


__all__ = [
    "Backlink",
    "CrossReferenceIndex",
    "LinkSuggester",
    "LinkSuggestion",
    "Wikilink",
    "parse_links",
]
