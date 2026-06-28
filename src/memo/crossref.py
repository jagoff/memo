"""EXPERIMENTAL — not covered by the test suite, not exposed via MCP. API may change without notice.

Cross-reference and backlink system for memories.

Detects and manages wikilinks between memories, enabling:
- Automatic detection of [[wikilinks]] in memory content
- Backlink queries (what memories reference this one)
- Link suggestions when saving (suggest linking to related memories)
- Link graph visualization

## Wikilink Detection

Parses memory content for Obsidian-style wikilinks [[memory-id]] or
[[memory-id|alias]]. Stores these in a separate index for fast backlink
queries.

## Backlink Index

A separate table in the vec store (or a new DB) that tracks:
- source_id: memory that contains the link
- target_id: memory being referenced
- link_type: wikilink, entity mention, etc.

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
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_WIKILINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")


@dataclass
class Wikilink:
    """A detected wikilink in memory content."""

    target: str  # The memory ID or name being linked to
    alias: str | None  # Optional display alias
    position: int  # Character position in content


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

    memoria_id: str
    title: str
    similarity: float
    reason: str  # Why this link is suggested


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

    def index_wikilinks(self, memoria_id: str, content: str) -> list[Wikilink]:
        """Detect and index wikilinks in memory content.

        Args:
            memoria_id: The ID of the memory containing the links.
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
                        memoria_id,
                        link.target,
                        content[max(0, link.position - 50) : link.position + 50],
                    )
                    for link in wikilinks
                ],
            )

        return wikilinks

    def get_backlinks(self, memoria_id: str) -> list[Backlink]:
        """Get all memories that reference this one.

        Args:
            memoria_id: The memory ID to find backlinks for.

        Returns:
            List of Backlink objects.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT source_id, target_id, link_type, context FROM backlinks WHERE target_id = ?",
            (memoria_id,),
        ).fetchall()

        backlinks = []
        for row in rows:
            backlinks.append(
                Backlink(
                    source_id=row["source_id"],
                    source_title="",  # Would need to fetch from memory store
                    target_id=row["target_id"],
                    link_type=row["link_type"],
                    context=row["context"] or "",
                )
            )

        return backlinks

    def get_outlinks(self, memoria_id: str) -> list[Wikilink]:
        """Get all memories that this one references.

        Args:
            memoria_id: The memory ID to find outlinks for.

        Returns:
            List of Wikilink objects.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT target_id, link_type, context FROM backlinks WHERE source_id = ?",
            (memoria_id,),
        ).fetchall()

        outlinks = []
        for row in rows:
            outlinks.append(
                Wikilink(
                    target=row["target_id"],
                    alias=None,
                    position=0,
                )
            )

        return outlinks

    def remove_memoria(self, memoria_id: str) -> None:
        """Remove all links for a memory (when deleted).

        Args:
            memoria_id: The memory ID to remove.
        """
        with self._tx() as conn:
            conn.execute("DELETE FROM backlinks WHERE source_id = ?", (memoria_id,))
            conn.execute("DELETE FROM backlinks WHERE target_id = ?", (memoria_id,))

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
        with suppress(Exception):
            self._conn.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        with suppress(Exception):
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
                        memoria_id=hit.id,
                        title=hit.title,
                        similarity=hit.score,
                        reason=f"High semantic similarity ({hit.score:.2f})",
                    )
                )

        # Strategy 2: Shared entities
        # Extract entities from content (would need LLM, placeholder for now)
        # For now, skip entity-based suggestions

        # Deduplicate by memoria_id
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s.memoria_id not in seen:
                seen.add(s.memoria_id)
                unique_suggestions.append(s)

        return unique_suggestions[:limit]

    def format_wikilink(self, memoria_id: str, title: str | None = None) -> str:
        """Format a memory ID as a wikilink.

        Args:
            memoria_id: The memory ID to link to.
            title: Optional display title (defaults to memoria_id).

        Returns:
            Wikilink string like [[memory-id]] or [[memory-id|Title]].
        """
        if title and title != memoria_id:
            return f"[[{memoria_id}|{title}]]"
        return f"[[{memoria_id}]]"


__all__ = [
    "Backlink",
    "CrossReferenceIndex",
    "LinkSuggester",
    "LinkSuggestion",
    "Wikilink",
]
