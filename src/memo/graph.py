"""Knowledge graph — entity index over memorias.

Schema lives in `~/.local/share/memo/graph.db` (separate file so write
load doesn't share WAL with the hot vec store):

```
CREATE TABLE entities (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,        -- canonical form, lower-cased
    type          TEXT NOT NULL,        -- person | project | technology | file | org | concept
    mention_count INTEGER NOT NULL DEFAULT 0,
    first_seen    TEXT,                 -- ISO date of earliest memoria mention
    last_seen     TEXT,                 -- ISO date of latest memoria mention
    UNIQUE(name, type)
);

CREATE TABLE entity_memoria (
    entity_id     INTEGER NOT NULL,
    memoria_id    TEXT NOT NULL,
    occurrences   INTEGER NOT NULL DEFAULT 1,  -- not really tracked; reserved
    extracted_at  TEXT NOT NULL,
    UNIQUE(entity_id, memoria_id)
);

CREATE INDEX idx_em_memoria ON entity_memoria(memoria_id);
CREATE INDEX idx_em_entity  ON entity_memoria(entity_id);
CREATE INDEX idx_e_type     ON entities(type);
CREATE INDEX idx_e_mc       ON entities(mention_count);
```

Why a separate DB:
- The vec store is hot-path read-heavy; entity writes are a batch job
  that runs occasionally (`memo extract-entities`). Splitting WAL
  avoids contention.
- A graph corruption never threatens search retrieval. Reset is safe:
  `rm graph.db && memo extract-entities --all`.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS entities (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    type          TEXT NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 0,
    first_seen    TEXT,
    last_seen     TEXT,
    UNIQUE(name, type)
);

CREATE TABLE IF NOT EXISTS entity_memoria (
    entity_id     INTEGER NOT NULL,
    memoria_id    TEXT NOT NULL,
    occurrences   INTEGER NOT NULL DEFAULT 1,
    extracted_at  TEXT NOT NULL,
    UNIQUE(entity_id, memoria_id)
);

CREATE INDEX IF NOT EXISTS idx_em_memoria ON entity_memoria(memoria_id);
CREATE INDEX IF NOT EXISTS idx_em_entity  ON entity_memoria(entity_id);
CREATE INDEX IF NOT EXISTS idx_e_type     ON entities(type);
CREATE INDEX IF NOT EXISTS idx_e_mc       ON entities(mention_count);
"""


VALID_ENTITY_TYPES = frozenset({
    "person", "project", "technology", "file", "org", "concept",
})


@dataclass(frozen=True)
class EntityMention:
    """Entity linked to a memoria."""
    name: str
    type: str
    mention_count: int


class GraphStore:
    """Entity index. Append-only writes via `record_extraction`,
    read-only queries via `top_entities`, `entity_memorias`,
    `memoria_entities`.

    All writes idempotent on (memoria_id) — running extraction twice
    on the same memoria refreshes the link set without duplicating.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), timeout=10.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL = concurrent readers + a writer; one shared connection across the
        # FastMCP threadpool, so serialise `_tx()` — two threads issuing
        # BEGIN IMMEDIATE on the same connection raise "transaction within a
        # transaction". drop_for_memoria runs on the hot Memory.delete() path.
        with suppress(sqlite3.Error):
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._tx_lock = threading.Lock()
        with self._conn:
            self._conn.executescript(_SCHEMA_DDL)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._tx_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def record_extraction(
        self, *, memoria_id: str, memoria_date: str,
        entities: list[dict[str, str]], extracted_at: str,
    ) -> int:
        """Idempotently link a memoria to its extracted entities.

        Steps under one tx:
        1. Drop any existing entity_memoria links for this memoria
           (so re-extraction reflects current state).
        2. Upsert each (name, type) into `entities`.
        3. Insert entity_memoria links for all current entities.
        4. Update mention_count + first_seen/last_seen for each
           touched entity.

        Returns the number of links written.
        """
        n = 0
        with self._tx() as cx:
            # Get current entity_ids for this memoria so we can
            # decrement mention_count if any are removed.
            old_eids = [
                r["entity_id"] for r in cx.execute(
                    "SELECT entity_id FROM entity_memoria WHERE memoria_id = ?",
                    (memoria_id,),
                ).fetchall()
            ]
            cx.execute(
                "DELETE FROM entity_memoria WHERE memoria_id = ?",
                (memoria_id,),
            )
            # Decrement mention_count for old entities (will be
            # re-incremented if they're still present).
            for eid in old_eids:
                cx.execute(
                    "UPDATE entities SET mention_count = MAX(0, mention_count - 1) "
                    "WHERE id = ?", (eid,),
                )

            for ent in entities:
                name = (ent.get("name") or "").strip().lower()
                etype = (ent.get("type") or "").strip().lower()
                if not name or etype not in VALID_ENTITY_TYPES:
                    continue
                # Upsert entity row.
                cx.execute(
                    "INSERT INTO entities (name, type, mention_count, first_seen, last_seen) "
                    "VALUES (?, ?, 0, ?, ?) ON CONFLICT(name, type) DO NOTHING",
                    (name, etype, memoria_date, memoria_date),
                )
                eid = cx.execute(
                    "SELECT id, first_seen, last_seen FROM entities WHERE name = ? AND type = ?",
                    (name, etype),
                ).fetchone()
                if eid is None:
                    continue
                cx.execute(
                    "INSERT OR IGNORE INTO entity_memoria "
                    "(entity_id, memoria_id, occurrences, extracted_at) "
                    "VALUES (?, ?, 1, ?)",
                    (eid["id"], memoria_id, extracted_at),
                )
                # Bump mention_count + adjust first/last_seen.
                cx.execute(
                    "UPDATE entities SET mention_count = mention_count + 1, "
                    "  first_seen = MIN(first_seen, ?), "
                    "  last_seen  = MAX(last_seen, ?) "
                    "WHERE id = ?",
                    (memoria_date, memoria_date, eid["id"]),
                )
                n += 1
        return n

    def drop_for_memoria(self, memoria_id: str) -> int:
        """Called when a memoria is deleted. Removes all entity_memoria
        edges for it and decrements mention_count on each touched
        entity. Returns the number of edges removed."""
        with self._tx() as cx:
            old = [
                r["entity_id"] for r in cx.execute(
                    "SELECT entity_id FROM entity_memoria WHERE memoria_id = ?",
                    (memoria_id,),
                ).fetchall()
            ]
            cx.execute(
                "DELETE FROM entity_memoria WHERE memoria_id = ?", (memoria_id,),
            )
            for eid in old:
                cx.execute(
                    "UPDATE entities SET mention_count = MAX(0, mention_count - 1) "
                    "WHERE id = ?", (eid,),
                )
        return len(old)

    def top_entities(
        self, *, limit: int = 50, type_: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT name, type, mention_count, first_seen, last_seen FROM entities"
        params: list[Any] = []
        if type_:
            sql += " WHERE type = ?"
            params.append(type_)
        sql += " ORDER BY mention_count DESC, last_seen DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def entity_memorias(self, name: str, type_: str | None = None) -> list[str]:
        """Memoria IDs that mention `name` (and optionally a specific type)."""
        name = name.strip().lower()
        params: tuple[str, ...]
        if type_:
            type_ = type_.strip().lower()
            sql = (
                "SELECT em.memoria_id FROM entity_memoria em "
                "JOIN entities e ON e.id = em.entity_id "
                "WHERE e.name = ? AND e.type = ?"
            )
            params = (name, type_)
        else:
            sql = (
                "SELECT em.memoria_id FROM entity_memoria em "
                "JOIN entities e ON e.id = em.entity_id "
                "WHERE e.name = ?"
            )
            params = (name,)
        return [r["memoria_id"] for r in self._conn.execute(sql, params).fetchall()]

    def memoria_entities(self, memoria_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT e.name, e.type, e.mention_count "
            "FROM entity_memoria em JOIN entities e ON e.id = em.entity_id "
            "WHERE em.memoria_id = ? "
            "ORDER BY e.mention_count DESC",
            (memoria_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_entity_mentions(self, memoria_id: str) -> list[EntityMention]:
        """Return entity mentions for compatibility with analytics callers."""
        return [
            EntityMention(
                name=row["name"],
                type=row["type"],
                mention_count=int(row["mention_count"]),
            )
            for row in self.memoria_entities(memoria_id)
        ]

    def count_entities(self) -> int:
        """Return the number of unique entities in the graph."""
        return int(self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0])

    def stats(self) -> dict[str, int]:
        n_entities = self.count_entities()
        n_links = self._conn.execute("SELECT COUNT(*) FROM entity_memoria").fetchone()[0]
        return {"entities": n_entities, "links": n_links}

    def close(self) -> None:
        with suppress(Exception):
            self._conn.close()


__all__ = ["VALID_ENTITY_TYPES", "EntityMention", "GraphStore"]
