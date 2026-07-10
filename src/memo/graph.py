"""Knowledge graph — entity index over memories.

Schema lives in `~/.local/share/memo/graph.db` (separate file so write
load doesn't share WAL with the hot vec store):

```
CREATE TABLE entities (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,        -- canonical form, lower-cased
    type          TEXT NOT NULL,        -- person | project | technology | file | org | concept
    mention_count INTEGER NOT NULL DEFAULT 0,
    first_seen    TEXT,                 -- ISO date of earliest memory mention
    last_seen     TEXT,                 -- ISO date of latest memory mention
    UNIQUE(name, type)
);

CREATE TABLE entity_memory (
    entity_id     INTEGER NOT NULL,
    memory_id    TEXT NOT NULL,
    occurrences   INTEGER NOT NULL DEFAULT 1,  -- not really tracked; reserved
    extracted_at  TEXT NOT NULL,
    UNIQUE(entity_id, memory_id)
);

CREATE INDEX idx_em_memory ON entity_memory(memory_id);
CREATE INDEX idx_em_entity  ON entity_memory(entity_id);
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
from collections.abc import Iterator, Sequence
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

CREATE TABLE IF NOT EXISTS entity_memory (
    entity_id     INTEGER NOT NULL,
    memory_id    TEXT NOT NULL,
    occurrences   INTEGER NOT NULL DEFAULT 1,
    extracted_at  TEXT NOT NULL,
    UNIQUE(entity_id, memory_id)
);

CREATE INDEX IF NOT EXISTS idx_em_memory ON entity_memory(memory_id);
CREATE INDEX IF NOT EXISTS idx_em_entity  ON entity_memory(entity_id);
CREATE INDEX IF NOT EXISTS idx_e_type     ON entities(type);
CREATE INDEX IF NOT EXISTS idx_e_mc       ON entities(mention_count);

CREATE TABLE IF NOT EXISTS co_recall (
    id_a   TEXT NOT NULL,
    id_b   TEXT NOT NULL,
    count  INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (id_a, id_b)
);

CREATE INDEX IF NOT EXISTS idx_cr_a ON co_recall(id_a);
CREATE INDEX IF NOT EXISTS idx_cr_b ON co_recall(id_b);

CREATE TABLE IF NOT EXISTS entity_edges (
    a_id        INTEGER NOT NULL,
    b_id        INTEGER NOT NULL,
    weight      INTEGER NOT NULL DEFAULT 1,
    first_seen  TEXT,
    last_seen   TEXT,
    PRIMARY KEY (a_id, b_id)
);

CREATE INDEX IF NOT EXISTS idx_ee_a ON entity_edges(a_id);
CREATE INDEX IF NOT EXISTS idx_ee_b ON entity_edges(b_id);

CREATE TABLE IF NOT EXISTS entity_aliases (
    alias_key    TEXT PRIMARY KEY,   -- fold_key|id of a merged-away spelling
    canonical_id INTEGER NOT NULL,
    alias_name   TEXT                -- original display spelling (provenance)
);
"""


VALID_ENTITY_TYPES = frozenset(
    {
        "person",
        "project",
        "technology",
        "file",
        "org",
        "concept",
    }
)


def decay_weight(
    weight: float,
    last_seen: str | None,
    *,
    now_iso: str,
    half_life_days: float = 180.0,
) -> float:
    """Time-decayed edge weight: weight * 0.5 ** (age_days / half_life_days).
    `last_seen` None or unparseable -> undecayed. Ready for Phase 2 ranking;
    Phase 0 callers use raw weight."""
    if not last_seen:
        return weight
    from datetime import date

    try:
        ls = date.fromisoformat(last_seen[:10])
        now = date.fromisoformat(now_iso[:10])
    except ValueError:
        return weight
    age = max(0, (now - ls).days)
    return weight * (0.5 ** (age / half_life_days))


@dataclass(frozen=True)
class EntityMention:
    """Entity linked to a memory."""

    name: str
    type: str
    mention_count: int


class GraphStore:
    """Entity index. Append-only writes via `record_extraction`,
    read-only queries via `top_entities`, `entity_memories`,
    `memory_entities`.

    All writes idempotent on (memory_id) — running extraction twice
    on the same memory refreshes the link set without duplicating.
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
            # Migrate a pre-rename DB BEFORE the IF NOT EXISTS DDL. Two cases:
            #  1. only legacy `entity_memoria` exists -> rename it (+ its
            #     `memoria_id` column) so the DDL doesn't create an empty table
            #     and orphan the data.
            #  2. BOTH `entity_memoria` and `entity_memory` exist -> an interim
            #     build created an empty `entity_memory` alongside the legacy
            #     table, splitting the graph. Fold the legacy rows in (dedup on
            #     UNIQUE(entity_id, memory_id)) then drop the legacy table.
            from memo.util import rename_legacy_columns, rename_legacy_table

            tables = {
                r[0]
                for r in self._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if "entity_memoria" in tables and "entity_memory" in tables:
                self._conn.execute(
                    "INSERT OR IGNORE INTO entity_memory "
                    "(entity_id, memory_id, occurrences, extracted_at) "
                    "SELECT entity_id, memoria_id, occurrences, extracted_at "
                    "FROM entity_memoria"
                )
                self._conn.execute("DROP TABLE entity_memoria")
            else:
                rename_legacy_table(self._conn, "entity_memoria", "entity_memory")
                rename_legacy_columns(self._conn, "entity_memory", {"memoria_id": "memory_id"})
            self._conn.executescript(_SCHEMA_DDL)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        import logging

        _log = logging.getLogger(__name__)

        with self._tx_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.commit()
            except Exception as e:
                _log.debug("graph tx failed: %s", e)
                self._conn.rollback()
                raise

    def record_extraction(
        self,
        *,
        memory_id: str,
        memory_date: str,
        entities: list[dict[str, str]],
        extracted_at: str,
    ) -> int:
        """Idempotently link a memory to its extracted entities.

        Steps under one tx:
        1. Drop any existing entity_memory links for this memory
           (so re-extraction reflects current state).
        2. Upsert each (name, type) into `entities`.
        3. Insert entity_memory links for all current entities.
        4. Update mention_count + first_seen/last_seen for each
           touched entity.

        Returns the number of links written.
        """
        n = 0
        with self._tx() as cx:
            # Get current entity_ids for this memory so we can
            # decrement mention_count if any are removed.
            old_eids = [
                r["entity_id"]
                for r in cx.execute(
                    "SELECT entity_id FROM entity_memory WHERE memory_id = ?",
                    (memory_id,),
                ).fetchall()
            ]
            cx.execute(
                "DELETE FROM entity_memory WHERE memory_id = ?",
                (memory_id,),
            )
            # Decrement mention_count for old entities (will be
            # re-incremented if they're still present).
            for eid in old_eids:
                cx.execute(
                    "UPDATE entities SET mention_count = MAX(0, mention_count - 1) WHERE id = ?",
                    (eid,),
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
                    (name, etype, memory_date, memory_date),
                )
                eid = cx.execute(
                    "SELECT id, first_seen, last_seen FROM entities WHERE name = ? AND type = ?",
                    (name, etype),
                ).fetchone()
                if eid is None:
                    continue
                cx.execute(
                    "INSERT OR IGNORE INTO entity_memory "
                    "(entity_id, memory_id, occurrences, extracted_at) "
                    "VALUES (?, ?, 1, ?)",
                    (eid["id"], memory_id, extracted_at),
                )
                # Bump mention_count + adjust first/last_seen.
                cx.execute(
                    "UPDATE entities SET mention_count = mention_count + 1, "
                    "  first_seen = MIN(first_seen, ?), "
                    "  last_seen  = MAX(last_seen, ?) "
                    "WHERE id = ?",
                    (memory_date, memory_date, eid["id"]),
                )
                n += 1
        return n

    def drop_for_memoria(self, memory_id: str) -> int:
        """Called when a memory is deleted. Removes all entity_memory
        edges for it and decrements mention_count on each touched
        entity. Returns the number of edges removed."""
        with self._tx() as cx:
            old = [
                r["entity_id"]
                for r in cx.execute(
                    "SELECT entity_id FROM entity_memory WHERE memory_id = ?",
                    (memory_id,),
                ).fetchall()
            ]
            cx.execute(
                "DELETE FROM entity_memory WHERE memory_id = ?",
                (memory_id,),
            )
            for eid in old:
                cx.execute(
                    "UPDATE entities SET mention_count = MAX(0, mention_count - 1) WHERE id = ?",
                    (eid,),
                )
        return len(old)

    def canonicalize_existing(self) -> int:
        """Fold fragmented entity rows (case/separator/alias variants, and the
        regex-"concept" vs LLM-typed duplicate) into one canonical row each.

        Groups all entities by fold_key. Canonical = highest mention_count;
        on a tie, the most specific type (non-"concept" wins). Rewrites
        entity_memory links to the canonical id, records merged spellings in
        entity_aliases, recomputes mention_count, deletes the merged rows.
        Idempotent: a graph with no fragments returns 0. Returns rows merged.
        """
        from collections import defaultdict

        from memo.graph_canonical import fold_key

        with self._tx() as cx:
            rows = cx.execute("SELECT id, name, type, mention_count FROM entities").fetchall()
            buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for r in rows:
                buckets[fold_key(r["name"])].append(dict(r))

            merged = 0
            for members in buckets.values():
                if len(members) < 2:
                    continue

                # Canonical: most mentions; tie-break prefers a non-"concept" type.
                def _rank(m: dict[str, Any]) -> tuple[int, int]:
                    return (int(m["mention_count"]), 0 if m["type"] == "concept" else 1)

                canonical = max(members, key=_rank)
                cid = canonical["id"]
                for m in members:
                    if m["id"] == cid:
                        continue
                    # Re-point links (dedup on UNIQUE(entity_id, memory_id)).
                    cx.execute(
                        "UPDATE OR IGNORE entity_memory SET entity_id = ? WHERE entity_id = ?",
                        (cid, m["id"]),
                    )
                    cx.execute("DELETE FROM entity_memory WHERE entity_id = ?", (m["id"],))
                    cx.execute(
                        "INSERT OR REPLACE INTO entity_aliases (alias_key, canonical_id, alias_name) "
                        "VALUES (?, ?, ?)",
                        (fold_key(m["name"]) + "|" + str(m["id"]), cid, m["name"]),
                    )
                    cx.execute("DELETE FROM entities WHERE id = ?", (m["id"],))
                    merged += 1
                # Recompute mention_count = distinct memories now linked.
                cx.execute(
                    "UPDATE entities SET mention_count = "
                    "(SELECT COUNT(*) FROM entity_memory WHERE entity_id = ?) WHERE id = ?",
                    (cid, cid),
                )
            return merged

    def list_entities(self, *, min_mentions: int = 1) -> list[dict[str, Any]]:
        """All entity rows (id, name, type, mention_count), most-mentioned
        first. Enumeration surface for external blocking (dream entity-canon)."""
        rows = self._conn.execute(
            "SELECT id, name, type, mention_count FROM entities "
            "WHERE mention_count >= ? ORDER BY mention_count DESC, name ASC",
            (min_mentions,),
        ).fetchall()
        return [dict(r) for r in rows]

    def merge_entity_pair(self, canonical_id: int, dup_id: int, dup_name: str) -> None:
        """Fold entity `dup_id` into `canonical_id` — the same statements as
        `canonicalize_existing`, for ONE externally-decided pair: re-point
        links, record the merged spelling in entity_aliases, delete the dup
        row, recompute the canonical mention_count. Idempotent per pair."""
        from memo.graph_canonical import fold_key

        with self._tx() as cx:
            cx.execute(
                "UPDATE OR IGNORE entity_memory SET entity_id = ? WHERE entity_id = ?",
                (canonical_id, dup_id),
            )
            cx.execute("DELETE FROM entity_memory WHERE entity_id = ?", (dup_id,))
            cx.execute(
                "INSERT OR REPLACE INTO entity_aliases (alias_key, canonical_id, alias_name) "
                "VALUES (?, ?, ?)",
                (fold_key(dup_name) + "|" + str(dup_id), canonical_id, dup_name),
            )
            cx.execute("DELETE FROM entities WHERE id = ?", (dup_id,))
            cx.execute(
                "UPDATE entities SET mention_count = "
                "(SELECT COUNT(*) FROM entity_memory WHERE entity_id = ?) WHERE id = ?",
                (canonical_id, canonical_id),
            )

    def rebuild_edges(self) -> int:
        """Materialize entity_edges from entity_memory co-occurrence. Idempotent.
        weight = #shared memories; last_seen = max endpoint last_seen."""
        from collections import defaultdict

        with self._tx() as cx:
            cx.execute("DELETE FROM entity_edges")
            last_seen = {
                r["id"]: r["last_seen"]
                for r in cx.execute("SELECT id, last_seen FROM entities")
            }
            mem_ents: dict[str, list[int]] = defaultdict(list)
            for r in cx.execute("SELECT memory_id, entity_id FROM entity_memory"):
                mem_ents[r["memory_id"]].append(r["entity_id"])
            edges: dict[tuple[int, int], int] = defaultdict(int)
            for ents in mem_ents.values():
                u = sorted(set(ents))
                for i in range(len(u)):
                    for j in range(i + 1, len(u)):
                        edges[(u[i], u[j])] += 1
            for (a, b), w in edges.items():
                la, lb = last_seen.get(a), last_seen.get(b)
                ls = max(x for x in (la, lb) if x) if (la or lb) else None
                cx.execute(
                    "INSERT INTO entity_edges (a_id, b_id, weight, last_seen) VALUES (?, ?, ?, ?)",
                    (a, b, w, ls),
                )
            return len(edges)

    def all_weighted_edges(self) -> list[tuple[str, str, float]]:
        rows = self._conn.execute(
            "SELECT ea.name AS a, eb.name AS b, e.weight AS w "
            "FROM entity_edges e "
            "JOIN entities ea ON ea.id = e.a_id "
            "JOIN entities eb ON eb.id = e.b_id"
        ).fetchall()
        return [(r["a"], r["b"], float(r["w"])) for r in rows]

    def weighted_neighbors(self, name: str) -> dict[str, float]:
        name = name.strip().lower()
        rows = self._conn.execute("SELECT id FROM entities WHERE name = ?", (name,)).fetchall()
        if not rows:
            return {}
        eids = [int(r["id"]) for r in rows]
        out: dict[str, float] = {}
        for eid in eids:
            for r in self._conn.execute(
                "SELECT CASE WHEN a_id = ? THEN b_id ELSE a_id END AS other, weight "
                "FROM entity_edges WHERE a_id = ? OR b_id = ?",
                (eid, eid, eid),
            ):
                nm = self._conn.execute(
                    "SELECT name FROM entities WHERE id = ?", (r["other"],)
                ).fetchone()
                if nm is not None:
                    out[nm["name"]] = max(float(r["weight"]), out.get(nm["name"], 0.0))
        return out

    def entity_names(self) -> set[str]:
        """All entity names (already lower-cased) — the graph's vocabulary, used
        to match query tokens against known entities so the recall boost can fire
        on natural lowercase prompts (one indexed scan, no MLX)."""
        return {r["name"] for r in self._conn.execute("SELECT name FROM entities")}

    def top_entities(
        self,
        *,
        limit: int = 50,
        type_: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT name, type, mention_count, first_seen, last_seen FROM entities"
        params: list[Any] = []
        if type_:
            sql += " WHERE type = ?"
            params.append(type_)
        sql += " ORDER BY mention_count DESC, last_seen DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def entity_memories(self, name: str, type_: str | None = None) -> list[str]:
        """Memory IDs that mention `name` (and optionally a specific type)."""
        name = name.strip().lower()
        params: tuple[str, ...]
        if type_:
            type_ = type_.strip().lower()
            sql = (
                "SELECT em.memory_id FROM entity_memory em "
                "JOIN entities e ON e.id = em.entity_id "
                "WHERE e.name = ? AND e.type = ?"
            )
            params = (name, type_)
        else:
            sql = (
                "SELECT em.memory_id FROM entity_memory em "
                "JOIN entities e ON e.id = em.entity_id "
                "WHERE e.name = ?"
            )
            params = (name,)
        return [r["memory_id"] for r in self._conn.execute(sql, params).fetchall()]

    def memory_entities(self, memory_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT e.name, e.type, e.mention_count "
            "FROM entity_memory em JOIN entities e ON e.id = em.entity_id "
            "WHERE em.memory_id = ? "
            "ORDER BY e.mention_count DESC",
            (memory_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_entity_mentions(self, memory_id: str) -> list[EntityMention]:
        """Return entity mentions for compatibility with analytics callers."""
        return [
            EntityMention(
                name=row["name"],
                type=row["type"],
                mention_count=int(row["mention_count"]),
            )
            for row in self.memory_entities(memory_id)
        ]

    def count_entities(self) -> int:
        """Return the number of unique entities in the graph."""
        return int(self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0])

    def total_indexed_memories(self) -> int:
        """Distinct memories carrying at least one entity link — the ``N`` for
        entity IDF (rarity) weighting of the graph recall signals."""
        row = self._conn.execute("SELECT COUNT(DISTINCT memory_id) FROM entity_memory").fetchone()
        return int(row[0]) if row and row[0] else 0

    def entity_doc_freqs(self, names: Sequence[str]) -> dict[str, float]:
        """Document frequency (distinct memories mentioning it) per entity name,
        lower-cased, in ONE batched query. Feeds the IDF weighting that lets the
        graph signals discriminate rare entities from ubiquitous ones. Names not
        in the graph are omitted (an unknown entity has no measurable rarity)."""
        wanted = [n.strip().lower() for n in names if n and n.strip()]
        if not wanted:
            return {}
        placeholders = ",".join("?" * len(wanted))
        sql = (
            "SELECT e.name AS name, COUNT(DISTINCT em.memory_id) AS df "  # noqa: S608
            "FROM entities e JOIN entity_memory em ON em.entity_id = e.id "
            f"WHERE e.name IN ({placeholders}) GROUP BY e.name"
        )
        return {r["name"]: float(r["df"]) for r in self._conn.execute(sql, wanted).fetchall()}

    def stats(self) -> dict[str, int]:
        n_entities = self.count_entities()
        n_links = self._conn.execute("SELECT COUNT(*) FROM entity_memory").fetchone()[0]
        return {"entities": n_entities, "links": n_links}

    def edge_stats(self) -> dict[str, float]:
        """Materialized-edge health: count + weight distribution. Surfaces graph
        substrate state (otherwise only visible via raw SQL) so canonicalization
        / weighting regressions are catchable from `memo graph stats`."""
        row = self._conn.execute(
            "SELECT COUNT(*), MIN(weight), MAX(weight), AVG(weight), "
            "SUM(CASE WHEN weight > 1 THEN 1 ELSE 0 END) FROM entity_edges"
        ).fetchone()
        if row is None or not row[0]:
            return {"edges": 0, "weight_min": 0.0, "weight_max": 0.0, "weight_mean": 0.0, "edges_gt1": 0}
        return {
            "edges": int(row[0]),
            "weight_min": float(row[1] or 0),
            "weight_max": float(row[2] or 0),
            "weight_mean": round(float(row[3] or 0), 3),
            "edges_gt1": int(row[4] or 0),
        }

    def record_co_recall(self, ids: list[str]) -> int:
        """Increment co-recall count for every pair in `ids`.

        Called after a search returns 2+ results. Pairs are stored with
        id_a < id_b so the primary key is order-independent. Returns the
        number of pairs upserted.
        """
        if len(ids) < 2:
            return 0
        sorted_ids = sorted(ids)
        pairs = [
            (sorted_ids[i], sorted_ids[j])
            for i in range(len(sorted_ids))
            for j in range(i + 1, len(sorted_ids))
        ]
        with self._tx() as cx:
            for a, b in pairs:
                cx.execute(
                    "INSERT INTO co_recall (id_a, id_b, count) VALUES (?, ?, 1) "
                    "ON CONFLICT(id_a, id_b) DO UPDATE SET count = count + 1",
                    (a, b),
                )
        return len(pairs)

    def co_recall_counts(self, anchor_id: str, candidate_ids: list[str]) -> dict[str, int]:
        """Co-recall counts between `anchor_id` and each candidate id.

        Returns a {candidate_id: count} map for candidates that have ever been
        co-recalled with the anchor (absent candidates have an implicit 0). One
        query — used by the search-time co-recall ranking boost.
        """
        ids = [c for c in candidate_ids if c != anchor_id]
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT id_a, id_b, count FROM co_recall "  # noqa: S608 (placeholders are bound params)
            f"WHERE (id_a = ? AND id_b IN ({placeholders})) "
            f"OR (id_b = ? AND id_a IN ({placeholders}))",
            (anchor_id, *ids, anchor_id, *ids),
        ).fetchall()
        counts: dict[str, int] = {}
        for r in rows:
            other = r["id_b"] if r["id_a"] == anchor_id else r["id_a"]
            counts[other] = int(r["count"])
        return counts

    def top_co_recalled(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most frequently co-recalled pairs."""
        rows = self._conn.execute(
            "SELECT id_a, id_b, count FROM co_recall ORDER BY count DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def distance_to_nearest_fact(self, memory_id: str) -> int:
        """Compute shortest path distance from memory_id to any FACT-type memory.

        Currently a stub returning 999 (unreachable). Will be implemented
        when memory-to-memory distance tracking is fully integrated.

        Args:
            memory_id: The source memory ID.

        Returns:
            Shortest path distance to nearest FACT memory, or 999 if unreachable.
        """
        return 999

    def close(self) -> None:
        with suppress(BaseException):
            self._conn.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        with suppress(BaseException):
            self.close()


__all__ = ["VALID_ENTITY_TYPES", "EntityMention", "GraphStore"]
