from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any

from memo.tiers import EVICTION_PROTECTED_TYPES

from ._base import _StoreBase


class _SignalQueriesMixin(_StoreBase):
    """Access-tracking and memory-health signal methods.

    Extracted from `_QueriesMixin` to keep each file under 800 lines.
    `VecStore` inherits these via `_QueriesMixin(_SignalQueriesMixin, ...)`.
    """

    # -- access tracking (cache tier hit counting) -------------------------
    #
    # The history log only records save/update/delete, never reads — so it
    # can't drive LRU/LFU. The `access` table fills that gap: `touch()` bumps
    # a per-memory hit count + last-access timestamp on every search/ask
    # hit. Cheap, write-light, and decoupled from the hot `meta`/`vec` path.

    def touch(self, ids: list[str], *, ts: str | None = None) -> None:
        """Record a read/hit for each id: ++access_count, set last_accessed.

        Batch upsert in one tx. No-op on empty input. Safe to call for ids
        with no `meta` row (the access row is harmless and cleaned on delete).
        Callers should invoke this fire-and-forget off the hot path — the
        recall hook's 5s budget must not wait on it.
        """
        if not ids:
            return
        now = ts or datetime.now(UTC).isoformat()
        with self._tx() as cx:
            cx.executemany(
                "INSERT INTO access (id, access_count, last_accessed) VALUES (?, 1, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "access_count = access_count + 1, last_accessed = excluded.last_accessed",
                [(i, now) for i in ids],
            )

    def get_access(self, id_: str) -> dict[str, Any]:
        """Return {access_count, last_accessed} for a memory.

        Defaults to count 0 / last_accessed None when never touched.
        """
        row = self._conn.execute(
            "SELECT access_count, last_accessed FROM access WHERE id = ?",
            (id_,),
        ).fetchone()
        if not row:
            return {"access_count": 0, "last_accessed": None}
        return {"access_count": int(row["access_count"]), "last_accessed": row["last_accessed"]}

    # -- cross-machine signal export/import (F3) ---------------------------
    #
    # The `.md` memories sync via git; the signal tables (access, health,
    # source_feedback) are local-only PRIMARY data not present in markdown.
    # `dump_signal` snapshots them for `memo sync export-signal`; `merge_signal`
    # folds a peer's snapshot back in, keyed on the stable memory id. Merge is
    # idempotent on re-pull: access = max, health = newer updated_at wins,
    # feedback = union by id. (source_feedback_vec embeddings are NOT synced —
    # they are re-derivable from query_text by a future re-embed pass.)

    def dump_signal(self) -> dict[str, list[dict[str, Any]]]:
        """Return every signal row, grouped by table, as plain dicts."""
        access = [
            {"id": r["id"], "access_count": int(r["access_count"]), "last_accessed": r["last_accessed"]}
            for r in self._conn.execute(
                "SELECT id, access_count, last_accessed FROM access"
            ).fetchall()
        ]
        health = [
            {
                "id": r["id"],
                "confidence": float(r["confidence"]),
                "roi_score": float(r["roi_score"]),
                "updated_at": r["updated_at"],
                "support_count": int(r["support_count"]),
            }
            for r in self._conn.execute(
                "SELECT id, confidence, roi_score, updated_at, support_count FROM memory_health"
            ).fetchall()
        ]
        feedback = [
            dict(r)
            for r in self._conn.execute(
                "SELECT id, source_id, query_text, rating, created_at, extra_json "
                "FROM source_feedback"
            ).fetchall()
        ]
        return {"access": access, "memory_health": health, "source_feedback": feedback}

    def merge_signal(self, payload: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
        """Merge a peer's signal snapshot into the local tables.

        Returns the count of rows applied per table. Idempotent: re-merging the
        same payload (or the local store's own export) never inflates counts.
        """
        access = payload.get("access") or []
        health = payload.get("memory_health") or []
        feedback = payload.get("source_feedback") or []
        with self._tx() as cx:
            if access:
                cx.executemany(
                    "INSERT INTO access (id, access_count, last_accessed) VALUES (?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "access_count = max(access_count, excluded.access_count), "
                    "last_accessed = nullif("
                    "  max(coalesce(last_accessed, ''), coalesce(excluded.last_accessed, '')), '')",
                    [(r["id"], int(r["access_count"]), r.get("last_accessed")) for r in access],
                )
            if health:
                cx.executemany(
                    "INSERT INTO memory_health (id, confidence, roi_score, updated_at, support_count) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "confidence = excluded.confidence, "
                    "roi_score = excluded.roi_score, "
                    "updated_at = excluded.updated_at, "
                    "support_count = max(memory_health.support_count, excluded.support_count) "
                    "WHERE coalesce(excluded.updated_at, '') > coalesce(memory_health.updated_at, '')",
                    [
                        (
                            r["id"],
                            float(r["confidence"]),
                            float(r["roi_score"]),
                            r.get("updated_at"),
                            int(r.get("support_count", 0)),
                        )
                        for r in health
                    ],
                )
            feedback_applied = 0
            if feedback:
                for r in feedback:
                    # Cancel-and-replace: any existing feedback for this
                    # (source_id, query_text) is replaced by the newest
                    # created_at, regardless of rating. Prevents stale
                    # opposite-rating rows from different devices coexisting.
                    existing = cx.execute(
                        "SELECT id, rating, created_at FROM source_feedback "
                        "WHERE source_id = ? AND query_text = ?",
                        (r["source_id"], r["query_text"]),
                    ).fetchone()
                    if existing is not None:
                        existing_created = existing["created_at"]
                        incoming_created = r.get("created_at") or ""
                        if incoming_created <= existing_created:
                            continue
                        cx.execute("DELETE FROM source_feedback_vec WHERE feedback_id = ?", (existing["id"],))
                        cx.execute("DELETE FROM source_feedback WHERE id = ?", (existing["id"],))
                    cx.execute(
                        "INSERT INTO source_feedback "
                        "(id, source_id, query_text, rating, created_at, extra_json) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            r["id"],
                            r["source_id"],
                            r["query_text"],
                            int(r["rating"]),
                            r["created_at"],
                            r.get("extra_json"),
                        ),
                    )
                    feedback_applied += 1
        return {"access": len(access), "memory_health": len(health), "source_feedback": feedback_applied}

    # -- memory health (confidence + roi_score) ----------------------------

    def bump_support_batch(self, ids: list[str], *, lift: float = 0.0, cap: float = 1.0) -> None:
        """Increment the corroboration counter for each id (one bump per list
        occurrence). `lift` > 0 additionally RESTORES confidence toward `cap`
        — bounded evidence strength: with the 1.0 cap a re-assertion can only
        undo prior penalties (contradiction/OCR quality), never boost a
        memory above neutral. Upserts missing rows. Best-effort caller
        contract; never on the recall-hook hot path."""
        if not ids:
            return
        with self._tx() as cx:
            cx.executemany(
                "INSERT INTO memory_health(id, confidence, roi_score, updated_at, support_count) "
                "VALUES(?, 1.0, 1.0, datetime('now'), 1) "
                "ON CONFLICT(id) DO UPDATE SET "
                "support_count = support_count + 1, "
                "confidence = min(?, confidence + ?), "
                "updated_at = datetime('now')",
                [(i, cap, lift) for i in ids],
            )

    def get_support_batch(self, ids: list[str]) -> dict[str, int]:
        """Return {id: support_count} for the given ids; missing ids are
        absent (callers treat missing as 0). Kept SEPARATE from
        get_health_batch so the search-scoring hot path keeps its exact
        current query."""
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT id, support_count FROM memory_health WHERE id IN ({placeholders})",  # noqa: S608
            ids,
        ).fetchall()
        return {r["id"]: int(r["support_count"]) for r in rows}

    def get_health_batch(self, ids: list[str]) -> dict[str, dict[str, float]]:
        """Return {id: {confidence, roi_score}} for the given IDs.

        IDs not in the table are absent from the result (callers treat missing
        as defaults: confidence=1.0, roi_score=1.0).
        """
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT id, confidence, roi_score FROM memory_health WHERE id IN ({placeholders})",  # noqa: S608
            ids,
        ).fetchall()
        return {
            r["id"]: {"confidence": float(r["confidence"]), "roi_score": float(r["roi_score"])}
            for r in rows
        }

    def boost_roi_batch(
        self,
        ids: list[str],
        delta: float = 0.05,
        cap: float = 1.5,
    ) -> None:
        """Increment roi_score for each id, capped at `cap`. Upserts new rows."""
        if not ids:
            return
        with self._tx() as cx:
            cx.executemany(
                "INSERT INTO memory_health(id, confidence, roi_score, updated_at) "
                "VALUES(?, 1.0, min(?, 1.0 + ?), datetime('now')) "
                "ON CONFLICT(id) DO UPDATE SET "
                "roi_score = min(?, roi_score + ?), "
                "updated_at = datetime('now')",
                [(i, cap, delta, cap, delta) for i in ids],
            )

    def penalize_confidence_batch(
        self,
        ids: list[str],
        delta: float = 0.15,
        floor: float = 0.1,
    ) -> None:
        """Decrement confidence for each id (e.g. open contradiction). Floor at `floor`."""
        if not ids:
            return
        with self._tx() as cx:
            cx.executemany(
                "INSERT INTO memory_health(id, confidence, roi_score, updated_at) "
                "VALUES(?, max(?, 1.0 - ?), 1.0, datetime('now')) "
                "ON CONFLICT(id) DO UPDATE SET "
                "confidence = max(?, confidence - ?), "
                "updated_at = datetime('now')",
                [(i, floor, delta, floor, delta) for i in ids],
            )

    def set_confidence_batch(
        self,
        pairs: list[tuple[str, float]],
        floor: float = 0.1,
    ) -> None:
        """Set an absolute confidence for each (id, confidence) pair, floored at
        ``floor``. Unlike :meth:`penalize_confidence_batch` (relative decrement),
        this writes the value directly — used to stamp OCR'd-image records with
        their measured quality so low-confidence screenshots rank below clean
        notes (search score x confidence). roi_score left neutral (1.0)."""
        if not pairs:
            return
        with self._tx() as cx:
            cx.executemany(
                "INSERT INTO memory_health(id, confidence, roi_score, updated_at) "
                "VALUES(?, max(?, ?), 1.0, datetime('now')) "
                "ON CONFLICT(id) DO UPDATE SET "
                "confidence = excluded.confidence, "
                "updated_at = datetime('now')",
                [(i, floor, c) for i, c in pairs],
            )

    def all_ids(self) -> list[str]:
        """Return every memory id in ``meta``. Used by the outcome loop to map
        the 8-char id prefixes stored in recall.log / grounding.log back to full
        ids (roi_score is keyed by the full id)."""
        return [r["id"] for r in self._conn.execute("SELECT id FROM meta").fetchall()]

    def set_roi_batch(
        self,
        pairs: list[tuple[str, float]],
        *,
        floor: float = 0.5,
        cap: float = 1.5,
    ) -> int:
        """Set an ABSOLUTE roi_score for each (id, roi) pair, clamped to
        ``[floor, cap]``. Authoritative write used by the outcome loop
        (`memo.outcome`) to overwrite the access-driven roi drift with a value
        derived from real grounding outcomes (was the surfaced memory actually
        USED in the answer?). Confidence is left at its existing value (new rows
        default 1.0). Returns the number of (id, roi) pairs written."""
        if not pairs:
            return 0
        clamped = [(i, max(floor, min(cap, float(r)))) for i, r in pairs]
        with self._tx() as cx:
            cx.executemany(
                "INSERT INTO memory_health(id, confidence, roi_score, updated_at) "
                "VALUES(?, 1.0, ?, datetime('now')) "
                "ON CONFLICT(id) DO UPDATE SET "
                "roi_score = excluded.roi_score, "
                "updated_at = datetime('now')",
                [(i, r) for i, r in clamped],
            )
        return len(clamped)

    def decay_roi(
        self,
        factor: float = 0.98,
        older_than_days: int = 30,
    ) -> int:
        """Multiply roi_score by `factor` for memories not accessed in `older_than_days`.

        Returns the count of rows updated. Used by Dream mode nightly pipeline.
        """
        with self._tx() as cx:
            cur = cx.execute(
                "UPDATE memory_health SET roi_score = max(0.1, roi_score * ?), "
                "updated_at = datetime('now') "
                "WHERE ("
                "  SELECT COALESCE(MAX(a.last_accessed), '1970-01-01') "
                "  FROM access a WHERE a.id = memory_health.id"
                ") < datetime('now', ? || ' days')",
                (factor, f"-{older_than_days}"),
            )
            return cur.rowcount

    def prune_floor_candidates(
        self,
        roi_floor: float = 0.15,
        min_age_days: int = 90,
        exclude_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return memories below roi_floor, never accessed, and older than min_age_days.

        Always excludes 'synthesis', 'reference', and EVICTION_PROTECTED_TYPES
        (bug/failure_pattern/procedure — see memo.tiers). Uses INNER JOIN
        because every meta row now has guaranteed access + memory_health rows
        (seeded on upsert since v2, backfilled for legacy rows). Returns list
        of {id, roi_score, days_old}.
        """
        excluded = (exclude_types or set()) | {"synthesis", "reference"} | EVICTION_PROTECTED_TYPES
        placeholders = ",".join("?" for _ in excluded)
        try:
            rows = self._conn.execute(
                f"""
                SELECT m.id,
                       h.roi_score                                      AS roi_score,
                       CAST(julianday('now') - julianday(m.updated) AS INTEGER) AS days_old
                  FROM meta m
                  LEFT JOIN memory_health h ON h.id = m.id
                  LEFT JOIN access a       ON a.id = m.id
                 WHERE COALESCE(h.roi_score, 1.0) < ?
                   AND COALESCE(a.access_count, 0) = 0
                   AND m.updated < datetime('now', '-' || ? || ' days')
                   AND m.type NOT IN ({placeholders})
                   AND (m.deleted_at IS NULL OR m.deleted_at = '')
                """,  # noqa: S608
                (roi_floor, min_age_days, *excluded),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such column" not in str(exc):
                raise
            # Fallback for old DBs without deleted_at column
            rows = self._conn.execute(
                f"""
                SELECT m.id,
                       h.roi_score                                      AS roi_score,
                       CAST(julianday('now') - julianday(m.updated) AS INTEGER) AS days_old
                  FROM meta m
                  LEFT JOIN memory_health h ON h.id = m.id
                  LEFT JOIN access a       ON a.id = m.id
                 WHERE COALESCE(h.roi_score, 1.0) < ?
                   AND COALESCE(a.access_count, 0) = 0
                   AND m.updated < datetime('now', '-' || ? || ' days')
                   AND m.type NOT IN ({placeholders})
                """,  # noqa: S608
                (roi_floor, min_age_days, *excluded),
            ).fetchall()
        return [{"id": r["id"], "roi_score": r["roi_score"], "days_old": r["days_old"]} for r in rows]

    def eviction_candidates(
        self,
        policy: str,
        limit: int,
        *,
        exclude_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return up to `limit` memories coldest-first under the given policy.

        Uses LEFT JOIN for safety (legacy rows may lack access row before
        migration v2 runs).  Since the upsert seed and backfill both set
        ``last_accessed = updated``, every access row has a non-NULL value
        once migration completes, so ORDER BY uses the indexed column
        directly instead of a cross-table COALESCE.

          - lru: order by last_accessed ASC (coldest first).
          - lfu: order by access_count ASC, then last_accessed ASC.
          - ttl: same ordering as lru; the age cutoff is applied by the caller.

        Returns dicts: {id, type, access_count, last_accessed, updated}.
        """
        if policy not in {"lru", "lfu", "ttl"}:
            raise ValueError(f"unknown eviction policy: {policy!r}")
        order = ("COALESCE(a.access_count, 0) ASC, COALESCE(a.last_accessed, m.updated) ASC"
                 if policy == "lfu"
                 else "COALESCE(a.last_accessed, m.updated) ASC")
        # Always filter soft-deleted rows; WHERE clause always has at least deleted_at guard.
        sql = (
            "SELECT m.id AS id, m.type AS type, "
            "COALESCE(a.access_count, 0) AS access_count, "
            "COALESCE(a.last_accessed, m.updated) AS last_accessed, m.updated AS updated "
            "FROM meta m LEFT JOIN access a ON a.id = m.id "
            "WHERE (m.deleted_at IS NULL OR m.deleted_at = '') "
        )
        params: list[Any] = []
        if exclude_types:
            placeholders = ",".join("?" for _ in exclude_types)
            sql += f"AND m.type NOT IN ({placeholders}) "
            params.extend(sorted(exclude_types))
        sql += f"ORDER BY {order} LIMIT ?"
        params.append(limit)
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such column" not in str(exc):
                raise
            # Fallback for old DBs without deleted_at column
            sql = (
                "SELECT m.id AS id, m.type AS type, "
                "COALESCE(a.access_count, 0) AS access_count, "
                "COALESCE(a.last_accessed, m.updated) AS last_accessed, m.updated AS updated "
                "FROM meta m LEFT JOIN access a ON a.id = m.id "
            )
            params = []
            if exclude_types:
                placeholders = ",".join("?" for _ in exclude_types)
                sql += f"WHERE m.type NOT IN ({placeholders}) "
                params.extend(sorted(exclude_types))
            sql += f"ORDER BY {order} LIMIT ?"
            params.append(limit)
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
