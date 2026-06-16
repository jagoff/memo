from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

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
    # a per-memoria hit count + last-access timestamp on every search/ask
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
        """Return {access_count, last_accessed} for a memoria.

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
    # The `.md` memorias sync via git; the signal tables (access, health,
    # source_feedback) are local-only PRIMARY data not present in markdown.
    # `dump_signal` snapshots them for `memo sync export-signal`; `merge_signal`
    # folds a peer's snapshot back in, keyed on the stable memoria id. Merge is
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
            }
            for r in self._conn.execute(
                "SELECT id, confidence, roi_score, updated_at FROM memory_health"
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
                    "INSERT INTO memory_health (id, confidence, roi_score, updated_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "confidence = excluded.confidence, "
                    "roi_score = excluded.roi_score, "
                    "updated_at = excluded.updated_at "
                    "WHERE coalesce(excluded.updated_at, '') > coalesce(memory_health.updated_at, '')",
                    [
                        (r["id"], float(r["confidence"]), float(r["roi_score"]), r.get("updated_at"))
                        for r in health
                    ],
                )
            if feedback:
                # union by id (and by the secondary UNIQUE(source_id,query_text,rating))
                cx.executemany(
                    "INSERT OR IGNORE INTO source_feedback "
                    "(id, source_id, query_text, rating, created_at, extra_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            r["id"],
                            r["source_id"],
                            r["query_text"],
                            int(r["rating"]),
                            r["created_at"],
                            r.get("extra_json"),
                        )
                        for r in feedback
                    ],
                )
        return {"access": len(access), "memory_health": len(health), "source_feedback": len(feedback)}

    # -- memory health (confidence + roi_score) ----------------------------

    def get_health_batch(self, ids: list[str]) -> dict[str, dict[str, float]]:
        """Return {id: {confidence, roi_score}} for the given IDs.

        IDs not in the table are absent from the result (callers treat missing
        as defaults: confidence=1.0, roi_score=1.0).
        """
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT id, confidence, roi_score FROM memory_health WHERE id IN ({placeholders})",
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
                "confidence = max(excluded.confidence, confidence - ?), "
                "updated_at = datetime('now')",
                [(i, floor, delta, delta) for i in ids],
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
                "confidence = min(confidence, ?), "
                "updated_at = datetime('now')",
                [(i, floor, c, c) for i, c in pairs],
            )

    def decay_roi(
        self,
        factor: float = 0.98,
        older_than_days: int = 30,
    ) -> int:
        """Multiply roi_score by `factor` for memorias not accessed in `older_than_days`.

        Returns the count of rows updated. Used by Dream mode nightly pipeline.
        """
        with self._tx() as cx:
            cur = cx.execute(
                "UPDATE memory_health SET roi_score = max(0.1, roi_score * ?), "
                "updated_at = datetime('now') "
                "WHERE updated_at < datetime('now', ? || ' days') "
                "OR updated_at IS NULL",
                (factor, f"-{older_than_days}"),
            )
            return cur.rowcount

    def eviction_candidates(
        self,
        policy: str,
        limit: int,
        *,
        exclude_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return up to `limit` memorias coldest-first under the given policy.

        Joins `meta` LEFT against `access` so never-accessed rows participate,
        falling back to `meta.updated` as their effective last-access time
        (a row written long ago and never read is genuinely cold).

          - lru: order by effective last-access ASC (coldest first).
          - lfu: order by access_count ASC, then effective last-access ASC.
          - ttl: same ordering as lru; the age cutoff is applied by the caller.

        Returns dicts: {id, type, access_count, last_accessed, updated}.
        """
        if policy not in {"lru", "lfu", "ttl"}:
            raise ValueError(f"unknown eviction policy: {policy!r}")
        eff = "COALESCE(a.last_accessed, m.updated)"
        # lru and ttl share coldest-first-by-recency ordering
        order = f"COALESCE(a.access_count, 0) ASC, {eff} ASC" if policy == "lfu" else f"{eff} ASC"
        sql = (
            "SELECT m.id AS id, m.type AS type, "
            "COALESCE(a.access_count, 0) AS access_count, "
            "a.last_accessed AS last_accessed, m.updated AS updated "
            "FROM meta m LEFT JOIN access a ON a.id = m.id "
        )
        params: list[Any] = []
        if exclude_types:
            placeholders = ",".join("?" for _ in exclude_types)
            sql += f"WHERE m.type NOT IN ({placeholders}) "
            params.extend(sorted(exclude_types))
        sql += f"ORDER BY {order} LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
