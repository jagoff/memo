"""Contradiction compatibility surface backed by canonical memory relations.

Persistent contradiction & staleness radar for the memory corpus.

This module sits on top of `TemporalAnalyzer` (which already classifies
pairs as contradiction / evolution / consistent / unrelated) and adds
the missing pieces needed for a triage workflow:

- **Legacy sidecar reader** (`contradictions.db`) retained only for migration.
  New scans and triage writes go to the canonical `memory_relations` table.
- **Corpus-wide scan** that walks every memory, finds near-neighbors
  via vec search, and classifies the pairs that look promising. Pair
  IDs are canonical (lower id first) so the same pair is never stored
  twice.
- **Resolution API** that the CLI / MCP triage walker calls to mark a
  pair as fused / kept-newer / kept-older / evolved / dismissed.
- **Stale links**: tags pairs whose older side is past a staleness
  cutoff so the triage UI can foreground them.

The store is kept in its own sqlite file (mirrors `history.db` /
`graph.db`) to avoid WAL contention with the hot vec reader. New anomaly
events are appended to Memo's own operational journal.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from memo.flags import flag_bool
from memo.temporal import Contradiction, TemporalAnalyzer

_log = logging.getLogger(__name__)


def _as_aware(ts: str) -> datetime:
    """Parse an ISO timestamp into an aware UTC datetime for ordering.

    `.updated` is written with a LOCAL UTC offset (`_now_iso` uses
    `.astimezone()`), so a raw string compare inverts order across a DST
    boundary in a positive-offset zone (e.g. Madrid CEST→CET makes the
    actually-earlier instant sort as `"+02:00" > "+01:00"`), reorienting a
    `supersedes` edge backwards. Compare instants, not strings. Unparseable or
    empty timestamps sort oldest so they never win a `kept_newer` reorientation.
    """
    try:
        dt = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return datetime.min.replace(tzinfo=UTC)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


VALID_STATUSES = {
    "open",  # pending triage
    "fused",  # merged into a new memory
    "kept_newer",  # newer side won, older deleted/archived
    "kept_older",  # older side won (rare; explicit user choice)
    "evolved",  # both kept, marked as legitimate evolution
    "competing",  # both kept: neither side dominates (trust within margin / N-way)
    "dismissed",  # false positive
}


_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS pairs (
    pair_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id_a   TEXT NOT NULL,
    memory_id_b   TEXT NOT NULL,
    relationship   TEXT NOT NULL,
    confidence     REAL NOT NULL,
    rationale      TEXT,
    status         TEXT NOT NULL DEFAULT 'open',
    detected_at    TEXT NOT NULL,
    resolved_at    TEXT,
    resolution_note TEXT,
    UNIQUE(memory_id_a, memory_id_b)
);
CREATE INDEX IF NOT EXISTS idx_pairs_status ON pairs(status);
CREATE INDEX IF NOT EXISTS idx_pairs_a ON pairs(memory_id_a);
CREATE INDEX IF NOT EXISTS idx_pairs_b ON pairs(memory_id_b);
"""


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    """Order a pair so the same two ids always hash to the same row."""
    return (a, b) if a <= b else (b, a)


# -- C4: mutability classes ---------------------------------------------------
#
# Regex heuristic over a memory body. VOLATILE facts (ports, versions,
# deploy/status words) churn legitimately: two volatile bodies that "contradict"
# are an update, not a conflict. EPHEMERAL facts are now-anchored and expected
# to expire. Everything else is STABLE (default).
#
# Spanish-corpus guard (review fix): the volatile word list must NOT contain
# everyday Spanish words. 'todo' (everything/all) and bare 'estado' (state, an
# ordinary noun) were dropped — a false volatile classification silently
# downgrades a REAL contradiction to a confidence demotion, so this list stays
# conservative. Only the unambiguous status idiom 'estado actual' is matched.

_MUTABILITY_EPHEMERAL = re.compile(
    r"(?ix)\b("
    r"hoy|today|esta\s+semana|this\s+week|ahora\s+mismo|right\s+now"
    r"|en\s+este\s+momento|esta\s+sesi[oó]n|this\s+session"
    r")\b"
)

_MUTABILITY_VOLATILE = re.compile(
    r"(?ix)"
    r"\bv?\d+\.\d+(?:\.\d+)*\b"  # version strings: 2.9.5 / v1.2
    r"|:\d{2,5}\b"  # :8765
    r"|\b(?:puerto|port)\s+\d{2,5}\b"  # puerto 8765 / port 8080
    r"|\b(?:status|running|corriendo|deployed|desplegado|enabled|disabled"
    r"|activado|desactivado|pendiente|pending|blocked|bloqueado|wip)\b"
    r"|\bestado\s+actual\b"  # bare 'estado'/'todo' deliberately excluded — see guard note
    r"|\b(?:actualmente|currently|por\s+ahora|for\s+now|latest|[uú]ltima\s+versi[oó]n)\b"
)


def classify_mutability(text: str) -> str:
    """Classify a memory body as ``stable`` | ``volatile`` | ``ephemeral``.

    Pure regex, dependency-free, no MLX — cheap enough for the scan loop.
    Ephemeral wins over volatile (more specific); stable is the default.
    """
    t = text or ""
    if _MUTABILITY_EPHEMERAL.search(t):
        return "ephemeral"
    if _MUTABILITY_VOLATILE.search(t):
        return "volatile"
    return "stable"


def downgrade_volatile_contradiction(
    contr: Contradiction, text_a: str, text_b: str
) -> Contradiction:
    """C4: volatile-vs-volatile is an UPDATE, not a conflict.

    When both sides of an LLM-classified *contradiction* are volatile-class,
    reclassify as ``evolution`` so `memo maintain` demotes the older side's
    confidence instead of archiving it. Non-contradiction verdicts and mixed
    classes pass through untouched.
    """
    if contr.relationship != "contradiction":
        return contr
    if classify_mutability(text_a) != "volatile" or classify_mutability(text_b) != "volatile":
        return contr
    return replace(
        contr,
        relationship="evolution",
        rationale=(contr.rationale or "") + " [mutability: volatile-vs-volatile → evolution]",
    )


@dataclass(frozen=True)
class PairRecord:
    """A persisted contradiction pair."""

    pair_id: int
    memory_id_a: str
    memory_id_b: str
    relationship: str
    confidence: float
    rationale: str
    status: str
    detected_at: str
    resolved_at: str | None
    resolution_note: str | None


_PAIR_COLS = (
    "pair_id, memory_id_a, memory_id_b, relationship, confidence,"
    " rationale, status, detected_at, resolved_at, resolution_note"
)


class ContradictionStore:
    """Legacy sidecar store retained as a read/import compatibility source.

    Lifecycle of a pair:
      1. `upsert_open(...)` — scanner inserts a newly detected pair.
         If the same pair already exists with status `open`, it is
         updated in-place (newer LLM verdict wins). Pairs already
         resolved (`fused`/`dismissed`/etc.) are left alone so the
         user does not re-litigate them.
      2. `list_open(...)` / `get(pair_id)` — triage walker reads.
      3. `resolve(pair_id, status, note)` — triage walker writes the
         resolution. Status must be one of `VALID_STATUSES` (minus
         `open`).
      4. `drop_for_memoria(memory_id)` — called when a memory is
         deleted so dangling pairs vanish.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so the one shared connection survives the
        # FastMCP worker threadpool; _tx() is serialised by _tx_lock (a single
        # connection can't hold two concurrent BEGIN IMMEDIATE transactions).
        self._conn = sqlite3.connect(
            self.db_path,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        # WAL + synchronous BEFORE the schema DDL — executescript commits in the
        # current journal mode, so setting WAL afterwards leaves the cold-open
        # schema in rollback-journal mode.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        # Migrate pre-rename DBs (memoria_id_a/b -> memory_id_a/b) BEFORE the
        # IF NOT EXISTS DDL, which would otherwise skip the existing `pairs`
        # table and leave the new code querying columns that don't exist.
        from memo.util import rename_legacy_columns

        rename_legacy_columns(
            self._conn, "pairs", {"memoria_id_a": "memory_id_a", "memoria_id_b": "memory_id_b"}
        )
        self._conn.executescript(_SCHEMA_DDL)
        self._tx_lock = threading.Lock()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._tx_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def close(self) -> None:
        with suppress(BaseException):
            self._conn.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        with suppress(BaseException):
            self.close()

    def upsert_open(
        self,
        memory_id_a: str,
        memory_id_b: str,
        relationship: str,
        confidence: float,
        rationale: str,
    ) -> int:
        """Insert a newly detected pair, or refresh it if still open.

        Resolved pairs are NOT overwritten — the user's verdict beats
        any subsequent LLM re-scan.

        Returns the pair_id.
        """
        a, b = _canonical_pair(memory_id_a, memory_id_b)
        now = datetime.now(UTC).isoformat()
        with self._tx() as cx:
            existing = cx.execute(
                "SELECT pair_id, status FROM pairs WHERE memory_id_a=? AND memory_id_b=?",
                (a, b),
            ).fetchone()
            if existing is None:
                cur = cx.execute(
                    "INSERT INTO pairs (memory_id_a, memory_id_b, relationship, "
                    "confidence, rationale, status, detected_at) "
                    "VALUES (?, ?, ?, ?, ?, 'open', ?)",
                    (a, b, relationship, confidence, rationale, now),
                )
                return int(cur.lastrowid or 0)
            if existing["status"] == "open":
                cx.execute(
                    "UPDATE pairs SET relationship=?, confidence=?, rationale=?, "
                    "detected_at=? WHERE pair_id=?",
                    (relationship, confidence, rationale, now, existing["pair_id"]),
                )
            return int(existing["pair_id"])

    def already_resolved(self, memory_id_a: str, memory_id_b: str) -> bool:
        """Did the user already give a verdict on this pair?"""
        a, b = _canonical_pair(memory_id_a, memory_id_b)
        row = self._conn.execute(
            "SELECT status FROM pairs WHERE memory_id_a=? AND memory_id_b=?",
            (a, b),
        ).fetchone()
        return bool(row and row["status"] != "open")

    def is_open_pair(self, memory_id_a: str, memory_id_b: str) -> bool:
        a, b = _canonical_pair(memory_id_a, memory_id_b)
        row = self._conn.execute(
            "SELECT status FROM pairs WHERE memory_id_a=? AND memory_id_b=?", (a, b)
        ).fetchone()
        return bool(row and row["status"] == "open")

    def list_open(
        self,
        limit: int = 50,
        min_confidence: float = 0.0,
        relationship: str | None = None,
    ) -> list[PairRecord]:
        sql = f"SELECT {_PAIR_COLS} FROM pairs WHERE status='open' AND confidence >= ? "  # noqa: S608
        params: list[Any] = [min_confidence]
        if relationship:
            sql += "AND relationship = ? "
            params.append(relationship)
        sql += "ORDER BY confidence DESC, detected_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_all(
        self,
        status: str | None = None,
        limit: int = 200,
    ) -> list[PairRecord]:
        sql = f"SELECT {_PAIR_COLS} FROM pairs "  # noqa: S608
        params: list[Any] = []
        if status:
            sql += "WHERE status = ? "
            params.append(status)
        sql += "ORDER BY detected_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get(self, pair_id: int) -> PairRecord | None:
        row = self._conn.execute(
            f"SELECT {_PAIR_COLS} FROM pairs WHERE pair_id=?",  # noqa: S608
            (pair_id,),
        ).fetchone()
        return self._row_to_record(row) if row else None

    def resolve(
        self,
        pair_id: int,
        status: str,
        note: str | None = None,
    ) -> bool:
        if status not in VALID_STATUSES or status == "open":
            raise ValueError(f"invalid resolution status: {status!r}")
        now = datetime.now(UTC).isoformat()
        with self._tx() as cx:
            cur = cx.execute(
                "UPDATE pairs SET status=?, resolved_at=?, resolution_note=? WHERE pair_id=?",
                (status, now, note, pair_id),
            )
            return cur.rowcount > 0

    def reopen(self, pair_id: int) -> bool:
        """Send a resolved pair back to the open queue."""
        with self._tx() as cx:
            cur = cx.execute(
                "UPDATE pairs SET status='open', resolved_at=NULL, "
                "resolution_note=NULL WHERE pair_id=?",
                (pair_id,),
            )
            return cur.rowcount > 0

    def drop_for_memoria(self, memory_id: str) -> int:
        """Delete all pairs touching this memory (called on memory delete)."""
        with self._tx() as cx:
            cur = cx.execute(
                "DELETE FROM pairs WHERE memory_id_a=? OR memory_id_b=?",
                (memory_id, memory_id),
            )
            return int(cur.rowcount or 0)

    def pairs_for_ids(
        self,
        ids: list[str],
        *,
        status: str = "open",
    ) -> list[PairRecord]:
        """Return pairs where either side is one of ``ids``.

        Used by the search pipeline to apply contradiction penalties to
        retrieved results without scanning the full pairs table.
        """
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        sql = (
            f"SELECT {_PAIR_COLS} FROM pairs WHERE status=? "  # noqa: S608
            f"AND (memory_id_a IN ({placeholders}) OR memory_id_b IN ({placeholders}))"
        )
        rows = self._conn.execute(sql, [status, *ids, *ids]).fetchall()
        return [self._row_to_record(r) for r in rows]

    def stats(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM pairs GROUP BY status"
        ).fetchall()
        return {r["status"]: int(r["n"]) for r in rows}

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> PairRecord:
        return PairRecord(
            pair_id=int(row["pair_id"]),
            memory_id_a=row["memory_id_a"],
            memory_id_b=row["memory_id_b"],
            relationship=row["relationship"],
            confidence=float(row["confidence"]),
            rationale=row["rationale"] or "",
            status=row["status"],
            detected_at=row["detected_at"],
            resolved_at=row["resolved_at"],
            resolution_note=row["resolution_note"],
        )


def _canonical_pair_id(row: dict[str, Any]) -> int:
    migration_key = str(row.get("migration_key") or "")
    if migration_key.startswith("legacy-contradiction:"):
        with suppress(ValueError):
            return int(migration_key.rsplit(":", 1)[-1])
    relation_id = str(row.get("id") or "")
    digest = relation_id.removeprefix("rel-")[:15]
    with suppress(ValueError):
        return int(digest, 16)
    return int(uuid.uuid5(uuid.NAMESPACE_URL, relation_id).int & ((1 << 63) - 1))


def _legacy_status(row: dict[str, Any]) -> str:
    provenance = row.get("provenance") or {}
    migrated_status = provenance.get("legacy_status") if isinstance(provenance, dict) else None
    if migrated_status in VALID_STATUSES:
        return str(migrated_status)
    state = str(row.get("judgment_status") or "pending")
    relation = str(row.get("relation") or "")
    if state == "pending":
        return "open"
    return {
        "not_conflict": "dismissed",
        "related": "evolved",
        "compatible": "fused",
        "supersedes": "kept_newer",
        "conflicts_with": "competing",
        "scoped": "competing",
    }.get(relation, "dismissed")


class CanonicalContradictionAdapter:
    """Old contradiction API projected from the canonical relation ledger."""

    def __init__(self, memory: Any) -> None:
        self.memory = memory

    def close(self) -> None:
        return None

    def upsert_open(
        self,
        memory_id_a: str,
        memory_id_b: str,
        relationship: str,
        confidence: float,
        rationale: str,
    ) -> int:
        suggested = "conflicts_with" if relationship == "contradiction" else "related"
        row = self.memory.store.create_relation_candidate(
            source_id=memory_id_a,
            target_id=memory_id_b,
            suggested_relation=suggested,
            reason=rationale,
            confidence=confidence,
            provenance={"generator": "contradiction_scanner"},
        )
        return _canonical_pair_id(row)

    def already_resolved(self, memory_id_a: str, memory_id_b: str) -> bool:
        row = self.memory.store.find_relation_pair(memory_id_a, memory_id_b)
        return bool(row and row.get("judgment_status") != "pending")

    def is_open_pair(self, memory_id_a: str, memory_id_b: str) -> bool:
        row = self.memory.store.find_relation_pair(memory_id_a, memory_id_b)
        return bool(row and row.get("judgment_status") == "pending")

    def _record(self, row: dict[str, Any]) -> PairRecord:
        relation = str(row.get("relation") or "")
        relationship = "contradiction" if relation == "conflicts_with" else "evolution"
        status = _legacy_status(row)
        updated = str(row.get("updated_at") or row.get("created_at") or "")
        return PairRecord(
            pair_id=_canonical_pair_id(row),
            memory_id_a=str(row.get("source_id") or ""),
            memory_id_b=str(row.get("target_id") or ""),
            relationship=relationship,
            confidence=float(row.get("confidence") or 0.0),
            rationale=str(row.get("reason") or ""),
            status=status,
            detected_at=str(row.get("created_at") or ""),
            resolved_at=updated if status != "open" else None,
            resolution_note=str(row.get("reason") or "") or None,
        )

    @staticmethod
    def _compatible_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows:
            provenance = row.get("provenance") or {}
            generator = provenance.get("generator") if isinstance(provenance, dict) else None
            if (
                generator == "contradiction_scanner"
                or row.get("migrated_from") == "contradictions.db"
                or row.get("relation")
                in {"conflicts_with", "related", "supersedes", "compatible", "not_conflict"}
            ):
                out.append(row)
        return out

    def list_open(
        self,
        limit: int = 50,
        min_confidence: float = 0.0,
        relationship: str | None = None,
    ) -> list[PairRecord]:
        rows = self._compatible_rows(
            self.memory.store.list_relations(status="pending", limit=max(limit * 3, limit))
        )
        records = [self._record(row) for row in rows]
        records = [record for record in records if record.confidence >= min_confidence]
        if relationship:
            records = [record for record in records if record.relationship == relationship]
        return records[:limit]

    def list_all(self, status: str | None = None, limit: int = 200) -> list[PairRecord]:
        rows = self._compatible_rows(self.memory.store.list_relations(limit=max(limit * 3, limit)))
        records = [self._record(row) for row in rows]
        if status:
            records = [record for record in records if record.status == status]
        return records[:limit]

    def get(self, pair_id: int) -> PairRecord | None:
        return next(
            (record for record in self.list_all(limit=1000) if record.pair_id == pair_id), None
        )

    def _row_for_pair_id(self, pair_id: int) -> dict[str, Any] | None:
        return next(
            (
                row
                for row in self._compatible_rows(self.memory.store.list_relations(limit=1000))
                if _canonical_pair_id(row) == pair_id
            ),
            None,
        )

    def resolve(self, pair_id: int, status: str, note: str | None = None) -> bool:
        if status not in VALID_STATUSES or status == "open":
            raise ValueError(f"invalid resolution status: {status!r}")
        row = self._row_for_pair_id(pair_id)
        if row is None:
            return False
        relation = {
            "dismissed": "not_conflict",
            "evolved": "related",
            "fused": "compatible",
            "competing": "conflicts_with",
            "kept_newer": "supersedes",
            "kept_older": "supersedes",
        }[status]
        if status in {"kept_newer", "kept_older"}:
            first = self.memory.get(str(row["source_id"]))
            second = self.memory.get(str(row["target_id"]))
            if first is not None and second is not None:
                older, newer = (
                    (first, second)
                    if _as_aware(first.updated) <= _as_aware(second.updated)
                    else (second, first)
                )
                source, target = (newer, older) if status == "kept_newer" else (older, newer)
                row = self.memory.store.reorient_pending_relation(
                    str(row["id"]), source_id=source.id, target_id=target.id
                )
        self.memory.judge_relation(
            str(row["id"]),
            relation,
            reason=note or f"legacy contradiction resolution: {status}",
            actor_kind="compatibility",
            provenance={"legacy_status": status},
        )
        return True

    def reopen(self, pair_id: int) -> bool:
        row = self._row_for_pair_id(pair_id)
        return bool(row and self.memory.store.reopen_relation(str(row["id"])))

    def drop_for_memoria(self, memory_id: str) -> int:
        return self.memory.store.orphan_relations_for(memory_id)

    def pairs_for_ids(self, ids: list[str], *, status: str = "open") -> list[PairRecord]:
        if not ids:
            return []
        rows = self._compatible_rows(self.memory.store.list_relations(memory_ids=ids, limit=1000))
        records = [self._record(row) for row in rows]
        return [record for record in records if record.status == status]

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.list_all(limit=1000):
            counts[record.status] = counts.get(record.status, 0) + 1
        return counts


@dataclass(frozen=True)
class ScanResult:
    scanned_memories: int
    pairs_examined: int
    pairs_inserted: int
    pairs_refreshed: int
    pairs_skipped_resolved: int
    contradictions_found: int
    evolutions_found: int
    time_limited: bool = False


class ContradictionScanner:
    """Corpus-wide contradiction detection driven by vec neighborhoods.

    For each memory, the scanner fetches the top-K vec neighbors (above
    a cosine floor), canonicalizes each (self, neighbor) pair, skips
    pairs already resolved or already open in the store, asks the LLM
    to classify the pair, and persists `contradiction` / `evolution`
    verdicts.

    Args:
        memory: Memory instance (provides embedder, store, get).
        store: ContradictionStore that backs persistence.
        analyzer: Optional TemporalAnalyzer (LLM classifier). If None
            uses `memory.temporal`.
    """

    def __init__(
        self,
        memory: Any,
        store: Any,
        analyzer: TemporalAnalyzer | None = None,
    ) -> None:
        self.memory = memory
        self.store = store
        self._analyzer = analyzer

    def _ensure_analyzer(self) -> TemporalAnalyzer:
        if self._analyzer is None:
            self._analyzer = self.memory.temporal
        return self._analyzer

    def scan_corpus(
        self,
        top_k: int = 5,
        sim_floor: float = 0.55,
        confidence_threshold: float = 0.7,
        min_days_apart: int = 0,
        max_memories: int = 2000,
        max_pairs: int = 500,
        since: str | None = None,
        type_: str | None = None,
        progress: Any = None,
        persist: bool = True,
        max_seconds: float | None = None,
    ) -> ScanResult:
        """Walk the corpus, classify near-neighbors, persist contradictions.

        Args:
            top_k: Neighbors to pull per memory via vec search.
            sim_floor: Skip neighbors with cosine below this. Cheap
                prefilter so the LLM only sees pairs that are at least
                topically related.
            confidence_threshold: LLM verdicts below this don't get stored.
            min_days_apart: Pairs whose `updated` timestamps are closer
                than this are skipped (same-day edits are usually
                revisions, not contradictions).
            max_memories: Hard cap on memories visited per run.
            max_pairs: Hard cap on pairs sent to the LLM per run.
            since: ISO date string; only memories `updated >= since` are
                used as scan anchors. Useful for incremental runs.
            type_: Optional type filter (e.g. only `decision` memories).
            progress: Optional callable `fn(current, total, title)` for
                CLI progress bars.
            persist: Store detected pairs and emit anomaly events. Set to
                False for read-only preview runs.

        Returns:
            ScanResult with counters.
        """
        # Incremental: push `since` into the DB so the freshest anchors are
        # returned within `max_memories`, instead of paging older rows and
        # filtering client-side (which dropped new anchors past the limit).
        records = self.memory.list(limit=max_memories, type_=type_, updated_since=since)

        scanned = 0
        examined = 0
        inserted = 0
        refreshed = 0
        skipped_resolved = 0
        contradictions = 0
        evolutions = 0

        seen_pairs: set[tuple[str, str]] = set()
        total = len(records)
        _deadline = time.monotonic() + max_seconds if max_seconds else None
        _time_limited = False

        for idx, rec in enumerate(records):
            scanned += 1
            if progress:
                with suppress(Exception):
                    progress(idx + 1, total, rec.title)

            if examined >= max_pairs:
                break

            if _deadline and time.monotonic() >= _deadline:
                _time_limited = True
                _log.warning(
                    "scan_corpus: wall-clock limit %.0fs reached after %d pairs examined",
                    max_seconds,
                    examined,
                )
                break

            body = rec.body or rec.title
            if not body.strip():
                continue

            try:
                emb = self.memory.embedder.embed_query(body)
            except Exception as exc:
                _log.debug("contradiction scanner: embed failed for id=%s: %s", rec.id[:8], exc)
                continue

            neighbors = self.memory.store.search(emb, limit=top_k + 1)
            for nb in neighbors:
                if examined >= max_pairs:
                    break
                if nb["id"] == rec.id:
                    continue
                if float(nb.get("score", 0.0)) < sim_floor:
                    continue

                pair_key = _canonical_pair(rec.id, nb["id"])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                if self.store.already_resolved(*pair_key):
                    skipped_resolved += 1
                    continue

                other = self.memory.get(nb["id"])
                if other is None:
                    continue

                if not _enough_days_apart(rec.updated, other.updated, min_days_apart):
                    continue

                examined += 1
                analyzer = self._ensure_analyzer()
                contr = analyzer._classify_pair(rec, other)
                if contr is None:
                    continue
                if contr.confidence < confidence_threshold:
                    continue
                if flag_bool("MEMO_CONTRADICT_MUTABILITY"):
                    contr = downgrade_volatile_contradiction(
                        contr, rec.body or rec.title, other.body or other.title
                    )
                if contr.relationship not in ("contradiction", "evolution"):
                    continue

                if persist:
                    # already_resolved pairs were `continue`d above, so only the
                    # open-state check is meaningful here.
                    existed = _is_open(self.store, *pair_key)
                    self.store.upsert_open(
                        memory_id_a=contr.memory_id_a,
                        memory_id_b=contr.memory_id_b,
                        relationship=contr.relationship,
                        confidence=contr.confidence,
                        rationale=contr.rationale,
                    )
                    if existed:
                        refreshed += 1
                    else:
                        inserted += 1

                if contr.relationship == "contradiction":
                    contradictions += 1
                    if persist:
                        emit_anomaly(
                            contr.memory_id_a,
                            contr.memory_id_b,
                            contr.relationship,
                            contr.confidence,
                            "open",
                            operational=getattr(self.memory, "operational", None),
                        )
                else:
                    evolutions += 1

        return ScanResult(
            scanned_memories=scanned,
            pairs_examined=examined,
            pairs_inserted=inserted,
            pairs_refreshed=refreshed,
            pairs_skipped_resolved=skipped_resolved,
            contradictions_found=contradictions,
            evolutions_found=evolutions,
            time_limited=_time_limited,
        )


def _enough_days_apart(a: str, b: str, min_days: int) -> bool:
    if min_days <= 0:
        return True
    try:
        da = datetime.fromisoformat(a.replace("Z", "+00:00"))
        db = datetime.fromisoformat(b.replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError):
        return True
    return abs((da - db).total_seconds()) / 86400 >= min_days


def _is_open(store: Any, a: str, b: str) -> bool:
    return bool(store.is_open_pair(a, b))


def emit_anomaly(
    memory_id_a: str,
    memory_id_b: str,
    relationship: str,
    confidence: float,
    status: str,
    *,
    operational: Any | None = None,
) -> str | None:
    """Append a native anomaly event and return its deterministic id."""
    anomaly_id = (
        "anomaly-"
        + uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"memo:semantic_contradiction:{memory_id_a}:{memory_id_b}",
        ).hex[:24]
    )
    state = "detected" if status == "open" else "resolved"
    severity = "high" if confidence >= 0.9 else "medium" if confidence >= 0.75 else "low"
    if operational is not None:
        operational.record_anomaly(
            {
                "anomaly_id": anomaly_id,
                "kind": "semantic_contradiction",
                "state": state,
                "summary": (
                    f"memo {relationship} between memories "
                    f"{memory_id_a[:12]} and {memory_id_b[:12]}"
                ),
                "evidence_uris": [
                    f"memo://memoria/{memory_id_a}",
                    f"memo://memoria/{memory_id_b}",
                ],
                "severity": severity,
                "memory_id_a": memory_id_a,
                "memory_id_b": memory_id_b,
                "relationship": relationship,
                "confidence": confidence,
                "status": status,
                "created_at": datetime.now(UTC).isoformat(timespec="milliseconds"),
            }
        )
    return anomaly_id


def is_stale(updated_iso: str, days_threshold: int) -> bool:
    """Helper: is this timestamp older than `days_threshold` days?"""
    try:
        dt = datetime.fromisoformat(updated_iso.replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError):
        return False
    return (datetime.now(UTC) - dt) > timedelta(days=days_threshold)


__all__ = [
    "VALID_STATUSES",
    "ContradictionScanner",
    "ContradictionStore",
    "PairRecord",
    "ScanResult",
    "emit_anomaly",
    "is_stale",
]
