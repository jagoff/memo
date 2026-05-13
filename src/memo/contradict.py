"""Persistent contradiction & staleness radar for the memory corpus.

This module sits on top of `TemporalAnalyzer` (which already classifies
pairs as contradiction / evolution / consistent / unrelated) and adds
the missing pieces needed for a triage workflow:

- **Sidecar DB** (`contradictions.db`) so the LLM verdict on a pair is
  not recomputed every time. Status of each pair lives across runs.
- **Corpus-wide scan** that walks every memoria, finds near-neighbors
  via vec search, and classifies the pairs that look promising. Pair
  IDs are canonical (lower id first) so the same pair is never stored
  twice.
- **Resolution API** that the CLI / MCP triage walker calls to mark a
  pair as fused / kept-newer / kept-older / evolved / dismissed.
- **Stale links**: tags pairs whose older side is past a staleness
  cutoff so the triage UI can foreground them.

The store is kept in its own sqlite file (mirrors `history.db` /
`graph.db`) to avoid WAL contention with the hot vec reader.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from memo.temporal import TemporalAnalyzer

VALID_STATUSES = {
    "open",          # pending triage
    "fused",         # merged into a new memoria
    "kept_newer",    # newer side won, older deleted/archived
    "kept_older",    # older side won (rare; explicit user choice)
    "evolved",       # both kept, marked as legitimate evolution
    "dismissed",     # false positive
}


_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS pairs (
    pair_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    memoria_id_a   TEXT NOT NULL,
    memoria_id_b   TEXT NOT NULL,
    relationship   TEXT NOT NULL,
    confidence     REAL NOT NULL,
    rationale      TEXT,
    status         TEXT NOT NULL DEFAULT 'open',
    detected_at    TEXT NOT NULL,
    resolved_at    TEXT,
    resolution_note TEXT,
    UNIQUE(memoria_id_a, memoria_id_b)
);
CREATE INDEX IF NOT EXISTS idx_pairs_status ON pairs(status);
CREATE INDEX IF NOT EXISTS idx_pairs_a ON pairs(memoria_id_a);
CREATE INDEX IF NOT EXISTS idx_pairs_b ON pairs(memoria_id_b);
"""


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    """Order a pair so the same two ids always hash to the same row."""
    return (a, b) if a <= b else (b, a)


@dataclass(frozen=True)
class PairRecord:
    """A persisted contradiction pair."""
    pair_id: int
    memoria_id_a: str
    memoria_id_b: str
    relationship: str
    confidence: float
    rationale: str
    status: str
    detected_at: str
    resolved_at: str | None
    resolution_note: str | None


class ContradictionStore:
    """Sidecar sqlite store for contradiction pairs.

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
      4. `drop_for_memoria(memoria_id)` — called when a memoria is
         deleted so dangling pairs vanish.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_DDL)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def upsert_open(
        self,
        memoria_id_a: str,
        memoria_id_b: str,
        relationship: str,
        confidence: float,
        rationale: str,
    ) -> int:
        """Insert a newly detected pair, or refresh it if still open.

        Resolved pairs are NOT overwritten — the user's verdict beats
        any subsequent LLM re-scan.

        Returns the pair_id.
        """
        a, b = _canonical_pair(memoria_id_a, memoria_id_b)
        now = datetime.now(UTC).isoformat()
        with self._tx() as cx:
            existing = cx.execute(
                "SELECT pair_id, status FROM pairs WHERE memoria_id_a=? AND memoria_id_b=?",
                (a, b),
            ).fetchone()
            if existing is None:
                cur = cx.execute(
                    "INSERT INTO pairs (memoria_id_a, memoria_id_b, relationship, "
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

    def already_resolved(self, memoria_id_a: str, memoria_id_b: str) -> bool:
        """Did the user already give a verdict on this pair?"""
        a, b = _canonical_pair(memoria_id_a, memoria_id_b)
        row = self._conn.execute(
            "SELECT status FROM pairs WHERE memoria_id_a=? AND memoria_id_b=?",
            (a, b),
        ).fetchone()
        return bool(row and row["status"] != "open")

    def list_open(
        self,
        limit: int = 50,
        min_confidence: float = 0.0,
        relationship: str | None = None,
    ) -> list[PairRecord]:
        sql = (
            "SELECT * FROM pairs WHERE status='open' AND confidence >= ? "
        )
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
        sql = "SELECT * FROM pairs "
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
            "SELECT * FROM pairs WHERE pair_id=?", (pair_id,),
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
                "UPDATE pairs SET status=?, resolved_at=?, resolution_note=? "
                "WHERE pair_id=?",
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

    def drop_for_memoria(self, memoria_id: str) -> int:
        """Delete all pairs touching this memoria (called on memoria delete)."""
        with self._tx() as cx:
            cur = cx.execute(
                "DELETE FROM pairs WHERE memoria_id_a=? OR memoria_id_b=?",
                (memoria_id, memoria_id),
            )
            return int(cur.rowcount or 0)

    def stats(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM pairs GROUP BY status"
        ).fetchall()
        return {r["status"]: int(r["n"]) for r in rows}

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> PairRecord:
        return PairRecord(
            pair_id=int(row["pair_id"]),
            memoria_id_a=row["memoria_id_a"],
            memoria_id_b=row["memoria_id_b"],
            relationship=row["relationship"],
            confidence=float(row["confidence"]),
            rationale=row["rationale"] or "",
            status=row["status"],
            detected_at=row["detected_at"],
            resolved_at=row["resolved_at"],
            resolution_note=row["resolution_note"],
        )


@dataclass(frozen=True)
class ScanResult:
    scanned_memorias: int
    pairs_examined: int
    pairs_inserted: int
    pairs_refreshed: int
    pairs_skipped_resolved: int
    contradictions_found: int
    evolutions_found: int


class ContradictionScanner:
    """Corpus-wide contradiction detection driven by vec neighborhoods.

    For each memoria, the scanner fetches the top-K vec neighbors (above
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
        store: ContradictionStore,
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
        min_days_apart: int = 1,
        max_memorias: int = 2000,
        max_pairs: int = 500,
        since: str | None = None,
        type_: str | None = None,
        progress: Any = None,
    ) -> ScanResult:
        """Walk the corpus, classify near-neighbors, persist contradictions.

        Args:
            top_k: Neighbors to pull per memoria via vec search.
            sim_floor: Skip neighbors with cosine below this. Cheap
                prefilter so the LLM only sees pairs that are at least
                topically related.
            confidence_threshold: LLM verdicts below this don't get stored.
            min_days_apart: Pairs whose `updated` timestamps are closer
                than this are skipped (same-day edits are usually
                revisions, not contradictions).
            max_memorias: Hard cap on memorias visited per run.
            max_pairs: Hard cap on pairs sent to the LLM per run.
            since: ISO date string; only memorias `updated >= since` are
                used as scan anchors. Useful for incremental runs.
            type_: Optional type filter (e.g. only `decision` memorias).
            progress: Optional callable `fn(current, total, title)` for
                CLI progress bars.

        Returns:
            ScanResult with counters.
        """
        records = self.memory.list(limit=max_memorias, type_=type_)
        if since:
            records = [r for r in records if r.updated >= since]

        scanned = 0
        examined = 0
        inserted = 0
        refreshed = 0
        skipped_resolved = 0
        contradictions = 0
        evolutions = 0

        seen_pairs: set[tuple[str, str]] = set()
        total = len(records)

        for idx, rec in enumerate(records):
            scanned += 1
            if progress:
                try:
                    progress(idx + 1, total, rec.title)
                except Exception:
                    pass

            if examined >= max_pairs:
                break

            body = rec.body or rec.title
            if not body.strip():
                continue

            try:
                emb = self.memory.embedder.embed([body])[0]
            except Exception:
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
                if contr.relationship not in ("contradiction", "evolution"):
                    continue

                existed = self.store.already_resolved(*pair_key) or _is_open(
                    self.store, *pair_key
                )
                self.store.upsert_open(
                    memoria_id_a=contr.memoria_id_a,
                    memoria_id_b=contr.memoria_id_b,
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
                else:
                    evolutions += 1

        return ScanResult(
            scanned_memorias=scanned,
            pairs_examined=examined,
            pairs_inserted=inserted,
            pairs_refreshed=refreshed,
            pairs_skipped_resolved=skipped_resolved,
            contradictions_found=contradictions,
            evolutions_found=evolutions,
        )


def _enough_days_apart(a: str, b: str, min_days: int) -> bool:
    if min_days <= 0:
        return True
    try:
        da = datetime.fromisoformat(a.replace("Z", "+00:00"))
        db = datetime.fromisoformat(b.replace("Z", "+00:00"))
    except Exception:
        return True
    return abs((da - db).days) >= min_days


def _is_open(store: ContradictionStore, a: str, b: str) -> bool:
    row = store._conn.execute(
        "SELECT status FROM pairs WHERE memoria_id_a=? AND memoria_id_b=?",
        (a, b),
    ).fetchone()
    return bool(row and row["status"] == "open")


def is_stale(updated_iso: str, days_threshold: int) -> bool:
    """Helper: is this timestamp older than `days_threshold` days?"""
    try:
        dt = datetime.fromisoformat(updated_iso.replace("Z", "+00:00"))
    except Exception:
        return False
    return (datetime.now(UTC) - dt) > timedelta(days=days_threshold)


__all__ = [
    "VALID_STATUSES",
    "ContradictionScanner",
    "ContradictionStore",
    "PairRecord",
    "ScanResult",
    "is_stale",
]
