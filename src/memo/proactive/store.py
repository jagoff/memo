from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from types import TracebackType

from .nudge import Nudge

_DDL = """
CREATE TABLE IF NOT EXISTS proactive_candidates (
    id TEXT PRIMARY KEY, kind TEXT, urgency REAL, value REAL,
    title TEXT, detail TEXT, evidence_json TEXT, action TEXT,
    created_at TEXT, ttl_days INTEGER);
CREATE TABLE IF NOT EXISTS proactive_state (
    id TEXT PRIMARY KEY, dismissed_at TEXT);
CREATE TABLE IF NOT EXISTS proactive_kind_snooze (
    kind TEXT PRIMARY KEY, snoozed_until TEXT);
CREATE TABLE IF NOT EXISTS proactive_feedback (
    id TEXT, kind TEXT, outcome TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS proactive_push_log (ts TEXT);
"""


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


class ProactiveStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Match the sidecar-store connection model (graph/history/crossref): a
        # WAL connection with a real busy timeout and check_same_thread=False so
        # a dream-refresh writer and a briefing reader don't hit SQLITE_BUSY
        # inside the 5s default, and a future threaded holder can't raise.
        self._conn = sqlite3.connect(str(db_path), timeout=10.0, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        # Bound the WAL: long-lived readers (daemons, MCP sessions) pin
        # snapshots, so a passive checkpoint never truncates on its own
        # (graph.db-wal was found at 80MB against a 127MB database).
        self._conn.execute("PRAGMA journal_size_limit=16777216")
        self._conn.execute("PRAGMA busy_timeout=10000")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)

    def put_candidates(self, nudges: list[Nudge]) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM proactive_candidates")
            self._conn.executemany(
                # OR REPLACE: `id` is a content hash of (kind, subject_id), so a
                # duplicate in one batch must not abort the refresh — the caller
                # already dedups, this keeps the write itself total.
                "INSERT OR REPLACE INTO proactive_candidates VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        n.id,
                        n.kind,
                        n.urgency,
                        n.value,
                        n.title,
                        n.detail,
                        json.dumps(list(n.evidence)),
                        n.action,
                        n.created_at,
                        n.ttl_days,
                    )
                    for n in nudges
                ],
            )

    def _snoozed_kinds(self, now: str) -> set[str]:
        rows = self._conn.execute(
            "SELECT kind FROM proactive_kind_snooze WHERE snoozed_until > ?", (now,)
        ).fetchall()
        return {r["kind"] for r in rows}

    def active_candidates(self, now: str) -> list[Nudge]:
        dismissed = {
            r["id"] for r in self._conn.execute("SELECT id FROM proactive_state").fetchall()
        }
        snoozed = self._snoozed_kinds(now)
        out: list[Nudge] = []
        for r in self._conn.execute("SELECT * FROM proactive_candidates").fetchall():
            if r["id"] in dismissed or r["kind"] in snoozed:
                continue
            if _parse(r["created_at"]) + timedelta(days=r["ttl_days"]) <= _parse(now):
                continue
            out.append(
                Nudge(
                    id=r["id"],
                    kind=r["kind"],
                    urgency=r["urgency"],
                    value=r["value"],
                    title=r["title"],
                    evidence=tuple(json.loads(r["evidence_json"])),
                    created_at=r["created_at"],
                    detail=r["detail"] or "",
                    action=r["action"],
                    ttl_days=r["ttl_days"],
                )
            )
        return out

    def dismiss(self, nudge_id: str, now: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO proactive_state VALUES (?, ?)", (nudge_id, now)
            )

    def snooze_kind(self, kind: str, until: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO proactive_kind_snooze VALUES (?, ?)", (kind, until)
            )

    def record_feedback(self, nudge_id: str, kind: str, outcome: str, ts: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO proactive_feedback VALUES (?,?,?,?)", (nudge_id, kind, outcome, ts)
            )

    def kind_multipliers(self, floor: float, *, since: str | None = None) -> dict[str, float]:
        """Aggregate `proactive_feedback` into a per-kind demotion multiplier.

        `since` (ISO timestamp), when given, windows the aggregation to rows
        with `ts >= since` so old dismissals don't cause permanent decay —
        callers thread a rolling window (e.g. `compute_routed` passes
        `now - 30d`). `None` (the direct-call default) aggregates all rows.
        """
        out: dict[str, float] = {}
        query = "SELECT kind, outcome, COUNT(*) c FROM proactive_feedback"
        params: tuple[str, ...] = ()
        if since is not None:
            query += " WHERE ts >= ?"
            params = (since,)
        query += " GROUP BY kind, outcome"
        rows = self._conn.execute(query, params).fetchall()
        agg: dict[str, dict[str, int]] = {}
        for r in rows:
            agg.setdefault(r["kind"], {})[r["outcome"]] = r["c"]
        for kind, counts in agg.items():
            acted = counts.get("acted", 0)
            noise = counts.get("dismissed", 0) + counts.get("ignored", 0)
            total = acted + noise
            mult = 1.0 if total == 0 else max(floor, min(1.0, (acted + 1) / (total + 1)))
            out[kind] = mult
        return out

    def last_push_at(self) -> str | None:
        r = self._conn.execute("SELECT MAX(ts) m FROM proactive_push_log").fetchone()
        return r["m"] if r and r["m"] else None

    def mark_pushed(self, ts: str) -> None:
        with self._conn:
            self._conn.execute("INSERT INTO proactive_push_log VALUES (?)", (ts,))

    def pushes_today(self, day: str) -> int:
        r = self._conn.execute(
            "SELECT COUNT(*) c FROM proactive_push_log WHERE ts LIKE ?", (day + "%",)
        ).fetchone()
        return int(r["c"])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ProactiveStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
