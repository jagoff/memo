from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

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
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)

    def put_candidates(self, nudges: list[Nudge]) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM proactive_candidates")
            self._conn.executemany(
                "INSERT INTO proactive_candidates VALUES (?,?,?,?,?,?,?,?,?,?)",
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

    def kind_multipliers(self, floor: float) -> dict[str, float]:
        out: dict[str, float] = {}
        rows = self._conn.execute(
            "SELECT kind, outcome, COUNT(*) c FROM proactive_feedback GROUP BY kind, outcome"
        ).fetchall()
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
