"""TurnStore — lexical (FTS5, no embeddings) turn-level verbatim index.

Sidecar over `.claude/projects/**/*.jsonl` transcripts already on disk: "what
did we say EXACTLY when we fixed X". Rebuildable from the transcripts; never
enters the recall hook (CLAUDE.md — cognition verbs stay off memo's automatic
surfaces; this is an on-demand, explicit-query-only store). Folds into
`memvec.db` when `MEMO_SINGLE_DB` is on; otherwise lives in its own
`verbatim.db` (see `Config.verbatim_db`).
"""

from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .connection import _ConnectionHolder, _ConnectionMixin
from .schema import _BM25_ES_STOPWORDS

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
MAX_VERBATIM_RESULTS = 100


class TurnStore(_ConnectionMixin):
    """sqlite FTS5-backed store of transcript turns. No vectors — lexical only."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._local = threading.local()
        # Strong references (matching VecStore): CPython does not guarantee the
        # order in which a terminating thread clears its ``threading.local``
        # values and finalizes sqlite connections, so ``close()`` (or the
        # dead-owner sweep in ``_connect``) is the single deterministic cleanup
        # boundary. A WeakSet would let a holder vanish before either fires.
        self._conn_holders: set[_ConnectionHolder] = set()
        self._conn_holders_lock = threading.Lock()
        self.db_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.db_path.parent.chmod(0o700)
        self._ensure_schema()
        self._secure_storage()

    def _secure_storage(self) -> None:
        """Transcript text and SQLite sidecars are private to the current user."""
        for path in (
            self.db_path,
            Path(f"{self.db_path}-wal"),
            Path(f"{self.db_path}-shm"),
        ):
            if path.exists():
                path.chmod(0o600)

    def _load_vec0(self, conn: sqlite3.Connection) -> None:
        # Turns are scalar rows indexed by FTS5; no vector extension needed.
        return None

    def _ensure_schema(self) -> None:
        conn = self._conn
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS turns (
                session_id TEXT NOT NULL,
                turn_idx INTEGER NOT NULL,
                agent TEXT,
                role TEXT,
                ts TEXT,
                text TEXT,
                PRIMARY KEY (session_id, turn_idx)
            );

            CREATE INDEX IF NOT EXISTS idx_turns_ts ON turns(ts);
            """
        )
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5("
            "session_id UNINDEXED, turn_idx UNINDEXED, text, "
            "tokenize='unicode61 remove_diacritics 2'"
            ")"
        )
        conn.commit()

    def replace_session(self, session_id: str, agent: str, turns: list[dict[str, Any]]) -> int:
        """Delete + insert all turns for `session_id` (transactional).

        Idempotent — re-ingesting a session that grew (or was re-parsed)
        simply swaps its rows in both `turns` and `turns_fts`.
        """
        with self._tx() as cx:
            cx.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            cx.execute("DELETE FROM turns_fts WHERE session_id = ?", (session_id,))
            for turn in turns:
                idx = int(turn["idx"])
                role = turn.get("role")
                ts = turn.get("ts")
                text = turn.get("text") or ""
                cx.execute(
                    "INSERT INTO turns (session_id, turn_idx, agent, role, ts, text) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (session_id, idx, agent, role, ts, text),
                )
                cx.execute(
                    "INSERT INTO turns_fts (session_id, turn_idx, text) VALUES (?, ?, ?)",
                    (session_id, idx, text),
                )
        return len(turns)

    def _match_expr(self, query: str) -> list[str] | None:
        raw_tokens = [t for t in _TOKEN_RE.findall(query) if t]
        if not raw_tokens:
            return None
        filtered = [t for t in raw_tokens if t.lower() not in _BM25_ES_STOPWORDS]
        return filtered if len(filtered) >= 2 else raw_tokens

    def _run_search(
        self,
        tokens: list[str],
        joiner: str,
        *,
        limit: int,
        session_id: str | None,
        since: str | None,
    ) -> list[sqlite3.Row]:
        expr = joiner.join(f'"{t}"' for t in tokens)
        sql = (
            "SELECT turns.session_id AS session_id, turns.turn_idx AS turn_idx, "
            "turns.role AS role, turns.ts AS ts, "
            "snippet(turns_fts, 2, '[', ']', '...', 10) AS snippet, "
            "bm25(turns_fts) AS bm25_score "
            "FROM turns_fts JOIN turns "
            "ON turns.session_id = turns_fts.session_id AND turns.turn_idx = turns_fts.turn_idx "
            "WHERE turns_fts MATCH ? "
        )
        params: list[Any] = [expr]
        if session_id is not None:
            sql += "AND turns.session_id = ? "
            params.append(session_id)
        if since is not None:
            sql += "AND turns.ts >= ? "
            params.append(since)
        sql += "ORDER BY bm25_score ASC LIMIT ?"
        params.append(max(1, min(int(limit), MAX_VERBATIM_RESULTS)))
        try:
            return list(self._conn.execute(sql, params).fetchall())
        except sqlite3.OperationalError:
            return []

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        session_id: str | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        """FTS5 MATCH over indexed turns.

        Tokenizes on `\\w+`, AND-joins tokens (each phrase-quoted) so a
        multi-word query matches any turn containing all tokens (order-
        independent), stripping Spanish stopwords when ≥2 content tokens
        remain. Falls back to an OR-join only when the AND match returns
        zero rows. `session_id`/`since` filter the join (`since` compares
        ISO-formatted `ts` lexicographically). Ranked by FTS5 bm25().
        """
        tokens = self._match_expr(query)
        if not tokens:
            return []
        rows = self._run_search(tokens, " ", limit=limit, session_id=session_id, since=since)
        if not rows and len(tokens) >= 2:
            rows = self._run_search(tokens, " OR ", limit=limit, session_id=session_id, since=since)
        out: list[dict[str, Any]] = []
        for row in rows:
            bm = float(row["bm25_score"])
            score = 1.0 - 1.0 / (1.0 + abs(bm)) if bm < 0 else 0.0
            out.append(
                {
                    "session_id": row["session_id"],
                    "turn_idx": row["turn_idx"],
                    "role": row["role"],
                    "ts": row["ts"],
                    "snippet": row["snippet"],
                    "score": score,
                }
            )
        return out

    def prune_older_than(self, days: int) -> int:
        """Delete turns whose `ts` is older than `days` ago. Returns rows removed.

        Timestamps are canonical UTC ISO8601 strings. Legacy null or malformed
        rows are removed conservatively so they cannot bypass retention.
        """
        from datetime import UTC, datetime, timedelta

        cutoff = (datetime.now(UTC) - timedelta(days=max(1, int(days)))).isoformat(
            timespec="seconds"
        )
        stale = "ts IS NULL OR datetime(ts) IS NULL OR datetime(ts) < datetime(?)"
        with self._tx() as cx:
            rows = cx.execute(
                f"SELECT session_id, turn_idx FROM turns WHERE {stale}",  # noqa: S608
                (cutoff,),
            ).fetchall()
            cur = cx.execute(f"DELETE FROM turns WHERE {stale}", (cutoff,))  # noqa: S608
            for row in rows:
                cx.execute(
                    "DELETE FROM turns_fts WHERE session_id = ? AND turn_idx = ?",
                    (row["session_id"], row["turn_idx"]),
                )
        return int(cur.rowcount)

    def stats(self) -> dict[str, int]:
        sessions = self._conn.execute(
            "SELECT COUNT(DISTINCT session_id) AS n FROM turns"
        ).fetchone()["n"]
        turns = self._conn.execute("SELECT COUNT(*) AS n FROM turns").fetchone()["n"]
        return {"sessions": int(sessions), "turns": int(turns)}


__all__ = ["MAX_VERBATIM_RESULTS", "TurnStore"]
