"""EXPERIMENTAL — not covered by the test suite, not exposed via MCP. API may change without notice.

Memory versioning & diff UI — track changes, visualize diffs, rollback.

Tracks version history for each memoria:
- Automatic version tracking on updates
- Diff visualization between versions
- Rollback to previous versions
- Version metadata (timestamp, reason, etc.)
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class Version:
    """A version of a memoria."""

    version_id: int
    memoria_id: str
    timestamp: str
    title: str
    type: str
    tags: list[str]
    body: str
    reason: str | None


@dataclass
class DiffResult:
    """Result of comparing two versions."""

    memoria_id: str
    version_a: int
    version_b: int
    unified_diff: str
    changes: list[str]


class VersionStore:
    """Stores version history for memorias.

    Args:
        db_path: Path to the version SQLite database.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = None
        self._init_schema()

    def _get_conn(self):
        import sqlite3

        if self._conn is None:
            # check_same_thread=False + WAL so the shared handle survives the
            # FastMCP worker threadpool (default would raise on the 2nd thread).
            self._conn = sqlite3.connect(
                str(self.db_path),
                timeout=10.0,
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            from contextlib import suppress

            with suppress(sqlite3.Error):
                self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_schema(self) -> None:
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS versions (
                version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                memoria_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                title TEXT NOT NULL,
                type TEXT NOT NULL,
                tags TEXT NOT NULL,
                body TEXT NOT NULL,
                reason TEXT,
                UNIQUE(version_id, memoria_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_versions_memoria ON versions(memoria_id)")
        conn.commit()

    def save_version(
        self,
        memoria_id: str,
        title: str,
        type: str,
        tags: list[str],
        body: str,
        reason: str | None = None,
    ) -> int:
        """Save a new version of a memoria."""
        conn = self._get_conn()
        cursor = conn.execute(
            """
            INSERT INTO versions
            (memoria_id, timestamp, title, type, tags, body, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memoria_id,
                datetime.now(UTC).isoformat(),
                title,
                type,
                json.dumps(tags),
                body,
                reason,
            ),
        )
        conn.commit()
        return cursor.lastrowid

    def get_versions(self, memoria_id: str, limit: int = 10) -> list[Version]:
        """Get all versions of a memoria, most recent first."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT version_id, memoria_id, timestamp, title, type, tags, body, reason"
            " FROM versions WHERE memoria_id = ? ORDER BY version_id DESC LIMIT ?",
            (memoria_id, limit),
        ).fetchall()

        versions = []
        for row in rows:
            versions.append(
                Version(
                    version_id=row["version_id"],
                    memoria_id=row["memoria_id"],
                    timestamp=row["timestamp"],
                    title=row["title"],
                    type=row["type"],
                    tags=json.loads(row["tags"]),
                    body=row["body"],
                    reason=row["reason"],
                )
            )
        return versions

    def get_version(self, memoria_id: str, version_id: int) -> Version | None:
        """Get a specific version of a memoria."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT version_id, memoria_id, timestamp, title, type, tags, body, reason"
            " FROM versions WHERE memoria_id = ? AND version_id = ?",
            (memoria_id, version_id),
        ).fetchone()

        if not row:
            return None

        return Version(
            version_id=row["version_id"],
            memoria_id=row["memoria_id"],
            timestamp=row["timestamp"],
            title=row["title"],
            type=row["type"],
            tags=json.loads(row["tags"]),
            body=row["body"],
            reason=row["reason"],
        )

    def delete_versions(self, memoria_id: str) -> None:
        """Delete all versions for a memoria."""
        conn = self._get_conn()
        conn.execute("DELETE FROM versions WHERE memoria_id = ?", (memoria_id,))
        conn.commit()


class VersionManager:
    """Manages versioning for memorias.

    Args:
        memory: The Memory instance to operate on.
    """

    def __init__(self, memory: Any) -> None:
        self.memory = memory
        self.version_store = VersionStore(memory.cfg.state_dir / "versions.db")

    def track_update(
        self,
        memoria_id: str,
        title: str,
        type: str,
        tags: list[str],
        body: str,
        reason: str | None = None,
    ) -> int:
        """Track a memoria update by saving a new version."""
        return self.version_store.save_version(memoria_id, title, type, tags, body, reason)

    def diff_versions(
        self,
        memoria_id: str,
        version_a: int | None = None,
        version_b: int | None = None,
    ) -> DiffResult | None:
        """Generate a diff between two versions of a memoria."""
        versions = self.version_store.get_versions(memoria_id, limit=10)

        if not versions:
            return None

        if version_a is None:
            v_a = versions[0] if len(versions) > 0 else None
        else:
            v_a = next((v for v in versions if v.version_id == version_a), None)

        if version_b is None:
            v_b = versions[1] if len(versions) > 1 else None
        else:
            v_b = next((v for v in versions if v.version_id == version_b), None)

        if not v_a or not v_b:
            return None

        body_a = v_a.body.splitlines(keepends=True)
        body_b = v_b.body.splitlines(keepends=True)

        diff = difflib.unified_diff(
            body_a,
            body_b,
            fromfile=f"v{v_a.version_id}",
            tofile=f"v{v_b.version_id}",
            lineterm="",
        )

        unified_diff = "".join(diff)
        changes = [
            line
            for line in unified_diff.split("\n")
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        ]

        return DiffResult(
            memoria_id=memoria_id,
            version_a=v_a.version_id,
            version_b=v_b.version_id,
            unified_diff=unified_diff,
            changes=changes,
        )

    def rollback_to_version(
        self,
        memoria_id: str,
        version_id: int,
        reason: str | None = None,
    ) -> bool:
        """Rollback a memoria to a previous version."""
        version = self.version_store.get_version(memoria_id, version_id)
        if not version:
            return False

        self.memory.update(
            memoria_id,
            title=version.title,
            content=version.body,
            type_=version.type,
            tags=version.tags,
        )

        return True

    def get_version_history(self, memoria_id: str, limit: int = 10) -> list[Version]:
        """Get the version history for a memoria."""
        return self.version_store.get_versions(memoria_id, limit=limit)


__all__ = [
    "DiffResult",
    "Version",
    "VersionManager",
    "VersionStore",
]
