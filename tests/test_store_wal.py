"""`memo ops checkpoint-wal` — the only thing that actually reclaims a WAL."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from memo.store.wal import checkpoint_wal

pytestmark = pytest.mark.resource_hygiene


def _db_with_wal(path: Path, rows: int) -> sqlite3.Connection:
    """Seed a WAL database and RETURN the open connection.

    Closing the last connection checkpoints and deletes the -wal, so a test
    that wants a WAL on disk has to keep one open — which is also the real
    situation this command exists for: memo's daemons never close theirs.
    """
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, blob TEXT)")
    con.executemany("INSERT INTO t (blob) VALUES (?)", [("x" * 4096,) for _ in range(rows)])
    con.commit()
    return con


def test_checkpoint_truncates_an_oversized_wal(tmp_path: Path) -> None:
    db = tmp_path / "big.db"
    con = _db_with_wal(db, 800)
    try:
        wal = db.with_name("big.db-wal")
        before = wal.stat().st_size
        assert before > 1_000_000

        result = checkpoint_wal(tmp_path, min_bytes=1_000_000)

        assert [entry["db"] for entry in result["checkpointed"]] == ["big.db"]
        assert result["freed_mb"] > 0
        after = wal.stat().st_size if wal.exists() else 0
        assert after < before
    finally:
        con.close()


def test_checkpoint_skips_small_wals(tmp_path: Path) -> None:
    """The common case must cost one stat per file, not a connection."""
    db = tmp_path / "small.db"
    con = _db_with_wal(db, 2)
    try:
        result = checkpoint_wal(tmp_path, min_bytes=100_000_000)
    finally:
        con.close()

    assert result["checkpointed"] == []
    assert result["freed_mb"] == 0


def test_checkpoint_ignores_a_directory_without_databases(tmp_path: Path) -> None:
    assert checkpoint_wal(tmp_path, min_bytes=1)["checkpointed"] == []
