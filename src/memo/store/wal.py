"""WAL maintenance for memo's sqlite stores.

A write-ahead log only shrinks at a checkpoint, and a checkpoint cannot advance
past the oldest open reader. memo keeps several long-lived ones — the recall
daemon, the watcher, every `memo-mcp` session — so the passive checkpoints
sqlite runs on its own never truncate, and `PRAGMA journal_size_limit` (set by
each store) caps the file only AFTER one succeeds. Something has to run the
checkpoint explicitly; the nightly pass does, through `memo ops checkpoint-wal`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

_BUSY_TIMEOUT_MS = 15_000
_DEFAULT_JOURNAL_SIZE_LIMIT = 16 * 1024 * 1024


def checkpoint_wal(state_dir: Path, *, min_bytes: int = 8 * 1024 * 1024) -> dict[str, Any]:
    """TRUNCATE-checkpoint every `*.db` in `state_dir` with an oversized WAL.

    Only databases whose `-wal` exceeds `min_bytes` are touched, so the common
    case costs one `stat` per file. Returns a per-database report plus the total
    megabytes reclaimed; a database whose readers never released is reported
    with `busy: True` rather than raising.
    """
    checkpointed: list[dict[str, Any]] = []
    freed = 0
    for db_path in sorted(Path(state_dir).glob("*.db")):
        wal_path = db_path.with_name(db_path.name + "-wal")
        try:
            before = wal_path.stat().st_size
        except OSError:
            continue
        if before < min_bytes:
            continue
        busy = False
        try:
            con = sqlite3.connect(str(db_path), timeout=_BUSY_TIMEOUT_MS / 1000)
        except sqlite3.Error:
            continue
        try:
            con.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            con.execute(f"PRAGMA journal_size_limit={_DEFAULT_JOURNAL_SIZE_LIMIT}")
            row = con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            # Row 0 is sqlite's own busy flag: 1 means readers held it open and
            # the log was NOT fully reclaimed.
            busy = bool(row and row[0])
        except sqlite3.Error:
            busy = True
        finally:
            con.close()
        after = wal_path.stat().st_size if wal_path.exists() else 0
        freed += max(0, before - after)
        checkpointed.append(
            {
                "db": db_path.name,
                "before_mb": before / 1048576,
                "after_mb": after / 1048576,
                "busy": busy,
            }
        )
    return {"checkpointed": checkpointed, "freed_mb": freed / 1048576}
