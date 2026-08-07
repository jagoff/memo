"""Nightly vector-index hygiene and derived-storage maintenance.

The Markdown vault and user feedback are primary data. Everything handled here
is rebuildable: stale repository embedding-cache identities and sqlite-vec
chunk allocation for source feedback. The pass is deliberately separate from
the recall hot path and is dry-run capable.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# Fraction of a DB file that must be free pages before VACUUM is worth its
# exclusive lock and full rewrite. Compaction returns pages to sqlite's freelist,
# never to the filesystem, so a store can be logically tidy and still occupy the
# disk it grew into: `graph.db` measured 52% free (73 MB of 140 MB) after months
# of nightly graph rebuilds, while `memvec.db` sat at 0.2% and would gain
# nothing. The ratio is the signal, which is why this needs no flag of its own.
_VACUUM_FREELIST_RATIO = 0.25
_VACUUM_MIN_FREE_BYTES = 32 * 1024 * 1024


def _reclaimable(path: Path) -> int:
    """Bytes VACUUM would return for `path`, or 0 when it is not worth it."""
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
    except sqlite3.Error:
        return 0
    try:
        free = int(conn.execute("PRAGMA freelist_count").fetchone()[0])
        pages = int(conn.execute("PRAGMA page_count").fetchone()[0])
        page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
    except sqlite3.Error:
        return 0
    finally:
        conn.close()
    if pages <= 0 or free / pages < _VACUUM_FREELIST_RATIO:
        return 0
    freed = free * page_size
    return freed if freed >= _VACUUM_MIN_FREE_BYTES else 0


def _vacuum_bloated_sidecars(cfg: Any) -> dict[str, int]:
    """Truncate the WAL of every managed sqlite file, then VACUUM the ones whose
    freelist has outgrown the ratio.

    Covers the sidecars (`graph.db`, `episodes.db`, …) as well as the main
    index: compaction elsewhere in this pass only ever touches `memvec.db`, so
    without this the largest single reclaim on a mature install is missed. The
    WAL half matters just as much — measured on a live install, `memvec.db-wal`
    held 257 MB against a 235 MB database, because only the main store was ever
    checkpointed and nothing ever truncated.
    """
    reclaimed: dict[str, int] = {}
    state_dir = Path(getattr(cfg, "state_dir", "") or "")
    if not state_dir.is_dir():
        return reclaimed
    for path in sorted(state_dir.glob("*.db")):
        wal_before = _wal_bytes(path)
        # Checkpoint first: pages still in the WAL are invisible to
        # freelist_count, so measuring before this under-reports what VACUUM
        # would reclaim.
        _checkpoint_truncate(path)
        freed = _reclaimable(path)
        if freed:
            try:
                conn = sqlite3.connect(str(path), timeout=30.0)
                try:
                    conn.execute("VACUUM")
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                # A live reader (the recall daemon) can hold the lock. Skipping
                # is correct: the space is still there tomorrow night.
                _log.info("vacuum skipped for %s: %s", path.name, exc)
                freed = 0
            else:
                # VACUUM rewrites through the WAL, so truncate again after.
                _checkpoint_truncate(path)
        total = freed + max(0, wal_before - _wal_bytes(path))
        if total:
            reclaimed[path.name] = total
    return reclaimed


def _wal_bytes(path: Path) -> int:
    wal = path.with_name(path.name + "-wal")
    try:
        return wal.stat().st_size
    except OSError:
        return 0


def _checkpoint_truncate(path: Path) -> None:
    """Fold the WAL back into the DB and shrink the -wal file to zero.

    Best-effort: a long-lived reader pinning the snapshot makes this a no-op,
    which is the correct outcome rather than an error.
    """
    try:
        conn = sqlite3.connect(str(path), timeout=10.0)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()
    except sqlite3.Error as exc:
        _log.debug("wal checkpoint skipped for %s: %s", path.name, exc)


def run_vector_hygiene(
    cfg: Any,
    mem: Any,
    *,
    dry_run: bool = False,
    vacuum: bool = False,
) -> dict[str, Any]:
    """Prune stale embedding-cache identities and compact feedback vectors."""
    result: dict[str, Any] = {
        "status": "skipped",
        "cache_packed": 0,
        "cache_pruned": 0,
        "feedback": {"before": 0, "after": 0, "rebuilt": False},
        "vacuumed": False,
        "reclaimed": {},
    }
    try:
        model = str(getattr(mem.store, "embedder_model", "") or "").strip()
        dims = int(getattr(cfg, "embedder_dims", 0) or 0)
        if model and dims > 0:
            result["cache_packed"] = mem.store.compact_repo_embedding_cache(dry_run=dry_run)
            result["cache_pruned"] = mem.store.prune_repo_embedding_cache(
                keep_models={(model, dims)},
                dry_run=dry_run,
            )
        else:
            result["cache_packed"] = 0
            result["cache_pruned"] = 0

        result["feedback"] = mem.store.compact_feedback_vectors(dry_run=dry_run)
        # HyPE storage is deliberately NOT reclaimed here. Its 110 MB looks like
        # dead weight while MEMO_HYPE_ENABLED is off, but that flag gates the
        # READ fold only: MEMO_DREAM_HYPE_ENABLED "builds dark; read fold is
        # gated separately", so the index is meant to accumulate before anything
        # reads it. Purging on the read flag would delete each night what the
        # build pass spent LLM calls producing, and the A/B that decides whether
        # the read flag graduates would only ever measure an empty index.
        if not dry_run:
            # Checkpoint is best-effort; dream's receipt should not become an
            # error merely because a long-lived reader pins the WAL snapshot.
            mem.store._checkpoint()
            if vacuum:
                mem.store._conn.execute("VACUUM")
                result["vacuumed"] = True
            result["reclaimed"] = _vacuum_bloated_sidecars(cfg)
        result["status"] = "dry_run" if dry_run else "done"
    except Exception as exc:  # surfaced in the dream receipt
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


__all__ = ["run_vector_hygiene"]
