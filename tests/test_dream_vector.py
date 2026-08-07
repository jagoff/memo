"""Tests for `run_vector_hygiene` — nightly vector-index hygiene pass."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from memo.dream_vector import run_vector_hygiene


def _cfg(embedder_dims: int = 2560) -> SimpleNamespace:
    return SimpleNamespace(embedder_dims=embedder_dims)


def _mem(embedder_model: str = "Qwen3-Embedding-4B") -> MagicMock:
    mem = MagicMock()
    mem.store.embedder_model = embedder_model
    mem.store.compact_repo_embedding_cache.return_value = 3
    mem.store.prune_repo_embedding_cache.return_value = 2
    mem.store.compact_feedback_vectors.return_value = {
        "before": 10,
        "after": 8,
        "rebuilt": True,
    }
    return mem


def test_dry_run_packs_and_prunes_cache_without_checkpoint_or_vacuum():
    mem = _mem()

    result = run_vector_hygiene(_cfg(), mem, dry_run=True, vacuum=True)

    mem.store.compact_repo_embedding_cache.assert_called_once_with(dry_run=True)
    mem.store.prune_repo_embedding_cache.assert_called_once_with(
        keep_models={("Qwen3-Embedding-4B", 2560)}, dry_run=True
    )
    mem.store._checkpoint.assert_not_called()
    mem.store._conn.execute.assert_not_called()
    assert result["status"] == "dry_run"
    assert result["cache_packed"] == 3
    assert result["cache_pruned"] == 2
    assert result["feedback"] == {"before": 10, "after": 8, "rebuilt": True}
    assert result["vacuumed"] is False


def test_real_run_without_vacuum_checkpoints_but_skips_vacuum():
    mem = _mem()

    result = run_vector_hygiene(_cfg(), mem, dry_run=False, vacuum=False)

    mem.store._checkpoint.assert_called_once_with()
    mem.store._conn.execute.assert_not_called()
    assert result["status"] == "done"
    assert result["vacuumed"] is False


def test_real_run_with_vacuum_executes_vacuum_after_checkpoint():
    mem = _mem()

    result = run_vector_hygiene(_cfg(), mem, dry_run=False, vacuum=True)

    mem.store._checkpoint.assert_called_once_with()
    mem.store._conn.execute.assert_called_once_with("VACUUM")
    assert result["status"] == "done"
    assert result["vacuumed"] is True


def test_missing_embedder_model_skips_cache_pack_and_prune():
    mem = _mem(embedder_model="")

    result = run_vector_hygiene(_cfg(), mem, dry_run=False)

    mem.store.compact_repo_embedding_cache.assert_not_called()
    mem.store.prune_repo_embedding_cache.assert_not_called()
    assert result["cache_packed"] == 0
    assert result["cache_pruned"] == 0
    # Feedback compaction and checkpoint still run regardless of cache state.
    mem.store.compact_feedback_vectors.assert_called_once_with(dry_run=False)
    mem.store._checkpoint.assert_called_once_with()


def test_zero_embedder_dims_skips_cache_pack_and_prune():
    mem = _mem()

    result = run_vector_hygiene(_cfg(embedder_dims=0), mem, dry_run=False)

    mem.store.compact_repo_embedding_cache.assert_not_called()
    mem.store.prune_repo_embedding_cache.assert_not_called()
    assert result["cache_packed"] == 0
    assert result["cache_pruned"] == 0


def test_store_exception_is_captured_in_receipt_not_raised():
    mem = _mem()
    mem.store.compact_repo_embedding_cache.side_effect = RuntimeError("boom")

    result = run_vector_hygiene(_cfg(), mem, dry_run=False)

    assert result["status"] == "error"
    assert result["error"] == "RuntimeError: boom"


# --- VACUUM sweep -------------------------------------------------------------


def _make_db(path, *, rows: int, delete: bool) -> None:
    import sqlite3

    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, blob BLOB)")
    conn.executemany("INSERT INTO t (blob) VALUES (?)", [(b"x" * 4000,) for _ in range(rows)])
    if delete:
        conn.execute("DELETE FROM t WHERE id > 20")
    conn.commit()
    conn.close()


def test_reclaimable_ignores_a_tidy_database(tmp_path):
    """Compaction returns pages to sqlite's freelist, not to the filesystem, so
    the freelist ratio is the signal for whether VACUUM earns its lock."""
    from memo.dream_vector import _reclaimable

    db = tmp_path / "tidy.db"
    _make_db(db, rows=20000, delete=False)

    assert _reclaimable(db) == 0


def test_reclaimable_reports_a_bloated_database(tmp_path):
    from memo.dream_vector import _reclaimable

    db = tmp_path / "bloated.db"
    _make_db(db, rows=20000, delete=True)

    assert _reclaimable(db) > 0


def test_vacuum_sweep_shrinks_bloated_sidecars_and_leaves_tidy_ones(tmp_path):
    """`graph.db` measured 52% free (73 MB) while the pass only ever vacuumed
    `memvec.db` — the largest single reclaim on a mature install was missed."""
    from types import SimpleNamespace

    from memo.dream_vector import _vacuum_bloated_sidecars

    bloated = tmp_path / "graph.db"
    tidy = tmp_path / "tidy.db"
    _make_db(bloated, rows=20000, delete=True)
    _make_db(tidy, rows=20000, delete=False)
    before = bloated.stat().st_size

    reclaimed = _vacuum_bloated_sidecars(SimpleNamespace(state_dir=tmp_path))

    assert set(reclaimed) == {"graph.db"}
    assert bloated.stat().st_size < before


def test_vacuum_sweep_on_a_missing_state_dir_is_a_noop(tmp_path):
    from types import SimpleNamespace

    from memo.dream_vector import _vacuum_bloated_sidecars

    assert _vacuum_bloated_sidecars(SimpleNamespace(state_dir=tmp_path / "nope")) == {}


def test_vacuum_sweep_skips_a_file_that_is_not_a_database(tmp_path):
    """A stray file must not fail the pass."""
    from types import SimpleNamespace

    from memo.dream_vector import _vacuum_bloated_sidecars

    (tmp_path / "junk.db").write_text("not sqlite", encoding="utf-8")

    assert _vacuum_bloated_sidecars(SimpleNamespace(state_dir=tmp_path)) == {}


def test_vacuum_sweep_truncates_the_wal_even_when_the_db_is_tidy(tmp_path):
    """Measured on a live install: `memvec.db-wal` held 257 MB against a 235 MB
    database. Only the main store was ever checkpointed, and nothing truncated,
    so a tidy database still cost twice its size on disk."""
    import sqlite3
    from types import SimpleNamespace

    from memo.dream_vector import _vacuum_bloated_sidecars

    db = tmp_path / "walheavy.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, blob BLOB)")
    conn.executemany("INSERT INTO t (blob) VALUES (?)", [(b"x" * 4000,) for _ in range(5000)])
    conn.commit()
    wal = tmp_path / "walheavy.db-wal"
    assert wal.stat().st_size > 0, "fixture must leave a populated WAL"
    before = wal.stat().st_size

    reclaimed = _vacuum_bloated_sidecars(SimpleNamespace(state_dir=tmp_path))
    conn.close()

    # With no other connection holding the snapshot, sqlite removes the -wal
    # outright rather than truncating it in place.
    after = wal.stat().st_size if wal.exists() else 0
    assert after < before
    assert reclaimed.get("walheavy.db", 0) > 0
