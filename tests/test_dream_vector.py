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
