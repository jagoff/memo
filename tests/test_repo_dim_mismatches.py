from __future__ import annotations

from pathlib import Path

import pytest

from memo.store.store import VecStore


def test_upsert_repo_embeddings_dim_mismatch(tmp_path: Path):
    db_path = tmp_path / "test.db"
    store = VecStore(db_path, dims=8)

    with pytest.raises(ValueError) as excinfo:
        store.upsert_repo_embeddings(
            repo_id="repo1",
            embeddings=[("chunk1", [1.0] * 4)],  # Wrong dims: 4 instead of 8
        )

    error_msg = str(excinfo.value)
    # Check for detailed message
    assert "Repo chunk embedding dim mismatch: got 4, expected 8" in error_msg
    assert "Fix: rm" in error_msg
    assert "memo reindex" in error_msg


def test_search_repo_vec_dim_mismatch(tmp_path: Path):
    db_path = tmp_path / "test.db"
    store = VecStore(db_path, dims=8)

    with pytest.raises(ValueError) as excinfo:
        store.search_repo_vec(
            embedding=[1.0] * 4,  # Wrong dims: 4 instead of 8
        )

    error_msg = str(excinfo.value)
    assert "Repo query embedding dim mismatch: got 4, expected 8" in error_msg
    assert "Fix: rm" in error_msg
    assert "memo reindex" in error_msg
