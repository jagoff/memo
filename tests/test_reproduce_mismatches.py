from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from memo.errors import StorageError
from memo.store.store import VecStore


def test_validate_vec_dims_error_message(tmp_path: Path):
    db_path = tmp_path / "test.db"
    # Create a DB with 4D vectors
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("CREATE TABLE vec (id TEXT PRIMARY KEY, embedding FLOAT[4])")
        conn.commit()
        # Add other required tables to skip full initialization if needed,
        # but VecStore will try to create them anyway if they don't exist.

    # Configure VecStore with 8D vectors
    # We expect RuntimeError (from _validate_vec_dims)
    with pytest.raises(RuntimeError) as excinfo:
        VecStore(db_path, dims=8)

    error_msg = str(excinfo.value)
    assert "Embedding dimension mismatch: store has 4D vectors but config expects 8D." in error_msg
    assert "Fix: Run 'memo reindex --rebuild'" in error_msg
    assert "memo reindex" in error_msg
    assert "MEMO_MODEL_PROFILE (current: 8D)" in error_msg


def test_check_embedder_version_mismatch(tmp_path: Path):
    db_path = tmp_path / "test_version.db"

    # 1. Initialize DB with model A
    base_store = VecStore(db_path, dims=8)
    try:
        base_store.embedder_model = "model-A"
        base_store._init_schema()  # This will stamp model-A in schema_meta
    finally:
        base_store.close()

    # 2. Try to open with model B
    # We need to bypass the __init__ call's _init_schema or just let it fail.
    # Actually, VecStore(db_path, dims=8) will call _init_schema() in __init__.

    # Let's manually set up the schema_meta
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)", ("embedder_model", "model-A"))
        conn.execute("INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)", ("embedder_dims", "8"))
        conn.commit()

    # Now open with current_model = "model-B"
    # We can't easily inject embedder_model into VecStore because it's not an __init__ arg,
    # it's usually set by the Memory facade.
    # But _check_embedder_version uses getattr(self, "embedder_model", "")

    class MockVecStore(VecStore):
        def __init__(self, db_path, dims, model):
            self.db_path = db_path
            self.dims = dims
            self.embedder_model = model
            self._conn_obj = sqlite3.connect(db_path)
            self._conn_obj.row_factory = sqlite3.Row
            # We don't call super().__init__ because it calls _init_schema

        @property
        def _conn(self):
            return self._conn_obj

        def close(self):
            self._conn_obj.close()

    store = MockVecStore(db_path, dims=8, model="model-B")

    try:
        with pytest.raises(StorageError) as excinfo:
            store._check_embedder_version()
    finally:
        store.close()

    error_msg = str(excinfo.value)
    assert "Embedder model mismatch: index was built with model-A (8d) but current config is model-B (8d)" in error_msg
    assert "Run 'memo reindex --rebuild'" in error_msg
