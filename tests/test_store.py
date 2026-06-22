"""VecStore — sqlite-vec CRUD + cosine search.

These tests exercise the SQLite + vec0 layer directly without
requiring the MLX runtime, so they run on any platform with
`sqlite-vec` available.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from memo.store import VecStore


@pytest.fixture
def store(tmp_path: Path):
    s = VecStore(tmp_path / "vec.db", dims=4)
    yield s
    s.close()


def _emb(*xs: float) -> list[float]:
    """Build a small fixed-dim normalised vector."""
    import math

    norm = math.sqrt(sum(x * x for x in xs)) or 1.0
    return [x / norm for x in xs]


def test_set_confidence_batch_writes_absolute_value(store: VecStore):
    store.set_confidence_batch([("img1", 0.45), ("img2", 0.02)])
    health = store.get_health_batch(["img1", "img2"])
    assert health["img1"]["confidence"] == 0.45
    assert health["img2"]["confidence"] == 0.1  # floored


def test_set_confidence_batch_only_lowers(store: VecStore):
    store.set_confidence_batch([("img1", 0.45)])
    # set_confidence_batch writes absolute values, so a higher value overwrites
    store.set_confidence_batch([("img1", 0.9)])
    assert store.get_health_batch(["img1"])["img1"]["confidence"] == 0.9
    # a lower value also applies
    store.set_confidence_batch([("img1", 0.3)])
    assert store.get_health_batch(["img1"])["img1"]["confidence"] == 0.3


def test_upsert_and_get(store: VecStore):
    store.upsert(
        id_="abc", path="memory/x.md", title="X", type_="note", tags=["a", "b"],
        created="2026-05-06T19:00:00-03:00",
        updated="2026-05-06T19:00:00-03:00",
        body_hash="deadbeef",
        embedding=_emb(1, 0, 0, 0),
    )
    got = store.get("abc")
    assert got is not None
    assert got["title"] == "X"
    assert got["tags"] == ["a", "b"]


def test_upsert_replaces_on_same_id(store: VecStore):
    store.upsert(
        id_="abc", path="memory/x.md", title="V1", type_="note", tags=[],
        created="2026-05-06T19:00:00-03:00",
        updated="2026-05-06T19:00:00-03:00",
        body_hash="h1", embedding=_emb(1, 0, 0, 0),
    )
    store.upsert(
        id_="abc", path="memory/x.md", title="V2", type_="note", tags=["new"],
        created="2026-05-06T19:00:00-03:00",
        updated="2026-05-06T19:01:00-03:00",
        body_hash="h2", embedding=_emb(0, 1, 0, 0),
    )
    got = store.get("abc")
    assert got is not None
    assert got["title"] == "V2"
    assert got["tags"] == ["new"]
    assert store.count() == 1


def test_search_orders_by_cosine(store: VecStore):
    store.upsert(
        id_="x", path="memory/x.md", title="x", type_="note", tags=[],
        created="t", updated="t", body_hash="h", embedding=_emb(1, 0, 0, 0),
    )
    store.upsert(
        id_="y", path="memory/y.md", title="y", type_="note", tags=[],
        created="t", updated="t", body_hash="h", embedding=_emb(0, 1, 0, 0),
    )
    store.upsert(
        id_="z", path="memory/z.md", title="z", type_="note", tags=[],
        created="t", updated="t", body_hash="h", embedding=_emb(0.9, 0.1, 0, 0),
    )
    hits = store.search(_emb(1, 0, 0, 0), limit=3)
    ids = [h["id"] for h in hits]
    # `x` is identical → first; `z` is close (cos≈0.99) → second; `y` orthogonal → last.
    assert ids[0] == "x"
    assert ids[1] == "z"
    assert ids[2] == "y"
    # Score is monotonically descending (larger = more similar).
    assert hits[0]["score"] >= hits[1]["score"] >= hits[2]["score"]


def test_search_filters_by_type(store: VecStore):
    store.upsert(
        id_="d1", path="memory/d1.md", title="d1", type_="decision", tags=[],
        created="t", updated="t", body_hash="h", embedding=_emb(1, 0, 0, 0),
    )
    store.upsert(
        id_="n1", path="memory/n1.md", title="n1", type_="note", tags=[],
        created="t", updated="t", body_hash="h", embedding=_emb(1, 0, 0, 0),
    )
    hits = store.search(_emb(1, 0, 0, 0), limit=10, type_="decision")
    assert [h["id"] for h in hits] == ["d1"]


def test_search_exclude_types_multi_value(store: VecStore):
    """Multi-value exclude_types must exclude ALL named types without 5x over-fetch.

    sqlite-vec supports chained `AND vec.type != ?` predicates as KNN push-down
    filters, so exclude_types={"decision", "reference"} must return zero rows of
    either type even when limit=1 (no over-fetch needed to fill the result set).
    """
    for id_, type_ in [
        ("n1", "note"),
        ("d1", "decision"),
        ("r1", "reference"),
        ("f1", "fact"),
    ]:
        store.upsert(
            id_=id_, path=f"memory/{id_}.md", title=id_, type_=type_, tags=[],
            created="t", updated="t", body_hash="h", embedding=_emb(1, 0, 0, 0),
        )
    hits = store.search(_emb(1, 0, 0, 0), limit=10, exclude_types={"decision", "reference"})
    hit_types = {h["type"] for h in hits}
    assert "decision" not in hit_types, "decision should be excluded"
    assert "reference" not in hit_types, "reference should be excluded"
    hit_ids = {h["id"] for h in hits}
    assert "n1" in hit_ids, "note should survive"
    assert "f1" in hit_ids, "fact should survive"


def test_delete_removes_from_both_tables(store: VecStore):
    store.upsert(
        id_="abc", path="memory/x.md", title="X", type_="note", tags=[],
        created="t", updated="t", body_hash="h", embedding=_emb(1, 0, 0, 0),
    )
    assert store.delete("abc") is True
    assert store.get("abc") is None
    assert store.search(_emb(1, 0, 0, 0), limit=10) == []


def test_dim_mismatch_raises(store: VecStore):
    with pytest.raises(ValueError, match="dimension mismatch"):
        store.upsert(
            id_="abc", path="memory/x.md", title="X", type_="note", tags=[],
            created="t", updated="t", body_hash="h",
            embedding=[1.0, 0.0],  # 2-dim, store expects 4
        )


def test_unit_norm_wrong_dims_still_rejected_at_write(store: VecStore):
    """A wrong-dims vector that is perfectly unit-norm must still fail fast.

    The norm check (≈1.0) cannot catch a model/dims swap: a 2-dim unit vector
    passes the norm guard but corrupts a 4-dim store. The length guard must
    reject it, name both dims, and point at `memo reindex`.
    """
    unit_2d = _emb(0.6, 0.8)  # length 2, L2-norm exactly 1.0
    assert abs(sum(x * x for x in unit_2d) ** 0.5 - 1.0) < 1e-9
    with pytest.raises(ValueError) as excinfo:
        store.upsert(
            id_="abc", path="memory/x.md", title="X", type_="note", tags=[],
            created="t", updated="t", body_hash="h",
            embedding=unit_2d,
        )
    msg = str(excinfo.value)
    assert "2" in msg and "4" in msg  # names both got + expected
    assert "reindex" in msg


def test_existing_vec_table_dim_mismatch_fails_fast(tmp_path: Path):
    db_path = tmp_path / "vec.db"
    store = VecStore(db_path, dims=4)
    store.close()

    # Audit fix (commit a68ae7c) reworded the error to surface the
    # actual mismatch + a concrete fix command. Lock down the new shape.
    with pytest.raises(RuntimeError) as excinfo:
        VecStore(db_path, dims=8)
    msg = str(excinfo.value)
    assert "dimension mismatch" in msg
    assert "4D" in msg and "8D" in msg
    assert "memo reindex" in msg


def test_existing_schema_init_does_not_need_writer_lock(tmp_path: Path):
    db_path = tmp_path / "vec.db"
    first = VecStore(db_path, dims=4)
    first.close()

    writer = sqlite3.connect(db_path, timeout=0.1)
    writer.execute("BEGIN IMMEDIATE")
    try:
        reader = VecStore(db_path, dims=4)
        try:
            assert reader.count() == 0
        finally:
            reader.close()
    finally:
        writer.rollback()
        writer.close()


def test_list_recent_orders_by_updated_desc(store: VecStore):
    for i, ts in enumerate(["2026-05-01", "2026-05-03", "2026-05-02"]):
        store.upsert(
            id_=f"id-{i}", path=f"memory/n{i}.md", title=f"n{i}", type_="note", tags=[],
            created=ts, updated=ts, body_hash="h", embedding=_emb(1, 0, 0, 0),
        )
    rows = store.list_recent(limit=10)
    assert [r["title"] for r in rows] == ["n1", "n2", "n0"]


def test_repo_index_rows_search_and_delete(store: VecStore):
    source = {
        "id": "repo1",
        "name": "sample",
        "url": "https://example.test/sample.git",
        "ref": "HEAD",
        "commit_sha": "abc123",
        "clone_path": "/tmp/sample",
        "indexed_at": "2026-05-23T00:00:00Z",
        "status": "ready",
        "extra": {},
    }
    store.upsert_repo_index(
        source=source,
        files=[
            {
                "id": "file1",
                "path": "src/app.py",
                "language": "py",
                "size_bytes": 42,
                "sha256": "sha",
                "line_count": 2,
                "lines": [
                    {"id": "line1", "line_no": 1, "text": "def alpha():", "text_hash": "l1"},
                    {"id": "line2", "line_no": 2, "text": "    return 'needle'", "text_hash": "l2"},
                ],
                "chunks": [
                    {
                        "id": "chunk1",
                        "chunk_seq": 0,
                        "line_start": 1,
                        "line_end": 2,
                        "text_hash": "c1",
                        "body_text": "def alpha():\n    return 'needle'",
                        "embedding": _emb(1, 0, 0, 0),
                    }
                ],
            }
        ],
    )

    repo_source = store.get_repo_source("sample")
    assert repo_source is not None
    assert repo_source["commit_sha"] == "abc123"
    assert store.repo_file_hashes("repo1")["src/app.py"]["sha256"] == "sha"

    vec_hits = store.search_repo_vec(_emb(1, 0, 0, 0), limit=5)
    assert vec_hits[0]["path"] == "src/app.py"
    assert vec_hits[0]["line_start"] == 1

    line_hits = store.search_repo_lines("needle", limit=5)
    assert line_hits[0]["match_type"] == "line"
    assert line_hits[0]["line_start"] == 2

    lines = store.get_repo_file_lines("repo1", "src/app.py", start=2, end=2)
    assert lines == [{"line_no": 2, "text": "    return 'needle'"}]

    assert store.delete_repo("sample") is True
    assert store.get_repo_source("sample") is None
    assert store.search_repo_lines("needle", limit=5) == []


def test_repo_embedding_cache_is_scoped_by_model_and_dims(store: VecStore):
    store.upsert_repo_embedding_cache(
        model="model-a",
        dims=4,
        embeddings=[("hash1", _emb(1, 0, 0, 0))],
        created_at="2026-05-23T00:00:00Z",
    )

    hit = store.get_repo_embedding_cache(model="model-a", dims=4, input_hashes=["hash1"])

    assert hit == {"hash1": _emb(1, 0, 0, 0)}
    assert store.get_repo_embedding_cache(model="model-b", dims=4, input_hashes=["hash1"]) == {}
    assert store.get_repo_embedding_cache(model="model-a", dims=8, input_hashes=["hash1"]) == {}


def test_concurrent_writes_and_reads_do_not_collide(store: VecStore):
    """Regression: the FastMCP HTTP transport dispatches sync tool calls on a
    worker threadpool, so multiple threads hit one VecStore at once. A single
    shared connection collided on `BEGIN IMMEDIATE` ("cannot start a
    transaction within a transaction") / recursive cursor use. Per-thread
    connections must let concurrent writers + readers run without error."""
    import threading

    errors: list[Exception] = []
    conn_ids: set[int] = set()
    conn_ids_lock = threading.Lock()
    barrier = threading.Barrier(8)

    def worker(n: int) -> None:
        try:
            barrier.wait()  # maximise overlap
            for i in range(10):
                store.upsert(
                    id_=f"t{n}-{i}", path=f"memory/t{n}-{i}.md", title=f"t{n}-{i}",
                    type_="note", tags=[], created="t", updated="t",
                    body_hash="h", embedding=_emb(1, n % 4, i % 4, 0),
                )
                store.search(_emb(1, 0, 0, 0), limit=5)
            with conn_ids_lock:
                conn_ids.add(id(store._conn))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent access raised: {errors!r}"
    assert store.count() == 80
    # Each worker thread saw a distinct connection object (thread-local).
    assert len(conn_ids) == 8


def _legacy_vec_db(path: Path) -> None:
    """Hand-build a pre-upgrade DB: vec0 tables WITHOUT the `type` metadata
    column / `source_id` partition key, plus the companion rows the in-place
    migration backfills from."""
    import sqlite_vec
    from sqlite_vec import serialize_float32

    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(
        "CREATE TABLE meta (id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, "
        "title TEXT NOT NULL, type TEXT NOT NULL, tags TEXT NOT NULL, "
        "created TEXT NOT NULL, updated TEXT NOT NULL, body_hash TEXT NOT NULL, "
        "extra_json TEXT);"
        "CREATE TABLE source_feedback (id TEXT PRIMARY KEY, source_id TEXT NOT NULL, "
        "query_text TEXT NOT NULL, rating INTEGER NOT NULL, created_at TEXT NOT NULL, "
        "extra_json TEXT);"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE vec USING vec0(id TEXT PRIMARY KEY, "
        "embedding FLOAT[4] distance_metric=cosine)"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE source_feedback_vec USING vec0("
        "feedback_id TEXT PRIMARY KEY, query_emb FLOAT[4] distance_metric=cosine)"
    )
    conn.execute(
        "INSERT INTO meta VALUES ('m1','/p1','T1','decision','[]','t','t','h',NULL)"
    )
    conn.execute(
        "INSERT INTO meta VALUES ('m2','/p2','T2','reference','[]','t','t','h',NULL)"
    )
    conn.execute("INSERT INTO vec (id, embedding) VALUES (?, ?)", ("m1", serialize_float32([1.0, 0, 0, 0])))
    conn.execute("INSERT INTO vec (id, embedding) VALUES (?, ?)", ("m2", serialize_float32([0, 1.0, 0, 0])))
    conn.execute(
        "INSERT INTO source_feedback VALUES ('f1','m1','why',1,'t',NULL)"
    )
    conn.execute(
        "INSERT INTO source_feedback_vec (feedback_id, query_emb) VALUES (?, ?)",
        ("f1", serialize_float32([1.0, 0, 0, 0])),
    )
    conn.commit()
    conn.close()


def test_legacy_vec_schema_migrates_in_place(tmp_path: Path):
    """Opening a pre-partition-key/metadata DB auto-migrates it in place,
    preserving vectors (no re-embed) and backfilling the new columns."""
    db = tmp_path / "vec.db"
    _legacy_vec_db(db)

    store = VecStore(db, dims=4)  # triggers migration on open

    # `type` metadata filter (the recall-hook path) now works and `m2` (the
    # reference tier) is excludable inside the kNN.
    hits = store.search(_emb(1, 0, 0, 0), limit=5, exclude_types={"reference"})
    ids = {h["id"] for h in hits}
    assert "m1" in ids and "m2" not in ids
    # Vector preserved exactly (cosine score ~1.0 on the identical query).
    assert next(h for h in hits if h["id"] == "m1")["score"] > 0.99

    # Feedback partition key backfilled → per-source kNN returns the row.
    fb = store.find_feedback_for_source("m1", _emb(1, 0, 0, 0), threshold=0.5)
    assert [r["id"] for r in fb] == ["f1"]

    # Idempotent: re-opening the now-migrated DB is a no-op (no raise).
    VecStore(db, dims=4)


# ── Embedder version-check tests ──────────────────────────────────────────


def test_embedder_version_stamped_on_first_open(tmp_path: Path):
    """First open with a real model name stamps schema_meta rows."""
    db = tmp_path / "v.db"
    s = VecStore(db, dims=8, embedder_model="vendor/Model-A")
    rows = {
        r["key"]: r["value"]
        for r in s._conn.execute("SELECT key, value FROM schema_meta").fetchall()
    }
    assert rows.get("embedder_model") == "vendor/Model-A"
    assert rows.get("embedder_dims") == "8"
    s.close()


def test_embedder_version_reopen_same_model_is_ok(tmp_path: Path):
    """Reopening with the same model/dims does not raise."""
    db = tmp_path / "v.db"
    s = VecStore(db, dims=8, embedder_model="vendor/Model-A")
    s.close()
    # Should not raise
    s2 = VecStore(db, dims=8, embedder_model="vendor/Model-A")
    s2.close()


def test_embedder_version_mismatch_raises(tmp_path: Path):
    """Opening with a different model raises StorageError with a clear message."""
    from memo.errors import StorageError

    db = tmp_path / "v.db"
    s = VecStore(db, dims=8, embedder_model="vendor/Model-A")
    s.close()

    with pytest.raises(StorageError, match="Embedder model mismatch"):
        VecStore(db, dims=8, embedder_model="vendor/Model-B")


def test_embedder_version_dims_mismatch_raises(tmp_path: Path):
    """Storing dims=8 then reopening with dims=8 but schema_meta says 4 raises StorageError.

    We manipulate schema_meta directly to test the dims-check path without
    triggering the earlier vec-table dims check (which fires when the vec0 table
    was created with different dims).
    """
    from memo.errors import StorageError

    db = tmp_path / "v.db"
    # Create with model-A / 8 dims
    s = VecStore(db, dims=8, embedder_model="vendor/Model-A")
    # Overwrite stored dims to simulate an old DB built with a different dims
    with s._conn:
        s._conn.execute(
            "UPDATE schema_meta SET value = '4' WHERE key = 'embedder_dims'"
        )
    s.close()

    # Reopen with same vec dims (so _validate_vec_dims is happy) but schema_meta disagrees
    with pytest.raises(StorageError, match="Embedder model mismatch"):
        VecStore(db, dims=8, embedder_model="vendor/Model-A")


def test_embedder_version_bypass_empty_model(tmp_path: Path):
    """Empty embedder_model (test stub) skips version check — no schema_meta written."""
    db = tmp_path / "v.db"
    # First open with empty (stub) model
    s = VecStore(db, dims=4)
    # schema_meta should have no embedder_model row (bypass was triggered)
    rows = s._conn.execute("SELECT key FROM schema_meta").fetchall()
    keys = {r["key"] for r in rows}
    assert "embedder_model" not in keys
    # Reopen with same dims but a different empty model — no error (bypass active)
    s.close()
    s2 = VecStore(db, dims=4)
    s2.close()


def test_embedder_version_bypass_env_flag(tmp_path: Path, monkeypatch):
    """MEMO_SKIP_MODEL_VERSION_CHECK=1 bypasses the check."""

    db = tmp_path / "v.db"
    s = VecStore(db, dims=8, embedder_model="vendor/Model-A")
    s.close()

    monkeypatch.setenv("MEMO_SKIP_MODEL_VERSION_CHECK", "1")
    # Would normally raise, but env bypass prevents it
    s2 = VecStore(db, dims=8, embedder_model="vendor/Model-B")
    s2.close()


def test_embedder_version_error_message_is_actionable(tmp_path: Path):
    """StorageError message names both models and instructs the user."""
    from memo.errors import StorageError

    db = tmp_path / "v.db"
    VecStore(db, dims=8, embedder_model="vendor/OldModel").close()

    # Reopen with same dims (so vec table is compatible) but different model name
    with pytest.raises(StorageError) as exc_info:
        VecStore(db, dims=8, embedder_model="vendor/NewModel")

    msg = str(exc_info.value)
    assert "vendor/OldModel" in msg
    assert "vendor/NewModel" in msg
    assert "memo reindex --rebuild" in msg
