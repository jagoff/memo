"""VecStore — sqlite-vec CRUD + cosine search.

These tests exercise the SQLite + vec0 layer directly without
requiring the MLX runtime, so they run on any platform with
`sqlite-vec` available.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

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


def _put_for_delete(store: VecStore, id_: str) -> None:
    store.upsert(
        id_=id_,
        path=f"memory/{id_}.md",
        title=f"Memory {id_}",
        type_="note",
        tags=["delete-contract"],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00",
        body_hash=f"hash-{id_}",
        embedding=_emb(1, 0, 0, 0),
    )


def test_meta_has_valid_time_columns(store: VecStore) -> None:
    cols = {row["name"] for row in store._conn.execute("PRAGMA table_info(meta)").fetchall()}
    assert "valid_at" in cols
    assert "invalid_at" in cols


def test_meta_has_invalid_at_partial_index(store: VecStore) -> None:
    idx = {row["name"] for row in store._conn.execute("PRAGMA index_list(meta)").fetchall()}
    assert "idx_meta_invalid_at" in idx


def test_delete_tantivy_document_commits_sidecar_delete(store: VecStore, monkeypatch) -> None:
    tantivy = MagicMock()
    monkeypatch.setattr(store, "_get_tantivy", MagicMock(return_value=tantivy))

    assert store._delete_tantivy_document("memory-1") is True

    tantivy.delete_document.assert_called_once_with("memory-1")
    tantivy.commit.assert_called_once_with()


def test_hard_delete_marks_tantivy_unhealthy_when_sidecar_delete_fails(
    store: VecStore, monkeypatch
) -> None:
    _put_for_delete(store, "memory-1")
    delete_sidecar = MagicMock(return_value=False)
    mark_unhealthy = MagicMock()
    monkeypatch.setattr(store, "_delete_tantivy_document", delete_sidecar)
    monkeypatch.setattr(store, "_mark_tantivy_unhealthy", mark_unhealthy)

    assert store.hard_delete("memory-1") is True

    assert store.get("memory-1") is None
    delete_sidecar.assert_called_once_with("memory-1")
    mark_unhealthy.assert_called_once_with()


def test_conditional_hard_delete_marks_tantivy_unhealthy_after_eligible_tombstone(
    store: VecStore, monkeypatch
) -> None:
    _put_for_delete(store, "memory-1")
    store.connection.execute(
        "UPDATE meta SET deleted_at = ? WHERE id = ?",
        ("2026-01-01T00:00:00+00:00", "memory-1"),
    )
    store.connection.commit()
    delete_sidecar = MagicMock(return_value=False)
    mark_unhealthy = MagicMock()
    monkeypatch.setattr(store, "_delete_tantivy_document", delete_sidecar)
    monkeypatch.setattr(store, "_mark_tantivy_unhealthy", mark_unhealthy)

    assert (
        store.hard_delete_if_soft_deleted_before("memory-1", before="2026-02-01T00:00:00+00:00")
        is True
    )

    assert store.list_soft_deleted() == []
    delete_sidecar.assert_called_once_with("memory-1")
    mark_unhealthy.assert_called_once_with()


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
        id_="abc",
        path="memory/x.md",
        title="X",
        type_="note",
        tags=["a", "b"],
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
        id_="abc",
        path="memory/x.md",
        title="V1",
        type_="note",
        tags=[],
        created="2026-05-06T19:00:00-03:00",
        updated="2026-05-06T19:00:00-03:00",
        body_hash="h1",
        embedding=_emb(1, 0, 0, 0),
    )
    store.upsert(
        id_="abc",
        path="memory/x.md",
        title="V2",
        type_="note",
        tags=["new"],
        created="2026-05-06T19:00:00-03:00",
        updated="2026-05-06T19:01:00-03:00",
        body_hash="h2",
        embedding=_emb(0, 1, 0, 0),
    )
    got = store.get("abc")
    assert got is not None
    assert got["title"] == "V2"
    assert got["tags"] == ["new"]
    assert store.count() == 1


def test_count_by_type_groups_active_records(store: VecStore):
    for id_, type_ in (("n1", "note"), ("n2", "note"), ("d1", "decision")):
        store.upsert(
            id_=id_,
            path=f"memory/{id_}.md",
            title=id_,
            type_=type_,
            tags=[],
            created="t",
            updated="t",
            body_hash=f"hash-{id_}",
            embedding=_emb(1, 0, 0, 0),
        )

    assert store.count_by_type() == {"decision": 1, "note": 2}


def test_search_orders_by_cosine(store: VecStore):
    store.upsert(
        id_="x",
        path="memory/x.md",
        title="x",
        type_="note",
        tags=[],
        created="t",
        updated="t",
        body_hash="h",
        embedding=_emb(1, 0, 0, 0),
    )
    store.upsert(
        id_="y",
        path="memory/y.md",
        title="y",
        type_="note",
        tags=[],
        created="t",
        updated="t",
        body_hash="h",
        embedding=_emb(0, 1, 0, 0),
    )
    store.upsert(
        id_="z",
        path="memory/z.md",
        title="z",
        type_="note",
        tags=[],
        created="t",
        updated="t",
        body_hash="h",
        embedding=_emb(0.9, 0.1, 0, 0),
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
        id_="d1",
        path="memory/d1.md",
        title="d1",
        type_="decision",
        tags=[],
        created="t",
        updated="t",
        body_hash="h",
        embedding=_emb(1, 0, 0, 0),
    )
    store.upsert(
        id_="n1",
        path="memory/n1.md",
        title="n1",
        type_="note",
        tags=[],
        created="t",
        updated="t",
        body_hash="h",
        embedding=_emb(1, 0, 0, 0),
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
            id_=id_,
            path=f"memory/{id_}.md",
            title=id_,
            type_=type_,
            tags=[],
            created="t",
            updated="t",
            body_hash="h",
            embedding=_emb(1, 0, 0, 0),
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
        id_="abc",
        path="memory/x.md",
        title="X",
        type_="note",
        tags=[],
        created="t",
        updated="t",
        body_hash="h",
        embedding=_emb(1, 0, 0, 0),
    )
    assert store.delete("abc") is True
    assert store.get("abc") is None
    assert store.search(_emb(1, 0, 0, 0), limit=10) == []


def test_find_by_topic_key_returns_active_record(store: VecStore):
    store.upsert(
        id_="abc",
        path="memory/x.md",
        title="X",
        type_="note",
        tags=[],
        created="2026-05-06T19:00:00-03:00",
        updated="2026-05-06T19:00:00-03:00",
        body_hash="h",
        embedding=_emb(1, 0, 0, 0),
        topic_key="project:x",
    )

    row = store.find_by_topic_key("project:x")

    assert row == {
        "id": "abc",
        "path": "memory/x.md",
        "created": "2026-05-06T19:00:00-03:00",
    }
    store.delete("abc")
    assert store.find_by_topic_key("project:x") is None


def test_get_fts_body_by_path(store: VecStore):
    store.upsert(
        id_="abc",
        path="memory/x.md",
        title="X",
        type_="note",
        tags=[],
        created="t",
        updated="t",
        body_hash="h",
        embedding=_emb(1, 0, 0, 0),
        body_text="indexed body",
    )

    assert store.get_fts_body_by_path("memory/x.md") == "indexed body"
    assert store.get_fts_body_by_path("memory/missing.md") == ""


def test_dim_mismatch_raises(store: VecStore):
    with pytest.raises(ValueError, match="dimension mismatch"):
        store.upsert(
            id_="abc",
            path="memory/x.md",
            title="X",
            type_="note",
            tags=[],
            created="t",
            updated="t",
            body_hash="h",
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
            id_="abc",
            path="memory/x.md",
            title="X",
            type_="note",
            tags=[],
            created="t",
            updated="t",
            body_hash="h",
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


@pytest.mark.concurrency
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
            id_=f"id-{i}",
            path=f"memory/n{i}.md",
            title=f"n{i}",
            type_="note",
            tags=[],
            created=ts,
            updated=ts,
            body_hash="h",
            embedding=_emb(1, 0, 0, 0),
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
    store.upsert_repo_source(source=source)
    store.upsert_repo_files(
        repo_id=source["id"],
        repo_name=source["name"],
        indexed_at=source["indexed_at"],
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
                    }
                ],
            }
        ],
    )
    store.upsert_repo_embeddings(
        repo_id=source["id"],
        embeddings=[("chunk1", _emb(1, 0, 0, 0))],
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


@pytest.mark.concurrency
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
                    id_=f"t{n}-{i}",
                    path=f"memory/t{n}-{i}.md",
                    title=f"t{n}-{i}",
                    type_="note",
                    tags=[],
                    created="t",
                    updated="t",
                    body_hash="h",
                    embedding=_emb(1, n % 4, i % 4, 0),
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


@pytest.mark.concurrency
def test_close_closes_live_worker_thread_connections(tmp_path: Path):
    """Closing the store must close thread-local connections owned by live workers.

    FastMCP HTTP transports keep worker threads alive across requests. If shutdown
    closes only the caller's thread-local connection, worker connections survive
    until process GC and surface as ResourceWarning/unclosed sqlite databases.
    """
    import threading

    store = VecStore(tmp_path / "vec.db", dims=4)
    ready = threading.Event()
    proceed = threading.Event()
    results: list[str] = []

    def worker() -> None:
        conn = store._conn
        conn.execute("SELECT 1").fetchone()
        ready.set()
        proceed.wait(timeout=5)
        try:
            conn.execute("SELECT 1").fetchone()
        except sqlite3.ProgrammingError:
            results.append("closed")
        else:
            results.append("open")

    t = threading.Thread(target=worker)
    t.start()
    assert ready.wait(timeout=5)

    store.close()
    proceed.set()
    t.join(timeout=5)

    assert results == ["closed"]


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
    conn.execute("INSERT INTO meta VALUES ('m1','/p1','T1','decision','[]','t','t','h',NULL)")
    conn.execute("INSERT INTO meta VALUES ('m2','/p2','T2','reference','[]','t','t','h',NULL)")
    conn.execute(
        "INSERT INTO vec (id, embedding) VALUES (?, ?)", ("m1", serialize_float32([1.0, 0, 0, 0]))
    )
    conn.execute(
        "INSERT INTO vec (id, embedding) VALUES (?, ?)", ("m2", serialize_float32([0, 1.0, 0, 0]))
    )
    conn.execute("INSERT INTO source_feedback VALUES ('f1','m1','why',1,'t',NULL)")
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
        s._conn.execute("UPDATE schema_meta SET value = '4' WHERE key = 'embedder_dims'")
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


def test_embedder_version_legacy_index_not_silently_stamped(tmp_path: Path, caplog):
    """Regression: opening a pre-schema_meta index that already contains
    vectors must NOT stamp the current model onto it — that vector space is
    unknown (possibly a different same-width model) and stamping would disarm
    the mismatch guard forever. It warns (pointing at reindex --rebuild) and
    leaves schema_meta unstamped until a real rebuild stamps it."""
    import logging

    db = tmp_path / "legacy.db"
    _legacy_vec_db(db)

    with caplog.at_level(logging.WARNING, logger="memo.store.schema"):
        s = VecStore(db, dims=4, embedder_model="vendor/Model-B")
    keys = {r["key"] for r in s._conn.execute("SELECT key FROM schema_meta").fetchall()}
    assert "embedder_model" not in keys, "legacy index must not be blessed as the current model"
    assert any("reindex --rebuild" in rec.getMessage() for rec in caplog.records)
    s.close()

    # Reopening — even with another model — must neither raise a bogus
    # mismatch nor adopt that model either.
    s2 = VecStore(db, dims=4, embedder_model="vendor/Model-C")
    keys = {r["key"] for r in s2._conn.execute("SELECT key FROM schema_meta").fetchall()}
    assert "embedder_model" not in keys
    s2.close()


# ── Soft-delete visibility on meta read surfaces ──────────────────────────


def test_soft_deleted_rows_hidden_from_meta_queries(store: VecStore):
    """Regression: list_by_tag / records_around_created / chunks_by_parent /
    chunks_adjacent must not resurface soft-deleted tombstones (soft delete
    is default ON). chunks_by_parent_id deliberately still returns them so
    delete/maintain can cascade hard-deletes over stale chunks."""

    def _put(id_: str, created: str, seq: int) -> None:
        store.upsert(
            id_=id_,
            path=f"memory/{id_}.md",
            title=f"T {id_}",
            type_="note",
            tags=["shared-tag"],
            created=created,
            updated=created,
            body_hash="h",
            embedding=_emb(1, 0, 0, 0),
            extra={"parent_path": "notes/parent.md", "parent_id": "parent-1", "chunk_seq": seq},
        )

    _put("keep1", "2026-01-01T00:00:00+00:00", 1)
    _put("gone1", "2026-01-02T00:00:00+00:00", 2)
    _put("keep2", "2026-01-03T00:00:00+00:00", 3)
    assert store.delete("gone1") is True
    # The meta row survives as a tombstone (soft delete keeps it for restore).
    row = store._conn.execute("SELECT deleted_at FROM meta WHERE id = 'gone1'").fetchone()
    assert row is not None and row["deleted_at"]

    assert {r["id"] for r in store.list_by_tag("shared-tag")} == {"keep1", "keep2"}
    around = store.records_around_created("2026-01-01T12:00:00+00:00", before=2, after=2)
    assert {r["id"] for r in around} == {"keep1", "keep2"}
    assert {r["id"] for r in store.chunks_by_parent("notes/parent.md", limit=10)} == {
        "keep1",
        "keep2",
    }
    assert {r["id"] for r in store.chunks_adjacent("notes/parent.md", 2, before=1, after=1)} == {
        "keep1",
        "keep2",
    }
    # Cascade/prune path must still see the tombstone.
    assert {r["id"] for r in store.chunks_by_parent_id("parent-1")} == {"keep1", "gone1", "keep2"}


# ── Tantivy dual-write ordering ───────────────────────────────────────────


@pytest.mark.concurrency
def test_tantivy_write_lock_spans_sqlite_commit(tmp_path: Path):
    """Regression: the tantivy dual-write must follow sqlite commit order.
    The write lock has to be held across the sqlite tx, so a writer that is
    mid-tantivy-write blocks a concurrent writer's sqlite COMMIT — not just
    its tantivy write. Otherwise B can commit sqlite v2 after A committed v1
    but index tantivy v2 BEFORE A indexes v1, leaving tantivy permanently
    serving the older version."""
    import threading

    s = VecStore(tmp_path / "vec.db", dims=4)
    in_tantivy = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    class _StubTantivy:
        def delete_document(self, id_: str) -> None:
            pass

        def add_document(self, id_: str, title: str, tags: str, body: str) -> None:
            calls.append(title)
            in_tantivy.set()
            release.wait(timeout=10)

        def commit(self) -> None:
            pass

    s._get_tantivy = lambda: _StubTantivy()  # type: ignore[method-assign]

    def _upsert(title: str) -> None:
        s.upsert(
            id_="abc",
            path="memory/x.md",
            title=title,
            type_="note",
            tags=[],
            created="t",
            updated="t",
            body_hash="h",
            embedding=_emb(1, 0, 0, 0),
        )

    a = threading.Thread(target=_upsert, args=("v1",))
    a.start()
    assert in_tantivy.wait(timeout=10)  # A committed sqlite v1, now mid-tantivy-write
    b = threading.Thread(target=_upsert, args=("v2",))
    b.start()
    b.join(timeout=0.5)  # give B time to (wrongly) commit / (rightly) block
    row = s._conn.execute("SELECT title FROM meta WHERE id = 'abc'").fetchone()
    assert row["title"] == "v1", "concurrent writer committed sqlite while A held the dual-write"
    release.set()
    a.join(timeout=10)
    b.join(timeout=10)
    assert not a.is_alive() and not b.is_alive()
    row = s._conn.execute("SELECT title FROM meta WHERE id = 'abc'").fetchone()
    assert row["title"] == "v2"
    # tantivy write order matches sqlite commit order.
    assert calls == ["v1", "v2"]
    s.close()


# ── QA remediation regressions (store lows) ───────────────────────────────


def test_tantivy_meta_lookup_tolerates_missing_deleted_at(store: VecStore):
    """Regression: the tantivy BM25/fuzzy meta-resolution SQL hard-coded
    `deleted_at`, so an old DB whose deleted_at ALTER was skipped (suppressed
    lock error) crashed every search with `no such column: deleted_at`. It now
    uses the same PRAGMA-guarded filter as every other meta reader."""

    class _StubTantivy:
        def delete_document(self, id_: str) -> None:
            pass

        def add_document(self, id_: str, title: str, tags: str, body: str) -> None:
            pass

        def commit(self) -> None:
            pass

        def search_bm25(self, query: str, k: int):
            return [{"id": "keep", "score": 2.0}, {"id": "gone", "score": 1.0}]

        def search_fuzzy(self, query: str, k: int):
            return [{"id": "keep", "score": 2.0}, {"id": "gone", "score": 1.0}]

    store._get_tantivy = lambda: _StubTantivy()  # type: ignore[method-assign]
    for id_ in ("keep", "gone"):
        store.upsert(
            id_=id_,
            path=f"memory/{id_}.md",
            title=f"T {id_}",
            type_="note",
            tags=[],
            created="t",
            updated="t",
            body_hash="h",
            embedding=_emb(1, 0, 0, 0),
        )
    assert store.delete("gone") is True

    # Current schema: the soft-deleted tombstone stays filtered out.
    assert [r["id"] for r in store.search_bm25("q")] == ["keep"]
    assert [r["id"] for r in store.search_fuzzy("q")] == ["keep"]

    # Pre-migration schema (deleted_at ALTER was skipped): searches must not
    # raise. The tombstone marker vanishes with the column, so `gone`
    # legitimately reappears.
    # v5's active-topic uniqueness predicate references deleted_at. Remove that
    # capability index to emulate the pre-migration schema this test targets.
    store._conn.execute("DROP INDEX IF EXISTS idx_meta_active_topic_unique")
    store._conn.execute("ALTER TABLE meta DROP COLUMN deleted_at")
    assert [r["id"] for r in store.search_bm25("q")] == ["keep", "gone"]
    assert [r["id"] for r in store.search_fuzzy("q")] == ["keep", "gone"]


def test_crush_cache_rejects_non_hex_hash(tmp_path: Path):
    """Regression: `CrushCache.retrieve` joined an unvalidated, externally
    supplied hash into a filesystem path — a `../`-style marker could read any
    {ts, content}-shaped JSON outside the cache dir. Non-hex keys are now
    refused before touching the filesystem."""
    import json
    from datetime import UTC, datetime

    from memo.errors import ValidationError
    from memo.store.crush_cache import CrushCache

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    cache = CrushCache(state_dir)

    # Decoy with the exact {ts, content} shape, one level above cache_dir.
    (state_dir / "decoy.json").write_text(
        json.dumps({"ts": datetime.now(UTC).isoformat(), "content": "secret"}),
        encoding="utf-8",
    )
    assert cache.retrieve("../decoy") is None
    assert cache.retrieve("ABCDEF123456") is None  # uppercase — not our digest
    assert cache.retrieve("abc123") is None  # < 8 chars
    with pytest.raises(ValidationError):
        cache.cache("../decoy", "boom")
    # The decoy was never overwritten or read through the traversal path.
    assert "secret" in (state_dir / "decoy.json").read_text(encoding="utf-8")

    # The real key shape (sha256 hexdigest prefix) still round-trips.
    cache.cache("a1b2c3d4e5f60718", "original")
    assert cache.retrieve("a1b2c3d4e5f60718") == "original"


def test_vec_dims_mismatch_raises_domain_storage_error(tmp_path: Path):
    """Regression: the dims-mismatch guard raised a bare RuntimeError, so
    CLI/MCP entry points catching MemoError missed it. It now raises
    StorageError (still a RuntimeError subclass, so old handlers keep
    working)."""
    from memo.errors import MemoError

    db_path = tmp_path / "vec.db"
    VecStore(db_path, dims=4).close()
    with pytest.raises(MemoError, match="dimension mismatch"):
        VecStore(db_path, dims=8)


@pytest.mark.concurrency
def test_vec_migration_snapshot_inside_tx(tmp_path: Path):
    """Regression: `_validate_vec_schema` snapshotted the old vec table on the
    autocommit connection BEFORE opening the migration transaction, so a row
    committed by another process in that window (e.g. a still-running recall
    daemon on the old binary) was silently dropped by the DROP + re-insert.
    The snapshot now happens inside the same BEGIN IMMEDIATE tx as the DROP."""
    import contextlib

    import sqlite_vec
    from sqlite_vec import serialize_float32

    db = tmp_path / "vec.db"
    store = VecStore(db, dims=4, vec_quant="off")
    # Re-arm the migration: swap the migrated vec table for the OLD layout
    # (no `type` metadata column) holding one row. Raw float32 blobs below
    # (serialize_float32) must match the store's own quant mode.
    store._conn.execute("DROP TABLE vec")
    store._conn.execute(
        "CREATE VIRTUAL TABLE vec USING vec0(id TEXT PRIMARY KEY, "
        "embedding FLOAT[4] distance_metric=cosine)"
    )
    store._conn.execute(
        "INSERT INTO meta (id, path, title, type, tags, created, updated, body_hash) "
        "VALUES ('m1', '/p1', 'T1', 'note', '[]', 't', 't', 'h')"
    )
    store._conn.execute(
        "INSERT INTO vec (id, embedding) VALUES (?, ?)",
        ("m1", serialize_float32([1.0, 0, 0, 0])),
    )
    store._conn.commit()

    real_tx = store._tx
    fired: list[bool] = []

    @contextlib.contextmanager
    def tx_with_concurrent_write():
        # Another process commits into the OLD table right as the migration
        # opens its transaction — i.e. after any outside-tx snapshot.
        if not fired:
            fired.append(True)
            other = sqlite3.connect(str(db))
            other.enable_load_extension(True)
            sqlite_vec.load(other)
            other.enable_load_extension(False)
            other.execute(
                "INSERT INTO meta (id, path, title, type, tags, created, updated, body_hash) "
                "VALUES ('m2', '/p2', 'T2', 'note', '[]', 't', 't', 'h')"
            )
            other.execute(
                "INSERT INTO vec (id, embedding) VALUES (?, ?)",
                ("m2", serialize_float32([0, 1.0, 0, 0])),
            )
            other.commit()
            other.close()
        with real_tx() as cx:
            yield cx

    store._tx = tx_with_concurrent_write  # type: ignore[method-assign]
    try:
        store._validate_vec_schema()
    finally:
        store._tx = real_tx  # type: ignore[method-assign]

    assert fired, "migration transaction never opened"
    ids = {r["id"] for r in store._conn.execute("SELECT id FROM vec").fetchall()}
    assert ids == {"m1", "m2"}, "concurrent commit lost during vec migration"
    store.close()
