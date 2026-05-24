"""VecStore — sqlite-vec CRUD + cosine search.

These tests exercise the SQLite + vec0 layer directly without
requiring the MLX runtime, so they run on any platform with
`sqlite-vec` available.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memo.store import VecStore


@pytest.fixture
def store(tmp_path: Path) -> VecStore:
    return VecStore(tmp_path / "vec.db", dims=4)


def _emb(*xs: float) -> list[float]:
    """Build a small fixed-dim normalised vector."""
    import math

    norm = math.sqrt(sum(x * x for x in xs)) or 1.0
    return [x / norm for x in xs]


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


def test_delete_removes_from_both_tables(store: VecStore):
    store.upsert(
        id_="abc", path="memory/x.md", title="X", type_="note", tags=[],
        created="t", updated="t", body_hash="h", embedding=_emb(1, 0, 0, 0),
    )
    assert store.delete("abc") is True
    assert store.get("abc") is None
    assert store.search(_emb(1, 0, 0, 0), limit=10) == []


def test_dim_mismatch_raises(store: VecStore):
    with pytest.raises(ValueError, match="dim mismatch"):
        store.upsert(
            id_="abc", path="memory/x.md", title="X", type_="note", tags=[],
            created="t", updated="t", body_hash="h",
            embedding=[1.0, 0.0],  # 2-dim, store expects 4
        )


def test_existing_vec_table_dim_mismatch_fails_fast(tmp_path: Path):
    db_path = tmp_path / "vec.db"
    store = VecStore(db_path, dims=4)
    store.close()

    with pytest.raises(RuntimeError, match=r"FLOAT\[4\].*FLOAT\[8\]"):
        VecStore(db_path, dims=8)


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
