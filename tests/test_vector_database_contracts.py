"""Behavioral contracts for memo's sqlite-vec database boundary.

These tests exercise the real SQLite + vec0 implementation.  They deliberately
avoid MLX and the ``Memory`` facade so persistence, transactions, filtering,
and index housekeeping failures are attributed to ``VecStore`` itself.
"""

from __future__ import annotations

import sqlite3
from inspect import signature
from pathlib import Path
from typing import Any

import pytest

from memo.store import VecStore
from memo.store.vec_base import VecStoreBase

pytestmark = [pytest.mark.db_contract, pytest.mark.resource_hygiene]

_X = [1.0, 0.0, 0.0, 0.0]
_Y = [0.0, 1.0, 0.0, 0.0]
_NEG_X = [-1.0, 0.0, 0.0, 0.0]


def test_base_store_required_extension_points_fail_explicitly() -> None:
    """The shared interface must never silently accept unsupported writes/reads."""
    base = VecStoreBase()
    assert signature(base.find_by_prefix).parameters["limit"].default == 10
    assert signature(base.list_recent).parameters["limit"].default == 20
    assert signature(base.upsert).parameters["body_text"].default == ""
    with pytest.raises(NotImplementedError):
        base.find_by_prefix("abc")
    with pytest.raises(NotImplementedError):
        base.list_recent()
    with pytest.raises(NotImplementedError):
        base.upsert(
            id_="id",
            path="memory/id.md",
            title="Title",
            type_="note",
            tags=[],
            created="2026-01-01T00:00:00+00:00",
            updated="2026-01-01T00:00:00+00:00",
            body_hash="hash",
            embedding=[1.0],
        )


@pytest.fixture
def vector_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A deterministic float32 store with no optional full-text sidecar."""
    monkeypatch.setenv("MEMO_TANTIVY_ENABLED", "0")
    monkeypatch.setenv("MEMO_SOFT_DELETE", "1")
    store = VecStore(
        tmp_path / "vectors.db",
        dims=4,
        embedder_model="tests/vector-contract-v1",
        vec_quant="off",
    )
    yield store
    store.close()


def _put(
    store: VecStore,
    id_: str,
    embedding: list[float],
    *,
    path: str | None = None,
    title: str | None = None,
    type_: str = "note",
    tags: list[str] | None = None,
    created: str = "2026-01-01T00:00:00+00:00",
    updated: str = "2026-01-01T00:00:00+00:00",
    body: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    store.upsert(
        id_=id_,
        path=path or f"memory/{id_}.md",
        title=title or id_.title(),
        type_=type_,
        tags=tags or [],
        created=created,
        updated=updated,
        body_hash=f"hash-{id_}",
        embedding=embedding,
        body_text=body or f"body for {id_}",
        extra=extra,
    )


def test_reopen_preserves_vector_metadata_and_lexical_indexes(
    vector_store: VecStore,
) -> None:
    _put(
        vector_store,
        "alpha",
        _X,
        title="Durable alpha",
        tags=["durable", "vector"],
        body="unique persistence marker",
        extra={"origin": "contract-test"},
    )
    _put(vector_store, "beta", _Y, body="unrelated text")
    db_path = vector_store.db_path
    vector_store.close()

    reopened = VecStore(
        db_path,
        dims=4,
        embedder_model="tests/vector-contract-v1",
        vec_quant="off",
    )
    try:
        row = reopened.get("alpha")
        assert row is not None
        assert row["title"] == "Durable alpha"
        assert row["tags"] == ["durable", "vector"]
        assert row["extra"] == {"origin": "contract-test"}
        assert [hit["id"] for hit in reopened.search(_X, limit=2)] == ["alpha", "beta"]
        assert [hit["id"] for hit in reopened.search_bm25("persistence marker", limit=5)] == [
            "alpha"
        ]
    finally:
        reopened.close()


def test_failed_path_collision_rolls_back_every_index(vector_store: VecStore) -> None:
    """A UNIQUE path failure must not leave orphan vec/FTS rows behind."""
    _put(
        vector_store,
        "owner",
        _X,
        path="memory/shared.md",
        title="Original owner",
        body="original searchable marker",
    )

    with pytest.raises(sqlite3.IntegrityError):
        _put(
            vector_store,
            "contender",
            _Y,
            path="memory/shared.md",
            title="Rejected contender",
            body="contenderonlytoken",
        )

    assert vector_store.count() == 1
    assert vector_store.get("contender") is None
    assert vector_store.get_fts_body_by_path("memory/shared.md") == "original searchable marker"
    assert [hit["id"] for hit in vector_store.search(_X, limit=10)] == ["owner"]
    assert vector_store.search_bm25("contenderonlytoken", limit=10) == []
    assert vector_store.connection.execute("SELECT COUNT(*) FROM vec").fetchone()[0] == 1


def test_invalid_replacement_keeps_previous_vector_and_text(vector_store: VecStore) -> None:
    _put(
        vector_store,
        "stable",
        _X,
        title="Stable version",
        body="old durable text",
    )

    with pytest.raises(ValueError, match="dimension mismatch"):
        _put(
            vector_store,
            "stable",
            [1.0, 0.0],
            title="Invalid replacement",
            body="new text must roll back",
        )

    row = vector_store.get("stable")
    assert row is not None and row["title"] == "Stable version"
    assert vector_store.get_fts_body_by_path("memory/stable.md") == "old durable text"
    assert [hit["id"] for hit in vector_store.search(_X, limit=1)] == ["stable"]
    assert vector_store.search_bm25("replacement", limit=5) == []


def test_vector_search_combines_date_type_and_exact_tag_filters(
    vector_store: VecStore,
) -> None:
    """Filtered nearest neighbours refill the requested result set correctly."""
    rows = [
        ("secret", _X, "note", ["private"], "2026-01-03T00:00:00+00:00"),
        ("near", [0.99, 0.1, 0.0, 0.0], "note", ["private-note"], "2026-01-03T01:00:00+01:00"),
        ("reference", [0.98, 0.2, 0.0, 0.0], "reference", [], "2026-01-03T00:30:00+00:00"),
        ("valid", [0.95, 0.3, 0.0, 0.0], "note", ["public"], "2026-01-04T00:00:00+00:00"),
        ("too-old", [0.9, 0.4, 0.0, 0.0], "note", ["public"], "2025-12-01T00:00:00+00:00"),
    ]
    for id_, embedding, type_, tags, updated in rows:
        _put(
            vector_store,
            id_,
            embedding,
            type_=type_,
            tags=tags,
            updated=updated,
        )

    hits = vector_store.search(
        _X,
        limit=2,
        exclude_types={"reference"},
        exclude_tags={"private"},
        date_from="2026-01-02T23:30:00+00:00",
        date_to="2026-01-04T00:00:00+00:00",
    )

    # Exact-tag filtering removes "private" but must not remove
    # "private-note". Mixed timezone offsets are compared as instants.
    assert [hit["id"] for hit in hits] == ["near", "valid"]
    assert hits[0]["score"] >= hits[1]["score"]


def test_vector_ranking_matches_reference_cosine_order(vector_store: VecStore) -> None:
    vectors = {
        "same": _X,
        "near": [0.8, 0.6, 0.0, 0.0],
        "orthogonal": _Y,
        "opposite": _NEG_X,
    }
    for id_, embedding in vectors.items():
        _put(vector_store, id_, embedding)

    hits = vector_store.search(_X, limit=len(vectors))

    expected = sorted(
        vectors,
        key=lambda id_: sum(a * b for a, b in zip(_X, vectors[id_], strict=True)),
        reverse=True,
    )
    assert [hit["id"] for hit in hits] == expected
    assert hits[0]["score"] == pytest.approx(1.0)
    assert hits[-1]["score"] == 0.0  # public scores clamp negative cosine to zero


def test_upsert_moves_existing_id_between_vector_and_type_partitions(
    vector_store: VecStore,
) -> None:
    _put(vector_store, "moving", _X, type_="note", body="oldonlytoken")
    _put(vector_store, "anchor", [0.8, 0.6, 0.0, 0.0], type_="note")
    assert [hit["id"] for hit in vector_store.search(_X, limit=2)] == ["moving", "anchor"]

    _put(
        vector_store,
        "moving",
        _Y,
        type_="decision",
        title="Moved decision",
        body="newonlytoken",
    )

    assert [hit["id"] for hit in vector_store.search(_X, limit=2)] == ["anchor", "moving"]
    assert [hit["id"] for hit in vector_store.search(_Y, limit=1)] == ["moving"]
    assert [hit["id"] for hit in vector_store.search(_Y, limit=5, type_="decision")] == ["moving"]
    assert vector_store.search_bm25("oldonlytoken", limit=5) == []
    assert [hit["id"] for hit in vector_store.search_bm25("newonlytoken", limit=5)] == ["moving"]
    row = vector_store.get("moving")
    assert row is not None and row["type"] == "decision" and row["title"] == "Moved decision"
    assert (
        vector_store.connection.execute("SELECT COUNT(*) FROM vec WHERE id = 'moving'").fetchone()[
            0
        ]
        == 1
    )


def test_query_dimension_mismatch_is_rejected_before_sql(vector_store: VecStore) -> None:
    _put(vector_store, "alpha", _X)

    with pytest.raises(ValueError, match=r"got 2, expected 4"):
        vector_store.search([1.0, 0.0], limit=5)

    assert vector_store.count() == 1


def test_clearing_derived_vector_index_preserves_primary_signal_tables(
    vector_store: VecStore,
) -> None:
    _put(vector_store, "alpha", _X, body="derived searchable body")
    vector_store.touch(["alpha"], ts="2026-01-02T00:00:00+00:00")
    vector_store.set_confidence_batch([("alpha", 0.42)])
    feedback_id = vector_store.record_source_feedback(
        source_id="alpha",
        query_text="why alpha",
        query_emb=_Y,
        rating=1,
        feedback_id="feedback-alpha",
    )
    vector_store.upsert_repo_source(
        {
            "id": "repo-1",
            "name": "memo",
            "url": "https://example.invalid/memo.git",
            "ref": "main",
            "commit_sha": "abc123",
            "clone_path": "/tmp/repo-1",
            "indexed_at": "2026-01-02T00:00:00+00:00",
            "status": "ready",
        }
    )
    signal_before = vector_store.dump_signal()

    assert vector_store.clear_memory_index() == 1

    assert vector_store.count() == 0
    assert vector_store.search(_X, limit=5) == []
    assert vector_store.search_bm25("searchable", limit=5) == []
    assert vector_store.dump_signal() == signal_before
    assert vector_store.get_repo_source("repo-1")["commit_sha"] == "abc123"
    matches = vector_store.find_feedback_for_source("alpha", _Y, threshold=0.99)
    assert [match["id"] for match in matches] == [feedback_id]


def test_hard_delete_vacuums_tombstone_and_all_attached_signals(
    vector_store: VecStore,
) -> None:
    _put(vector_store, "expired", _NEG_X)
    vector_store.touch(["expired"])
    vector_store.set_confidence_batch([("expired", 0.35)])
    vector_store.record_source_feedback(
        source_id="expired",
        query_text="expired query",
        query_emb=_Y,
        rating=-1,
        feedback_id="feedback-expired",
    )
    assert vector_store.delete("expired") is True
    assert vector_store.list_soft_deleted() == ["expired"]

    assert vector_store.hard_delete("expired") is True

    assert vector_store.list_soft_deleted() == []
    assert vector_store.get_access("expired") == {"access_count": 0, "last_accessed": None}
    signal = vector_store.dump_signal()
    assert all(row["id"] != "expired" for row in signal["access"])
    assert all(row["id"] != "expired" for row in signal["memory_health"])
    assert all(row["source_id"] != "expired" for row in signal["source_feedback"])
    assert (
        vector_store.connection.execute(
            "SELECT COUNT(*) FROM source_feedback_vec WHERE source_id = 'expired'"
        ).fetchone()[0]
        == 0
    )


def test_soft_delete_cutoff_compares_timezone_offsets_as_instants(
    vector_store: VecStore,
) -> None:
    for id_ in ("older", "same-instant", "newer"):
        _put(vector_store, id_, _X)
        assert vector_store.delete(id_) is True
    vector_store.connection.executemany(
        "UPDATE meta SET deleted_at = ? WHERE id = ?",
        [
            ("2026-01-01T01:59:59+02:00", "older"),
            ("2026-01-01T00:00:00+00:00", "same-instant"),
            ("2026-01-01T00:00:01+00:00", "newer"),
        ],
    )
    vector_store.connection.commit()

    assert vector_store.list_soft_deleted(before="2026-01-01T00:00:00+00:00") == ["older"]
