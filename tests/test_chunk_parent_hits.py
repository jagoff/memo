"""Chunk->parent rollup for auto-recall (MEMO_RECALL_CHUNK_PARENT).

Covers the gap measured 2026-08-16: a chunked long durable memory's
fragments (type=reference) never reach auto-recall because
MEMO_RECALL_EXCLUDE_REFERENCE drops the whole tier at the SQL layer, so the
canonical parent (whose single-vector embedding dilutes across the whole
document) never surfaces either."""

from __future__ import annotations

from types import SimpleNamespace

from memo.memory.record import MemoryRecord
from memo.recall_logic import fetch_chunk_parent_hits


def _parent(id_: str = "608e16fcfe504cc09fbb900a50ccdd7a") -> MemoryRecord:
    return MemoryRecord(
        id=id_,
        path="memo/gate-fix.md",
        title="El pre-push recall gate mide el árbol compartido",
        type="bug",
        tags=[],
        created="2026-08-16T00:00:00+00:00",
        updated="2026-08-16T00:00:00+00:00",
        body="canonical body " * 10,
    )


def _chunk_hit(parent_id: str | None, *, score: float = 0.97, type_: str = "reference") -> object:
    return SimpleNamespace(
        id=f"{parent_id or 'orphan'}_chunk_0",
        type=type_,
        extra={"parent_id": parent_id} if parent_id else {"parent_path": "notes/vault.md"},
        score=score,
    )


def test_resolves_parent_id_chunk_to_canonical_parent():
    parent = _parent()
    mem = SimpleNamespace(
        search=lambda *a, **kw: [_chunk_hit(parent.id, score=0.97)],
        get=lambda id_: parent if id_ == parent.id else None,
    )
    out = fetch_chunk_parent_hits(mem, "query", mode="vec", limit=5, budget_ms=400.0)
    assert len(out) == 1
    assert out[0].id == parent.id
    assert out[0].type == "bug"  # the parent's own type, not "reference"
    assert out[0].score == 0.97  # carries the CHUNK's score, not the parent's


def test_does_not_surface_bulk_vault_reference_without_parent_id():
    """The parent_path-only ingest schema (memo ingest --chunk) has no
    durable parent to resolve to and must stay excluded — that's the whole
    point of MEMO_RECALL_EXCLUDE_REFERENCE."""
    mem = SimpleNamespace(
        search=lambda *a, **kw: [_chunk_hit(None)],
        get=lambda id_: None,
    )
    out = fetch_chunk_parent_hits(mem, "query", mode="vec", limit=5, budget_ms=400.0)
    assert out == []


def test_dedups_multiple_chunks_of_the_same_parent():
    parent = _parent()
    mem = SimpleNamespace(
        search=lambda *a, **kw: [
            _chunk_hit(parent.id, score=0.97),
            _chunk_hit(parent.id, score=0.90),
        ],
        get=lambda id_: parent if id_ == parent.id else None,
    )
    out = fetch_chunk_parent_hits(mem, "query", mode="vec", limit=5, budget_ms=400.0)
    assert len(out) == 1
    assert out[0].score == 0.97  # first (best-ranked) chunk's score wins


def test_skips_a_chunk_whose_parent_was_deleted():
    mem = SimpleNamespace(
        search=lambda *a, **kw: [_chunk_hit("gone-id", score=0.97)],
        get=lambda id_: None,
    )
    out = fetch_chunk_parent_hits(mem, "query", mode="vec", limit=5, budget_ms=400.0)
    assert out == []


def test_never_raises_on_search_failure():
    def _boom(*a, **kw):
        raise RuntimeError("db locked")

    mem = SimpleNamespace(search=_boom, get=lambda id_: None)
    assert fetch_chunk_parent_hits(mem, "query", mode="vec", limit=5, budget_ms=400.0) == []


def test_never_raises_on_get_failure():
    def _boom(id_):
        raise RuntimeError("store gone")

    mem = SimpleNamespace(
        search=lambda *a, **kw: [_chunk_hit("some-id", score=0.97)],
        get=_boom,
    )
    assert fetch_chunk_parent_hits(mem, "query", mode="vec", limit=5, budget_ms=400.0) == []
