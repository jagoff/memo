"""Tests for MEMO_CHUNK_INGEST — chunked reindex of long curated memorias.

When MEMO_CHUNK_INGEST=1, `Memory.reindex()` splits long multi-section notes
into heading-aware chunk records (type='reference', extra.parent_id=<parent>).
Short notes are indexed whole-note only (no extra chunk records).
Default behaviour (flag off) is unchanged.
"""

from __future__ import annotations

import pytest

from memo.chunker import DEFAULT_TARGET_CHARS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _long_body(n_sections: int = 3, words_per_section: int = 300) -> str:
    """Build a markdown body long enough to trigger chunking (> DEFAULT_TARGET_CHARS)."""
    sections = []
    for i in range(1, n_sections + 1):
        heading = f"## Section {i}"
        para = f"word{i} " * words_per_section
        sections.append(f"{heading}\n\n{para.strip()}")
    body = "\n\n".join(sections)
    assert len(body) > DEFAULT_TARGET_CHARS, (
        f"generated body ({len(body)} chars) is not long enough — increase n_sections/words_per_section"
    )
    return body


def _chunk_ids_for(store, parent_id: str) -> list[str]:
    """Collect all chunk record IDs whose extra.parent_id == parent_id."""
    all_rows = store.list_recent(limit=100_000)
    return [
        r["id"]
        for r in all_rows
        if isinstance(r.get("extra") or {}, dict)
        and r.get("extra", {}).get("parent_id") == parent_id
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reindex_chunk_flag_off_no_chunks(mock_memory, monkeypatch):
    """Default (flag off): reindex produces ONE record per memoria, no chunks."""
    monkeypatch.setenv("MEMO_CHUNK_INGEST", "0")

    body = _long_body()
    rec = mock_memory.save(content=body, title="Long Note", tags=["test"])

    # Force reindex so the record is processed.
    counts = mock_memory.reindex(force=True)

    assert counts["checked"] >= 1
    chunk_ids = _chunk_ids_for(mock_memory.store, rec.id)
    assert chunk_ids == [], f"Expected no chunks when flag is off, got: {chunk_ids}"


def test_reindex_chunk_flag_on_emits_multiple_chunks(mock_memory, monkeypatch):
    """MEMO_CHUNK_INGEST=1: long multi-section note produces ≥2 chunk records."""
    monkeypatch.setenv("MEMO_CHUNK_INGEST", "1")

    body = _long_body(n_sections=3, words_per_section=300)
    rec = mock_memory.save(content=body, title="Long Chunked Note", tags=["test"])

    counts = mock_memory.reindex(force=True)

    assert counts["checked"] >= 1
    chunk_ids = _chunk_ids_for(mock_memory.store, rec.id)
    assert len(chunk_ids) >= 2, (
        f"Expected ≥2 chunk records for long note, got {len(chunk_ids)}: {chunk_ids}"
    )


def test_reindex_chunk_short_note_no_extra_chunks(mock_memory, monkeypatch):
    """MEMO_CHUNK_INGEST=1: a short note (< DEFAULT_TARGET_CHARS) still has no chunk records."""
    monkeypatch.setenv("MEMO_CHUNK_INGEST", "1")

    short_body = "A brief note. " * 10  # well under DEFAULT_TARGET_CHARS
    assert len(short_body) < DEFAULT_TARGET_CHARS
    rec = mock_memory.save(content=short_body, title="Short Note", tags=["test"])

    mock_memory.reindex(force=True)

    chunk_ids = _chunk_ids_for(mock_memory.store, rec.id)
    assert chunk_ids == [], f"Short note should produce no chunk records, got: {chunk_ids}"


def test_reindex_chunk_records_are_reference_type(mock_memory, monkeypatch):
    """Chunk records have type='reference'."""
    monkeypatch.setenv("MEMO_CHUNK_INGEST", "1")

    body = _long_body()
    rec = mock_memory.save(content=body, title="Typed Chunks", tags=["test"])
    mock_memory.reindex(force=True)

    chunk_ids = _chunk_ids_for(mock_memory.store, rec.id)
    assert chunk_ids, "Expected chunk records"

    for cid in chunk_ids:
        row = mock_memory.store.get(cid)
        assert row is not None
        assert row["type"] == "reference", (
            f"chunk {cid} has type={row['type']!r}, expected 'reference'"
        )


def test_reindex_chunk_extra_fields(mock_memory, monkeypatch):
    """Chunk extra contains parent_id, chunk_index, and chunk_count."""
    monkeypatch.setenv("MEMO_CHUNK_INGEST", "1")

    body = _long_body(n_sections=2, words_per_section=400)
    rec = mock_memory.save(content=body, title="Extra Fields", tags=["test"])
    mock_memory.reindex(force=True)

    chunk_ids = _chunk_ids_for(mock_memory.store, rec.id)
    assert chunk_ids

    for cid in chunk_ids:
        row = mock_memory.store.get(cid)
        assert row is not None
        extra = row.get("extra") or {}
        assert extra.get("parent_id") == rec.id, f"chunk {cid}: parent_id mismatch"
        assert "chunk_index" in extra, f"chunk {cid}: missing chunk_index"
        assert "chunk_count" in extra, f"chunk {cid}: missing chunk_count"


def test_derived_chunk_ids_remain_resolvable(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_CHUNK_INGEST", "1")
    rec = mock_memory.save(content=_long_body(), title="Resolvable Chunks", tags=["test"])
    mock_memory.reindex(force=True)
    chunk_id = _chunk_ids_for(mock_memory.store, rec.id)[0]

    assert mock_memory.resolve_id(chunk_id) == chunk_id
    fetched = mock_memory.get(chunk_id)
    assert fetched is not None
    assert fetched.id == chunk_id
    assert mock_memory.around(chunk_id)["mode"] == "chunk_seq"


def test_derived_chunk_ids_are_read_only(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_CHUNK_INGEST", "1")
    rec = mock_memory.save(content=_long_body(), title="Read-only chunks", tags=["test"])
    mock_memory.reindex(force=True)
    chunk_id = _chunk_ids_for(mock_memory.store, rec.id)[0]

    with pytest.raises(ValueError, match="read-only"):
        mock_memory.update(chunk_id, content="ghost mutation")
    with pytest.raises(ValueError, match="read-only"):
        mock_memory.forget(chunk_id)
    with pytest.raises(ValueError, match="read-only"):
        mock_memory.delete(chunk_id)
    assert not any("#chunk-" in str(path) for path in mock_memory.cfg.memory_dir.rglob("*"))


def test_reindex_chunk_title_contains_parent_title(mock_memory, monkeypatch):
    """Chunk titles include the parent note title."""
    monkeypatch.setenv("MEMO_CHUNK_INGEST", "1")

    body = _long_body(n_sections=2, words_per_section=300)
    rec = mock_memory.save(content=body, title="Parent Title", tags=["test"])
    mock_memory.reindex(force=True)

    chunk_ids = _chunk_ids_for(mock_memory.store, rec.id)
    assert chunk_ids

    for cid in chunk_ids:
        row = mock_memory.store.get(cid)
        assert row is not None
        assert "Parent Title" in row["title"], (
            f"chunk title {row['title']!r} does not contain parent title"
        )


def test_reindex_chunk_idempotent(mock_memory, monkeypatch):
    """Running reindex twice with the same content does not create duplicate chunks."""
    monkeypatch.setenv("MEMO_CHUNK_INGEST", "1")

    body = _long_body()
    rec = mock_memory.save(content=body, title="Idempotent Note", tags=["test"])

    mock_memory.reindex(force=True)
    chunk_ids_first = set(_chunk_ids_for(mock_memory.store, rec.id))

    mock_memory.reindex(force=True)
    chunk_ids_second = set(_chunk_ids_for(mock_memory.store, rec.id))

    assert chunk_ids_first == chunk_ids_second, (
        f"Second reindex changed chunk set:\n  first: {chunk_ids_first}\n  second: {chunk_ids_second}"
    )
    assert len(chunk_ids_first) >= 2


def test_gc_preserves_chunks_while_parent_markdown_exists(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_CHUNK_INGEST", "1")
    rec = mock_memory.save(content=_long_body(), title="GC Parent", tags=["test"])
    mock_memory.reindex(force=True)
    chunk_ids = _chunk_ids_for(mock_memory.store, rec.id)
    assert chunk_ids

    report = mock_memory.gc(fix=True)

    assert set(chunk_ids).isdisjoint(report["orphan_store"])
    assert all(mock_memory.store.get(chunk_id) is not None for chunk_id in chunk_ids)


def test_delete_parent_removes_derived_chunks(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_CHUNK_INGEST", "1")
    rec = mock_memory.save(content=_long_body(), title="Delete Parent", tags=["test"])
    mock_memory.reindex(force=True)
    chunk_ids = _chunk_ids_for(mock_memory.store, rec.id)
    assert chunk_ids

    assert mock_memory.delete(rec.id) is True

    assert mock_memory.store.chunks_by_parent_id(rec.id) == []
    assert all(mock_memory.store.get(chunk_id) is None for chunk_id in chunk_ids)


def test_reindex_long_to_short_prunes_stale_chunks(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_CHUNK_INGEST", "1")
    rec = mock_memory.save(content=_long_body(), title="Shortened Parent", tags=["test"])
    mock_memory.reindex(force=True)
    assert _chunk_ids_for(mock_memory.store, rec.id)

    mock_memory.update(rec.id, content="now short")
    mock_memory.reindex(force=True)

    assert mock_memory.store.chunks_by_parent_id(rec.id) == []


def test_disabling_chunk_ingest_prunes_existing_chunks(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_CHUNK_INGEST", "1")
    rec = mock_memory.save(content=_long_body(), title="Disabled Chunks", tags=["test"])
    mock_memory.reindex(force=True)
    assert _chunk_ids_for(mock_memory.store, rec.id)

    monkeypatch.setenv("MEMO_CHUNK_INGEST", "0")
    mock_memory.reindex(force=True)

    assert mock_memory.store.chunks_by_parent_id(rec.id) == []
