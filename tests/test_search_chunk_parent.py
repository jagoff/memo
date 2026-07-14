"""Chunk-hit → parent mapping in explicit search (MEMO_SEARCH_CHUNK_PARENT)."""

from __future__ import annotations


def _long_body(n_sections: int = 3, words_per_section: int = 300) -> str:
    sections = []
    for i in range(1, n_sections + 1):
        sections.append(f"## Section {i}\n\n" + (f"tokenword{i} " * words_per_section).strip())
    return "\n\n".join(sections)


def _chunked_note(mem, monkeypatch):
    monkeypatch.setenv("MEMO_CHUNK_INGEST", "1")
    rec = mem.save(content=_long_body(), title="Chunked Note", tags=["test"])
    mem.reindex(force=True)
    return rec


def test_default_off_chunks_surface_as_chunks(mock_memory, monkeypatch):
    _chunked_note(mock_memory, monkeypatch)
    monkeypatch.delenv("MEMO_SEARCH_CHUNK_PARENT", raising=False)
    hits = mock_memory.search("tokenword2", mode="bm25", limit=10)
    assert any("_chunk_" in h.id for h in hits)


def test_flag_on_maps_chunk_to_parent_and_dedups(mock_memory, monkeypatch):
    rec = _chunked_note(mock_memory, monkeypatch)
    monkeypatch.setenv("MEMO_SEARCH_CHUNK_PARENT", "1")
    hits = mock_memory.search("tokenword2", mode="bm25", limit=10)
    ids = [h.id for h in hits]
    assert rec.id in ids
    assert not any("_chunk_" in i for i in ids)  # no fragment survives
    assert ids.count(rec.id) == 1  # parent surfaced exactly once


def test_flag_on_mapped_parent_keeps_a_score(mock_memory, monkeypatch):
    rec = _chunked_note(mock_memory, monkeypatch)
    monkeypatch.setenv("MEMO_SEARCH_CHUNK_PARENT", "1")
    hits = mock_memory.search("tokenword2", mode="bm25", limit=10)
    parent = next(h for h in hits if h.id == rec.id)
    assert parent.score is not None


def test_flag_on_explicit_reference_ask_keeps_chunks(mock_memory, monkeypatch):
    _chunked_note(mock_memory, monkeypatch)
    monkeypatch.setenv("MEMO_SEARCH_CHUNK_PARENT", "1")
    hits = mock_memory.search("tokenword2", mode="bm25", limit=10, type_="reference")
    assert any("_chunk_" in h.id for h in hits)  # explicit tier ask wins


def test_flag_on_emits_trace_stage(mock_memory, monkeypatch):
    _chunked_note(mock_memory, monkeypatch)
    monkeypatch.setenv("MEMO_SEARCH_CHUNK_PARENT", "1")
    res = mock_memory.search_with_trace("tokenword2", mode="bm25", limit=10)
    assert any(t["stage"] == "chunk_parent" for t in res["trace"])
