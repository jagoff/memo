"""Memory.around: seq-adjacent siblings for chunks, created-time window for durables."""
from __future__ import annotations


def _chunk(mem, seq: int):
    return mem.save(
        content=f"contenido del chunk número {seq} con texto suficiente para pasar el gate de ruido",
        title=f"Doc §{seq}/9",
        type_="reference",
        tags=["chunk"],
        extra={"parent_path": "notes/doc.md", "chunk_seq": seq},
    )


def test_chunks_adjacent_returns_seq_window(mock_memory):
    recs = {seq: _chunk(mock_memory, seq) for seq in range(1, 8)}
    rows = mock_memory.store.chunks_adjacent("notes/doc.md", 4, before=2, after=2)
    assert [r["extra"]["chunk_seq"] for r in rows] == [2, 3, 4, 5, 6]


def test_around_chunk_mode_excludes_anchor(mock_memory):
    recs = {seq: _chunk(mock_memory, seq) for seq in range(1, 6)}
    out = mock_memory.around(recs[3].id, before=1, after=1)
    assert out["mode"] == "chunk_seq"
    assert out["anchor"]["id"] == recs[3].id
    assert {n["extra"]["chunk_seq"] for n in out["neighbors"]} == {2, 4}


def test_around_durable_mode_uses_created_window(mock_memory):
    a = mock_memory.save(content="primer hecho del lunes sobre el deploy " * 2,
                         title="A", created="2026-01-01T10:00:00")
    b = mock_memory.save(content="segundo hecho del martes sobre el deploy " * 2,
                         title="B", created="2026-01-02T10:00:00")
    c = mock_memory.save(content="tercer hecho del miércoles sobre el deploy " * 2,
                         title="C", created="2026-01-03T10:00:00")
    out = mock_memory.around(b.id, before=1, after=1)
    assert out["mode"] == "created"
    assert [n["id"] for n in out["neighbors"]] == [a.id, c.id]
    assert all("body_snippet" in n for n in out["neighbors"])


def test_around_unknown_id_returns_empty(mock_memory):
    out = mock_memory.around("ffffffff" * 4)
    assert out == {"anchor": None, "mode": None, "neighbors": []}
