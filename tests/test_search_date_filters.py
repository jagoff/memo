"""date_from/date_to hard window on Memory.search / VecStore.search."""
from __future__ import annotations


def _backdate(mem, id_: str, updated: str) -> None:
    mem.store._conn.execute("UPDATE meta SET updated = ? WHERE id = ?", (updated, id_))
    mem.store._conn.commit()


def test_vec_search_date_from_drops_old_rows(mock_memory):
    old = mock_memory.save(content="hecho viejo sobre zanahorias moradas " * 3, title="Viejo")
    new = mock_memory.save(content="hecho nuevo sobre zanahorias moradas " * 3, title="Nuevo")
    _backdate(mock_memory, old.id, "2026-01-01T00:00:00")
    ids = {h.id for h in mock_memory.search("zanahorias", mode="vec", date_from="2026-06-01")}
    assert new.id in ids
    assert old.id not in ids


def test_vec_search_date_to_is_end_of_day_inclusive(mock_memory):
    old = mock_memory.save(content="hecho viejo sobre zanahorias moradas " * 3, title="Viejo2")
    _backdate(mock_memory, old.id, "2026-01-15T18:30:00")
    ids = {h.id for h in mock_memory.search("zanahorias", mode="vec", date_to="2026-01-15")}
    assert old.id in ids  # bare date must include the whole day


def test_bm25_search_respects_date_window(mock_memory):
    old = mock_memory.save(content="reunión sobre flamencos rosados del proyecto " * 3, title="ViejoBM")
    new = mock_memory.save(content="reunión sobre flamencos rosados del sprint " * 3, title="NuevoBM")
    _backdate(mock_memory, old.id, "2026-01-01T00:00:00")
    ids = {h.id for h in mock_memory.search("flamencos", mode="bm25", date_from="2026-06-01")}
    assert new.id in ids
    assert old.id not in ids
