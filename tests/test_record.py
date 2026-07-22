"""MemoryRecord temporal fields (valid_at/invalid_at) — save→get row round-trip.

Task 2 of record-level bi-temporal validity: ``save()`` passes an explicit
``valid_at`` through to the store row, and ``get()`` reads it back via
``record_from_row``. ``invalid_at`` defaults to ``None`` (interval still open).
"""

from memo.memory import Memory


def test_record_roundtrips_valid_time(mock_memory: Memory) -> None:
    rec = mock_memory.save(
        content="prod db is postgres",
        type_="fact",
        valid_at="2026-06-01T00:00:00",
    )
    got = mock_memory.get(rec.id)
    assert got is not None
    assert got.valid_at == "2026-06-01T00:00:00"
    assert got.invalid_at is None
