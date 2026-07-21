"""Concurrency regressions for the shared HistoryStore connection."""

from __future__ import annotations

import threading
from pathlib import Path

from memo.history import HistoryStore


def test_read_waits_for_rollback_and_does_not_observe_dirty_event(tmp_path: Path) -> None:
    history = HistoryStore(tmp_path / "history.db", device_id="local")
    inserted = threading.Event()
    release_writer = threading.Event()
    reader_started = threading.Event()
    reader_done = threading.Event()
    observed: list[dict] = []

    def _write_then_rollback() -> None:
        try:
            with history._tx() as connection:
                connection.execute(
                    "INSERT INTO events "
                    "(ts, op, record_id, title, type, device_id) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    ("2026-01-01T00:00:00Z", "save", "r1", "Dirty", "note", "local"),
                )
                inserted.set()
                if not release_writer.wait(timeout=5):
                    raise TimeoutError("test did not release writer")
                raise RuntimeError("force rollback")
        except RuntimeError:
            pass

    def _read() -> None:
        reader_started.set()
        observed.extend(history.list_recent(limit=10))
        reader_done.set()

    writer = threading.Thread(target=_write_then_rollback, daemon=True)
    reader = threading.Thread(target=_read, daemon=True)
    try:
        writer.start()
        assert inserted.wait(timeout=5)
        reader.start()
        assert reader_started.wait(timeout=5)

        read_was_blocked = not reader_done.wait(timeout=0.5)
        release_writer.set()
        writer.join(timeout=5)
        reader.join(timeout=5)

        assert read_was_blocked
        assert not writer.is_alive() and not reader.is_alive()
        assert observed == []
    finally:
        release_writer.set()
        writer.join(timeout=5)
        reader.join(timeout=5)
        history.close()
