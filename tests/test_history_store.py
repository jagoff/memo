"""Tests for HistoryStore (append-only audit log)."""

from __future__ import annotations

from pathlib import Path

from memo.history import HistoryStore


def _store(tmp_path: Path) -> HistoryStore:
    return HistoryStore(tmp_path / "history.db")


def test_log_and_list_recent(tmp_path: Path) -> None:
    h = _store(tmp_path)
    h.log_save(ts="2026-01-01T00:00:00Z", record_id="r1", title="A", type_="note")
    h.log_update(ts="2026-01-01T00:01:00Z", record_id="r1", title="A2", type_="note",
                 delta={"title": ("A", "A2")})
    h.log_delete(ts="2026-01-01T00:02:00Z", record_id="r1", title="A2", type_="note")
    assert h.count() == 3
    rows = h.list_recent(limit=10)
    assert [r["op"] for r in rows] == ["delete", "update", "save"]  # newest first


def test_list_recent_filters(tmp_path: Path) -> None:
    h = _store(tmp_path)
    h.log_save(ts="2026-01-01T00:00:00Z", record_id="r1", title="A", type_="note")
    h.log_save(ts="2026-01-01T00:00:01Z", record_id="r2", title="B", type_="note")
    assert len(h.list_recent(op="save")) == 2
    assert len(h.list_recent(record_id="r1")) == 1


def test_provenance_roundtrips(tmp_path: Path) -> None:
    h = _store(tmp_path)
    h.log_save(ts="2026-01-01T00:00:00Z", record_id="r1", title="A", type_="note",
               provenance={"synapse_trace_id": "t123"})
    row = h.list_recent(record_id="r1")[0]
    assert row["delta"]["_provenance"]["synapse_trace_id"] == "t123"


def test_error_count_starts_zero(tmp_path: Path) -> None:
    h = _store(tmp_path)
    assert h.error_count == 0
