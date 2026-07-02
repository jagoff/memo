"""Tests for the orientation summary pass in dream run."""

from __future__ import annotations

from unittest.mock import MagicMock

from memo.cli_dream import _build_orientation


def _make_mock_mem():
    mem = MagicMock()
    conn = MagicMock()
    mem.store._conn = conn
    mem.contradict_store.list_open.return_value = []
    return mem, conn


def test_orientation_returns_required_keys():
    mem, conn = _make_mock_mem()

    mock_row = MagicMock()
    mock_row.__getitem__ = MagicMock(return_value=0)
    conn.execute.return_value.fetchone.return_value = mock_row
    conn.execute.return_value.fetchall.return_value = []

    result = _build_orientation(mem)
    assert "total" in result
    assert "by_type" in result
    assert "low_roi" in result
    assert "stale_candidates" in result
    assert "open_contradictions" in result
    assert "unindexed_entities" in result


def test_orientation_open_contradictions_counted():
    mem, conn = _make_mock_mem()
    mock_row = MagicMock()
    mock_row.__getitem__ = MagicMock(return_value=0)
    conn.execute.return_value.fetchone.return_value = mock_row
    conn.execute.return_value.fetchall.return_value = []
    mem.contradict_store.list_open.return_value = [MagicMock(), MagicMock()]

    result = _build_orientation(mem)
    assert result["open_contradictions"] == 2
