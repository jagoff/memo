"""Tests for prune_floor_candidates store method and dream pipeline integration."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from memo.store.store import VecStore


def _make_store(tmp_path: Path) -> VecStore:
    return VecStore(tmp_path / "test.db", dims=4)


def _insert_memoria(
    store: VecStore, id_: str, type_: str, days_old: int, roi_score: float, access_count: int
) -> None:
    with store._conn as cx:
        cx.execute(
            "INSERT OR REPLACE INTO meta "
            "(id, title, type, tags, path, created, updated, body_hash) "
            "VALUES (?, ?, ?, ?, ?, datetime('now', ? || ' days'), datetime('now', ? || ' days'), ?)",
            (
                id_,
                f"title-{id_}",
                type_,
                "[]",
                f"/fake/{id_}.md",
                f"-{days_old}",
                f"-{days_old}",
                f"hash-{id_}",
            ),
        )
        cx.execute(
            "INSERT OR REPLACE INTO memory_health (id, confidence, roi_score, updated_at) "
            "VALUES (?, 1.0, ?, datetime('now', ? || ' days'))",
            (id_, roi_score, f"-{days_old}"),
        )
        if access_count > 0:
            cx.execute(
                "INSERT OR REPLACE INTO access (id, access_count, last_accessed) "
                "VALUES (?, ?, datetime('now'))",
                (id_, access_count),
            )


def test_prune_floor_returns_low_roi_zero_access(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _insert_memoria(store, "aaa", "note", days_old=100, roi_score=0.10, access_count=0)
    candidates = store.prune_floor_candidates(roi_floor=0.15, min_age_days=90)
    ids = [c["id"] for c in candidates]
    assert "aaa" in ids


def test_prune_floor_excludes_accessed(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _insert_memoria(store, "bbb", "note", days_old=100, roi_score=0.10, access_count=3)
    candidates = store.prune_floor_candidates(roi_floor=0.15, min_age_days=90)
    ids = [c["id"] for c in candidates]
    assert "bbb" not in ids


def test_prune_floor_excludes_synthesis_and_reference(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _insert_memoria(store, "ccc", "synthesis", days_old=200, roi_score=0.05, access_count=0)
    _insert_memoria(store, "ddd", "reference", days_old=200, roi_score=0.05, access_count=0)
    candidates = store.prune_floor_candidates(roi_floor=0.15, min_age_days=90)
    ids = [c["id"] for c in candidates]
    assert "ccc" not in ids
    assert "ddd" not in ids


def test_prune_floor_excludes_too_recent(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _insert_memoria(store, "eee", "note", days_old=30, roi_score=0.10, access_count=0)
    candidates = store.prune_floor_candidates(roi_floor=0.15, min_age_days=90)
    ids = [c["id"] for c in candidates]
    assert "eee" not in ids


def test_prune_floor_excludes_above_floor(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _insert_memoria(store, "fff", "note", days_old=100, roi_score=0.50, access_count=0)
    candidates = store.prune_floor_candidates(roi_floor=0.15, min_age_days=90)
    ids = [c["id"] for c in candidates]
    assert "fff" not in ids


def test_prune_floor_result_has_required_keys(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _insert_memoria(store, "ggg", "note", days_old=100, roi_score=0.10, access_count=0)
    candidates = store.prune_floor_candidates(roi_floor=0.15, min_age_days=90)
    assert candidates
    c = candidates[0]
    assert "id" in c
    assert "roi_score" in c
    assert "days_old" in c


def test_prune_floor_in_dream_pipeline_archives_candidates(tmp_path: Path) -> None:
    """Integration: dream pipeline calls archive_memoria for each candidate."""
    store = _make_store(tmp_path)
    _insert_memoria(store, "zzz", "note", days_old=100, roi_score=0.10, access_count=0)

    archived = []
    mem = MagicMock()
    mem.store = store
    mem.lifecycle.archive_memoria.side_effect = lambda id_: archived.append(id_) or True

    from memo.cli_dream import _run_prune_floor

    result = _run_prune_floor(mem, roi_floor=0.15, min_age_days=90, dry_run=False)
    assert any(r["id"] == "zzz" for r in result)
    assert "zzz" in archived


def test_prune_floor_dry_run_does_not_archive(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _insert_memoria(store, "yyy", "note", days_old=100, roi_score=0.10, access_count=0)

    mem = MagicMock()
    mem.store = store

    from memo.cli_dream import _run_prune_floor

    result = _run_prune_floor(mem, roi_floor=0.15, min_age_days=90, dry_run=True)
    assert any(r["id"] == "yyy" for r in result)
    mem.lifecycle.archive_memoria.assert_not_called()
