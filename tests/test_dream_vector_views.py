"""Tests for `run_title_view_pass` — deterministic title/tag view indexing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from memo.dream_vector_views import run_title_view_pass
from memo.store.hype_store import HypeStore

DIMS = 4


def _cfg(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(db_path=tmp_path / "memvec.db", embedder_dims=DIMS)


def _row(type_: str = "note", title: str = "A title", tags=None, body_hash: str = "hash1") -> dict:
    return {"type": type_, "title": title, "tags": tags or ["tag1"], "body_hash": body_hash}


def _mem(rows: dict[str, dict]) -> MagicMock:
    mem = MagicMock()
    mem.store.embedder_model = "test-model"
    mem.store.all_ids.return_value = list(rows)
    mem.store.get.side_effect = lambda mid: rows.get(mid)
    return mem


@pytest.fixture(autouse=True)
def _stub_embed(monkeypatch):
    monkeypatch.setattr(
        "memo.dream_vector_views._embed_question", lambda mem, text: [0.1, 0.2, 0.3, 0.4]
    )


def test_indexes_one_durable_memory_and_reports_done(tmp_path: Path):
    mem = _mem({"m1": _row()})

    result = run_title_view_pass(_cfg(tmp_path), mem)

    assert result["status"] == "done"
    assert result["indexed"] == 1
    assert result["errors"] == 0
    assert result["backlog"] == 1
    assert result["view_kind"] == "title"

    store = HypeStore(tmp_path / "memvec.db", DIMS)
    try:
        assert store.view_body_hash_for("m1", "title") == "hash1"
    finally:
        store.close()


def test_dry_run_counts_backlog_but_indexes_nothing(tmp_path: Path):
    mem = _mem({"m1": _row(), "m2": _row(title="Another", body_hash="hash2")})

    result = run_title_view_pass(_cfg(tmp_path), mem, dry_run=True)

    assert result["status"] == "dry_run"
    assert result["backlog"] == 2
    assert result["indexed"] == 0

    store = HypeStore(tmp_path / "memvec.db", DIMS)
    try:
        assert store.view_body_hash_for("m1", "title") is None
    finally:
        store.close()


def test_non_durable_type_is_skipped(tmp_path: Path):
    mem = _mem({"m1": _row(type_="reference")})

    result = run_title_view_pass(_cfg(tmp_path), mem)

    assert result["backlog"] == 0
    assert result["indexed"] == 0


def test_missing_record_is_skipped(tmp_path: Path):
    mem = _mem({})
    mem.store.all_ids.return_value = ["ghost"]
    mem.store.get.return_value = None

    result = run_title_view_pass(_cfg(tmp_path), mem)

    assert result["backlog"] == 0
    assert result["status"] == "done"


def test_already_indexed_same_body_hash_is_not_reindexed(tmp_path: Path):
    mem = _mem({"m1": _row()})
    run_title_view_pass(_cfg(tmp_path), mem)

    second = run_title_view_pass(_cfg(tmp_path), mem)

    assert second["backlog"] == 0
    assert second["indexed"] == 0


def test_changed_body_hash_is_reindexed(tmp_path: Path):
    mem = _mem({"m1": _row(body_hash="hash1")})
    run_title_view_pass(_cfg(tmp_path), mem)

    mem.store.get.side_effect = lambda mid: _row(body_hash="hash2")
    result = run_title_view_pass(_cfg(tmp_path), mem)

    assert result["backlog"] == 1
    assert result["indexed"] == 1


def test_empty_title_and_tags_is_skipped(tmp_path: Path):
    row = {"type": "note", "title": "", "tags": [], "body_hash": "hash1"}
    mem = _mem({"m1": row})

    result = run_title_view_pass(_cfg(tmp_path), mem)

    assert result["backlog"] == 0


def test_missing_body_hash_is_skipped(tmp_path: Path):
    mem = _mem({"m1": _row(body_hash="")})

    result = run_title_view_pass(_cfg(tmp_path), mem)

    assert result["backlog"] == 0


def test_night_cap_bounds_the_batch(tmp_path: Path):
    rows = {f"m{i}": _row(body_hash=f"hash{i}") for i in range(5)}
    mem = _mem(rows)

    result = run_title_view_pass(_cfg(tmp_path), mem, night_cap=2)

    assert result["backlog"] == 2
    assert result["indexed"] == 2


def test_per_item_embed_failure_is_counted_as_error_and_continues(tmp_path: Path, monkeypatch):
    mem = _mem({"m1": _row(body_hash="hash1"), "m2": _row(body_hash="hash2")})
    calls = []

    def flaky_embed(mem_arg, text):
        calls.append(text)
        if len(calls) == 1:
            raise RuntimeError("embed failed")
        return [0.1, 0.2, 0.3, 0.4]

    monkeypatch.setattr("memo.dream_vector_views._embed_question", flaky_embed)

    result = run_title_view_pass(_cfg(tmp_path), mem)

    assert result["errors"] == 1
    assert result["indexed"] == 1
    assert result["status"] == "done"


def test_top_level_failure_is_captured_and_store_still_closed(tmp_path: Path):
    mem = MagicMock()
    mem.store.embedder_model = "test-model"
    mem.store.all_ids.side_effect = RuntimeError("store unavailable")

    result = run_title_view_pass(_cfg(tmp_path), mem)

    assert result["status"] == "error"
    assert result["error"] == "RuntimeError: store unavailable"
