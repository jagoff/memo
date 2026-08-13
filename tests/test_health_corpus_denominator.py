"""`memo health` counts the same corpus `memo stats` and `memo doctor` count.

`health` used a raw `SELECT COUNT(*) FROM meta`, which includes soft-deleted
rows; `stats` and `doctor` use `store.count()`, which excludes them. On the
developer's machine 2026-08-09 that read as three numbers for one corpus —
health 13,050, stats 10,742, doctor 10,742 — with nothing on screen explaining
that the 2,338-row gap was records the user had deleted.

The deleted rows are still worth surfacing, so they get their own number rather
than being folded into the headline one. `archived` stays what it always was —
a count of `.md` files under `memory_dir/archived/`, a DISK figure that is not a
subset of the index count it sits beside — and the CLI line now says so.
"""

from __future__ import annotations

from typing import Any

from memo.health_report import build_health_report


def _save(memory: Any, content: str) -> str:
    record = memory.save(content=content, type_="note")
    return str(record["id"] if isinstance(record, dict) else record.id)


def test_health_matches_store_count(mock_memory: Any) -> None:
    _save(mock_memory, "kept one")
    _save(mock_memory, "kept two")

    report = build_health_report(mock_memory)

    assert report["corpus"]["memories"] == mock_memory.store.count() == 2


def test_soft_deleted_rows_leave_the_headline_count(mock_memory: Any) -> None:
    _save(mock_memory, "kept")
    mock_memory.delete(_save(mock_memory, "deleted"))

    report = build_health_report(mock_memory)

    assert report["corpus"]["memories"] == mock_memory.store.count() == 1


def test_soft_deleted_rows_are_reported_separately(mock_memory: Any) -> None:
    """Dropped from the headline, not hidden."""
    _save(mock_memory, "kept")
    mock_memory.delete(_save(mock_memory, "deleted"))

    report = build_health_report(mock_memory)

    assert report["corpus"]["soft_deleted"] >= 0
    assert report["corpus"]["memories"] + report["corpus"]["soft_deleted"] >= 2


def test_a_clean_corpus_reports_no_soft_deleted(mock_memory: Any) -> None:
    _save(mock_memory, "kept")

    assert build_health_report(mock_memory)["corpus"]["soft_deleted"] == 0
