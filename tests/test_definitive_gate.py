from __future__ import annotations

from memo.definitive import (
    definitive_check,
    independence_audit,
    run_journal_benchmark,
)


def test_independence_audit_proves_runtime_is_native() -> None:
    report = independence_audit()
    assert report == {
        "ok": True,
        "forbidden_imports": [],
        "retired_modules_present": [],
    }


def test_definitive_check_covers_required_capabilities(mock_memory) -> None:
    report = definitive_check(mock_memory)
    assert report["ok"] is True
    assert all(report["checks"].values())
    assert all(report["capabilities"].values())
    assert set(report["contracts"]) == {
        "event",
        "evidence",
        "operational",
        "federation",
    }


def test_journal_benchmark_is_reproducible_and_verified() -> None:
    report = run_journal_benchmark(events=50, min_events_per_second=1.0)
    assert report["ok"] is True
    assert report["events"] == 50
    assert report["events_per_second"] > 1.0
    assert report["verification"]["events"] == 50
