"""Tests for the dream per-phase instrumentation + checkpoint harness."""

from __future__ import annotations

import json
from typing import Any

from memo.dream_phases import (
    DreamCheckpoint,
    PhaseRecorder,
    _coerce_count,
    _infer_counts,
    summarize_phases,
)


class _FakeConn:
    def __init__(self, count: int, ts: str) -> None:
        self._count = count
        self._ts = ts

    def execute(self, _sql: str) -> _FakeConn:
        return self

    def fetchone(self) -> tuple[int, str]:
        return (self._count, self._ts)


class _FakeStore:
    def __init__(self, count: int, ts: str) -> None:
        self._conn = _FakeConn(count, ts)


class _FakeMem:
    """Minimal stand-in exposing what ``_corpus_fingerprint`` reads."""

    def __init__(self, count: int, ts: str) -> None:
        self.store = _FakeStore(count, ts)

    def set_fp(self, count: int, ts: str) -> None:
        self.store = _FakeStore(count, ts)


def test_coerce_count_handles_lists_ints_and_bools() -> None:
    assert _coerce_count([1, 2, 3]) == 3
    assert _coerce_count(7) == 7
    assert _coerce_count(True) == 1
    assert _coerce_count("nope") == 0
    assert _coerce_count(None) == 0


def test_infer_counts_reads_maintenance_vocabulary() -> None:
    frag = {"scanned": 50, "merged": [1, 2], "archived": 3}
    inputs, mutations = _infer_counts(frag)
    assert inputs == 50
    assert mutations == 5  # len([1,2]) + 3


def test_infer_counts_non_dict_is_zero() -> None:
    assert _infer_counts(None) == (0, 0)
    assert _infer_counts([1, 2]) == (0, 0)


def test_recorder_builds_structured_record() -> None:
    receipt: dict[str, Any] = {"errors": []}
    rec = PhaseRecorder(receipt)
    ph = rec.begin("hype", fragment_key="hype")
    receipt["hype"] = {"scanned": 50, "saved": [1, 2, 3]}
    ph.skipped_count = 47
    rec.end(ph)

    assert len(receipt["phases"]) == 1
    r = receipt["phases"][0]
    # spec-mandated fields all present
    for key in (
        "phase",
        "status",
        "duration_ms",
        "input_count",
        "changed_count",
        "skipped_count",
        "mutations",
        "errors",
        "warnings",
        "quality_before",
        "quality_after",
        "in_fingerprint",
        "out_fingerprint",
    ):
        assert key in r
    assert r["phase"] == "hype"
    assert r["status"] == "done"
    assert r["input_count"] == 50
    assert r["mutations"] == 3
    assert r["changed_count"] == 3
    assert r["skipped_count"] == 47
    assert r["duration_ms"] >= 0.0


def test_recorder_error_status_from_receipt_errors_delta() -> None:
    receipt: dict[str, Any] = {"errors": []}
    rec = PhaseRecorder(receipt)
    ph = rec.begin("contradict")
    receipt["errors"].append("contradict: boom")
    rec.end(ph)
    r = receipt["phases"][0]
    assert r["status"] == "error"
    assert r["errors"] == ["contradict: boom"]


def test_recorder_error_status_from_fragment_status() -> None:
    receipt: dict[str, Any] = {"errors": []}
    rec = PhaseRecorder(receipt)
    ph = rec.begin("tuner", fragment_key="tuner")
    receipt["tuner"] = {"status": "error", "error": "nope"}
    rec.end(ph)
    assert receipt["phases"][0]["status"] == "error"


def test_recorder_fingerprint_change_detection() -> None:
    mem = _FakeMem(10, "2026-08-02T00:00:00")
    receipt: dict[str, Any] = {"errors": []}
    rec = PhaseRecorder(receipt, mem=mem)
    ph = rec.begin("consolidate")
    mem.set_fp(9, "2026-08-02T01:00:00")  # a merge dropped a row
    rec.end(ph)
    r = receipt["phases"][0]
    assert r["in_fingerprint"] == "10:2026-08-02T00:00:00"
    assert r["out_fingerprint"] == "9:2026-08-02T01:00:00"
    assert r["changed_count"] == 1  # fp moved => at least one change


def test_recorder_end_is_idempotent_and_safe() -> None:
    receipt: dict[str, Any] = {"errors": []}
    rec = PhaseRecorder(receipt)
    ph = rec.begin("x")
    rec.end(ph)
    rec.end(ph)  # second call must not append a duplicate
    assert len(receipt["phases"]) == 1


def test_explicit_status_override_wins() -> None:
    receipt: dict[str, Any] = {"errors": []}
    rec = PhaseRecorder(receipt)
    ph = rec.begin("skippy")
    ph.status = "skipped"
    rec.end(ph)
    assert receipt["phases"][0]["status"] == "skipped"


def test_checkpoint_persist_and_reload(tmp_path: Any) -> None:
    path = tmp_path / "dream" / "checkpoint.json"
    ck = DreamCheckpoint(path, run_fingerprint="fp-a")
    ck.record("hype", {"phase": "hype", "status": "done"}, {"saved": [1]})
    assert path.exists()

    reloaded = DreamCheckpoint(path, run_fingerprint="fp-a")
    assert reloaded.resumable() is True
    assert reloaded.is_done("hype") is True
    assert reloaded.fragment("hype") == {"saved": [1]}
    assert reloaded.phase_record("hype")["status"] == "done"


def test_checkpoint_fingerprint_mismatch_not_resumable(tmp_path: Any) -> None:
    path = tmp_path / "checkpoint.json"
    DreamCheckpoint(path, "fp-a").record("hype", {"phase": "hype"}, None)
    # a new run with a different previous-corpus fingerprint must start clean
    fresh = DreamCheckpoint(path, run_fingerprint="fp-b")
    assert fresh.resumable() is False
    assert fresh.is_done("hype") is False


def test_checkpoint_clear_removes_file(tmp_path: Any) -> None:
    path = tmp_path / "checkpoint.json"
    ck = DreamCheckpoint(path, "fp-a")
    ck.record("hype", {"phase": "hype"}, None)
    ck.clear()
    assert not path.exists()
    assert ck.resumable() is False


def test_checkpoint_corrupt_file_is_ignored(tmp_path: Any) -> None:
    path = tmp_path / "checkpoint.json"
    path.write_text("{ not json", encoding="utf-8")
    ck = DreamCheckpoint(path, "fp-a")
    assert ck.resumable() is False  # unreadable => fresh, no crash


def test_resume_restores_fragment_and_skips_work(tmp_path: Any) -> None:
    path = tmp_path / "checkpoint.json"
    ck = DreamCheckpoint(path, "fp-a")
    ck.record("hype", {"phase": "hype", "status": "done", "mutations": 3}, {"saved": [1, 2, 3]})

    receipt: dict[str, Any] = {"errors": []}
    rec = PhaseRecorder(receipt, checkpoint=DreamCheckpoint(path, "fp-a"), resume=True)
    ph = rec.begin("hype", fragment_key="hype")
    ran = False
    if rec.restore(ph):
        pass  # skipped
    else:
        ran = True
    rec.end(ph)

    assert ran is False  # work was skipped
    assert receipt["hype"] == {"saved": [1, 2, 3]}  # fragment restored
    assert receipt["phases"][0]["resumed"] is True
    assert ph.restored is True


def test_resume_off_does_not_skip(tmp_path: Any) -> None:
    path = tmp_path / "checkpoint.json"
    ck = DreamCheckpoint(path, "fp-a")
    ck.record("hype", {"phase": "hype"}, {"saved": [1]})
    receipt: dict[str, Any] = {"errors": []}
    # resume=False (default nightly behavior): never skip
    rec = PhaseRecorder(receipt, checkpoint=DreamCheckpoint(path, "fp-a"), resume=False)
    ph = rec.begin("hype", fragment_key="hype")
    assert rec.restore(ph) is False


def test_checkpoint_written_after_each_phase(tmp_path: Any) -> None:
    path = tmp_path / "checkpoint.json"
    receipt: dict[str, Any] = {"errors": []}
    ck = DreamCheckpoint(path, "fp-a")
    rec = PhaseRecorder(receipt, checkpoint=ck)
    rec.end(rec.begin("a"))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "a" in data["done"]  # partial receipt survives an interruption here


def test_timed_runs_thunk_and_records() -> None:
    receipt: dict[str, Any] = {"errors": []}
    rec = PhaseRecorder(receipt)
    calls: list[int] = []

    def _run() -> dict[str, Any]:
        calls.append(1)
        return {"scanned": 5, "saved": [1, 2]}

    receipt["hype"] = rec.timed("hype", _run, fragment_key="hype")
    assert calls == [1]
    assert receipt["hype"] == {"scanned": 5, "saved": [1, 2]}
    r = receipt["phases"][0]
    assert r["phase"] == "hype"
    assert r["input_count"] == 5
    assert r["mutations"] == 2


def test_timed_error_fragment_marks_error() -> None:
    receipt: dict[str, Any] = {"errors": []}
    rec = PhaseRecorder(receipt)
    receipt["t"] = rec.timed("t", lambda: {"status": "error"}, fragment_key="t")
    assert receipt["phases"][0]["status"] == "error"


def test_timed_exception_records_error_and_propagates(tmp_path: Any) -> None:
    receipt: dict[str, Any] = {"errors": []}
    rec = PhaseRecorder(receipt)

    def _boom() -> dict[str, Any]:
        raise RuntimeError("kaboom")

    try:
        rec.timed("boom", _boom)
    except RuntimeError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("timed must propagate the pass exception")
    # a phase record is still emitted for the crashed pass
    assert receipt["phases"][0]["phase"] == "boom"


def test_timed_resumable_skips_thunk_and_returns_cached(tmp_path: Any) -> None:
    path = tmp_path / "checkpoint.json"
    ck = DreamCheckpoint(path, "fp-a")
    ck.record("hype", {"phase": "hype", "status": "done"}, {"saved": [1, 2, 3]})

    receipt: dict[str, Any] = {"errors": []}
    rec = PhaseRecorder(receipt, checkpoint=DreamCheckpoint(path, "fp-a"), resume=True)
    calls: list[int] = []
    frag = rec.timed(
        "hype", lambda: calls.append(1) or {"saved": []}, fragment_key="hype", resumable=True
    )
    assert calls == []  # thunk never ran
    assert frag == {"saved": [1, 2, 3]}  # cached fragment returned
    assert receipt["phases"][0]["resumed"] is True


def test_summarize_phases() -> None:
    receipt = {
        "phases": [
            {"phase": "a", "duration_ms": 10.0, "mutations": 2, "errors": []},
            {"phase": "b", "duration_ms": 40.0, "mutations": 0, "errors": ["x"], "resumed": True},
        ]
    }
    s = summarize_phases(receipt)
    assert s["count"] == 2
    assert s["total_duration_ms"] == 50.0
    assert s["mutations"] == 2
    assert s["errors"] == 1
    assert s["resumed"] == 1
    assert s["slowest"]["phase"] == "b"
