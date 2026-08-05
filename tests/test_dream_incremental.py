"""Fase 5 — per-pass incremental skip (dependency-fingerprint gating).

Hermetic: fingerprint state is a small JSON sidecar under a tmp_path/dream/; the
durable-content fingerprint reads one sqlite aggregate through a fake store.
"""

from __future__ import annotations

from pathlib import Path

from memo import dream_incremental as inc


class _FakeConn:
    def __init__(self, row):
        self._row = row

    def execute(self, _sql, _params=None):
        return self

    def fetchone(self):
        return self._row


class _FakeStore:
    def __init__(self, row):
        self._conn = _FakeConn(row)


class _FakeMem:
    def __init__(self, row):
        self.store = _FakeStore(row)


def test_durable_content_fingerprint_shape():
    mem = _FakeMem((3, "2026-01-01T00:00:00"))
    assert inc.durable_content_fingerprint(mem) == "3:2026-01-01T00:00:00"


def test_durable_content_fingerprint_none_on_store_error():
    class _Boom:
        class store:
            class _conn:
                @staticmethod
                def execute(*_a, **_k):
                    raise RuntimeError("db gone")

    assert inc.durable_content_fingerprint(_Boom()) is None


def test_should_skip_only_on_exact_match(tmp_path: Path):
    assert inc.should_skip(tmp_path, "entities", "abc") is False  # nothing stored
    inc.record_success(tmp_path, "entities", "abc")
    assert inc.should_skip(tmp_path, "entities", "abc") is True
    assert inc.should_skip(tmp_path, "entities", "def") is False  # dependency moved
    assert inc.should_skip(tmp_path, "synthesis", "abc") is False  # per-pass keyed


def test_none_fingerprint_never_skips_and_clears(tmp_path: Path):
    inc.record_success(tmp_path, "entities", "abc")
    assert inc.should_skip(tmp_path, "entities", None) is False  # fail safe = run
    inc.record_success(tmp_path, "entities", None)  # clears the stored value
    assert inc.should_skip(tmp_path, "entities", "abc") is False


def test_run_or_skip_runs_then_skips_on_unchanged(tmp_path: Path):
    calls = []

    def runner():
        calls.append(1)
        return {"status": "done", "extracted": 5}

    r1 = inc.run_or_skip(tmp_path, "entities", "fp1", runner)
    assert r1["extracted"] == 5
    assert len(calls) == 1

    r2 = inc.run_or_skip(tmp_path, "entities", "fp1", runner)
    assert r2["status"] == "skipped_incremental"
    assert r2["fingerprint"] == "fp1"
    assert len(calls) == 1  # runner NOT called again

    r3 = inc.run_or_skip(tmp_path, "entities", "fp2", runner)  # dependency changed
    assert r3["status"] == "done"
    assert len(calls) == 2


def test_run_or_skip_does_not_stamp_on_error(tmp_path: Path):
    def failing():
        return {"status": "error", "error": "boom"}

    inc.run_or_skip(tmp_path, "entities", "fp1", failing)
    # errored run must NOT be recorded → next run re-runs, not skips
    assert inc.should_skip(tmp_path, "entities", "fp1") is False


def test_clear_forgets_everything(tmp_path: Path):
    inc.record_success(tmp_path, "entities", "abc")
    inc.record_success(tmp_path, "synthesis", "xyz")
    inc.clear(tmp_path)
    assert inc.should_skip(tmp_path, "entities", "abc") is False
    assert inc.should_skip(tmp_path, "synthesis", "xyz") is False
