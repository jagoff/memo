from __future__ import annotations

import time
from datetime import UTC, datetime
from os import environ
from types import SimpleNamespace

import pytest

from memo.runtime import sleep_cycle
from memo.runtime.sleep_cycle import _get_last_activity


def test_last_activity_treats_naive_store_timestamp_as_utc(tmp_path, monkeypatch) -> None:
    if not hasattr(time, "tzset"):
        pytest.skip("requires time.tzset")

    old_tz = environ.get("TZ")
    try:
        monkeypatch.setenv("TZ", "America/Argentina/Cordoba")
        time.tzset()

        updated = "2026-01-01T00:00:00"
        mem = SimpleNamespace(
            store=SimpleNamespace(list_recent=lambda limit: [{"updated": updated}])
        )
        cfg = SimpleNamespace(state_dir=tmp_path)

        assert _get_last_activity(mem, cfg) == datetime(2026, 1, 1, tzinfo=UTC).timestamp()
    finally:
        if old_tz is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", old_tz)
        time.tzset()


def test_run_sleep_cycle_preserves_zero_interval_and_threshold(tmp_path, monkeypatch) -> None:
    calls: list[str] = []
    wait_calls: list[float | None] = []

    class _FakeMemory:
        def __init__(self, cfg):
            self.cfg = cfg

        def synthesize_cross_cluster(self):
            calls.append("synthesize")

        def consolidate(self):
            calls.append("consolidate")

        def close(self):
            calls.append("close")

    class _FakeEvent:
        def __init__(self) -> None:
            self._set = False

        def is_set(self) -> bool:
            return self._set

        def set(self) -> None:
            self._set = True

        def wait(self, timeout: float | None = None) -> bool:
            wait_calls.append(timeout)
            self._set = True
            return True

    monkeypatch.setattr(sleep_cycle.Config, "from_env", lambda: SimpleNamespace(state_dir=tmp_path))
    monkeypatch.setattr(sleep_cycle, "Memory", _FakeMemory)
    monkeypatch.setattr(sleep_cycle, "_get_last_activity", lambda mem, cfg: 0.0)
    monkeypatch.setattr(sleep_cycle.time, "time", lambda: 0.0)
    monkeypatch.setattr(sleep_cycle.signal, "signal", lambda *a, **k: None)
    monkeypatch.setattr(sleep_cycle.threading, "Event", _FakeEvent)
    monkeypatch.setattr(
        sleep_cycle,
        "flag_int",
        lambda name: (
            0
            if name
            in {
                "MEMO_MAINT_SLEEP_CYCLE_INTERVAL",
                "MEMO_MAINT_IDLE_THRESHOLD_SECS",
            }
            else None
        ),
    )
    monkeypatch.setattr(sleep_cycle, "flag_bool", lambda name: False)

    sleep_cycle.run_sleep_cycle(debug=False)

    assert calls == ["synthesize", "consolidate", "close"]
    assert wait_calls == [0.0]
