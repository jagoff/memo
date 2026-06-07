"""Tests for the `timer` decorator (memo/perf.py)."""

from __future__ import annotations

import logging
import time

from memo.perf import timer


def test_timer_preserves_name_and_signature() -> None:
    @timer()
    def add(a: int, b: int) -> int:
        """Adds."""
        return a + b

    assert add.__name__ == "add"
    assert add.__doc__ == "Adds."
    assert add(2, 3) == 5


def test_timer_logs_when_over_threshold(caplog) -> None:
    @timer(name="slow", log_threshold_ms=0.0, level=logging.WARNING)
    def slow() -> str:
        time.sleep(0.001)
        return "ok"

    with caplog.at_level(logging.WARNING, logger="memo.perf"):
        assert slow() == "ok"
    assert any("slow took" in r.message for r in caplog.records)


def test_timer_silent_under_threshold(caplog) -> None:
    @timer(name="fast", log_threshold_ms=10_000.0)
    def fast() -> int:
        return 1

    with caplog.at_level(logging.DEBUG, logger="memo.perf"):
        assert fast() == 1
    assert not [r for r in caplog.records if "fast took" in r.message]


def test_timer_logs_even_when_wrapped_raises(caplog) -> None:
    @timer(name="boom", log_threshold_ms=0.0, level=logging.WARNING)
    def boom() -> None:
        raise ValueError("nope")

    with caplog.at_level(logging.WARNING, logger="memo.perf"):
        try:
            boom()
        except ValueError:
            pass
    assert any("boom took" in r.message for r in caplog.records)
