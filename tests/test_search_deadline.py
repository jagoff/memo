"""The deadline primitive. Monotonic, unlimited-safe, and cheap enough to
consult before every stage."""

from __future__ import annotations

import math
import time

from memo.search_deadline import Deadline


def test_unlimited_budget_never_expires() -> None:
    d = Deadline.start(0)
    assert d.unlimited
    assert not d.expired
    assert math.isinf(d.remaining_ms())
    assert d.afford(10_000_000)


def test_remaining_shrinks_monotonically() -> None:
    d = Deadline.start(1000)
    first = d.remaining_ms()
    time.sleep(0.01)
    assert d.remaining_ms() < first
    assert d.remaining_ms() <= 1000


def test_expired_after_the_budget() -> None:
    d = Deadline.start(5)
    time.sleep(0.02)
    assert d.expired


def test_afford_refuses_a_stage_that_will_not_fit() -> None:
    d = Deadline.start(100)
    assert d.afford(10)
    assert not d.afford(10_000)


def test_start_reads_the_flag_when_unset(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_SEARCH_BUDGET_MS", "1234")
    assert Deadline.start().budget_ms == 1234
