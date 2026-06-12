"""Temporal decay on feedback boosts: recent votes weigh more than old ones.

learn-to-rank from feedback already feeds ranking (rerank_ops boosts), but a
year-old thumbs_up counted the same as today's. A half-life on the vote's age
fades stale positive feedback. Hard thumbs_down (exclusion) does not decay.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from memo.memory.rerank_ops import _feedback_recency_weight


def test_recency_weight_fresh_is_one():
    now = datetime.now(tz=UTC)
    w = _feedback_recency_weight(now.isoformat(), halflife_days=180.0, now=now)
    assert abs(w - 1.0) < 0.01


def test_recency_weight_halves_at_one_halflife():
    now = datetime.now(tz=UTC)
    old = (now - timedelta(days=180)).isoformat()
    w = _feedback_recency_weight(old, halflife_days=180.0, now=now)
    assert abs(w - 0.5) < 0.01


def test_recency_weight_disabled_when_halflife_zero():
    now = datetime.now(tz=UTC)
    old = (now - timedelta(days=10000)).isoformat()
    assert _feedback_recency_weight(old, halflife_days=0.0, now=now) == 1.0


def test_recency_weight_unparseable_is_one():
    now = datetime.now(tz=UTC)
    assert _feedback_recency_weight("not-a-date", halflife_days=180.0, now=now) == 1.0
