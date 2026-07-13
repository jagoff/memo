import datetime as dt

import pytest

from memo import dream_distill as dd


def _now() -> dt.datetime:
    return dt.datetime(2026, 7, 13, tzinfo=dt.UTC)


def _member(created: str, confidence: float, support: int) -> dict:
    return {"id": "x" * 32, "created": created, "confidence": confidence, "support_count": support}


def test_provenance_hash_is_stable_and_order_independent():
    a = dd.provenance_hash(["b", "a", "c"])
    b = dd.provenance_hash(["c", "b", "a"])
    assert a == b
    assert len(a) == 16


def test_cluster_maturity_aggregates():
    members = [
        _member("2026-06-01", 0.9, 4),   # ~42 days old
        _member("2026-06-20", 0.7, 2),   # ~23 days old (youngest)
    ]
    stats = dd.cluster_maturity(members, now=_now())
    assert stats.size == 2
    assert stats.mean_support == pytest.approx(3.0)
    assert stats.mean_confidence == pytest.approx(0.8)
    # youngest member drives min_age_days
    assert stats.min_age_days == pytest.approx(23.0, abs=1.0)


def test_is_mature_true_when_all_floors_clear():
    members = [_member("2026-06-01", 0.9, 4), _member("2026-06-01", 0.8, 3)]
    stats = dd.cluster_maturity(members, now=_now())
    assert dd.is_mature(stats, min_cluster=2, min_support=2, min_confidence=0.5, min_age_days=14) is True


def test_is_mature_false_when_too_young():
    members = [_member("2026-07-10", 0.9, 4), _member("2026-07-11", 0.9, 4)]  # ~2-3 days old
    stats = dd.cluster_maturity(members, now=_now())
    assert dd.is_mature(stats, min_cluster=2, min_support=2, min_confidence=0.5, min_age_days=14) is False


def test_is_mature_false_when_under_supported():
    members = [_member("2026-06-01", 0.9, 1), _member("2026-06-01", 0.9, 1)]  # mean support 1 < 2
    stats = dd.cluster_maturity(members, now=_now())
    assert dd.is_mature(stats, min_cluster=2, min_support=2, min_confidence=0.5, min_age_days=14) is False


def test_missing_created_is_treated_as_fresh():
    members = [_member("", 0.9, 4), _member("not-a-date", 0.9, 4)]
    stats = dd.cluster_maturity(members, now=_now())
    assert stats.min_age_days == 0.0  # conservative: unknown age => fresh => fails age floor


def test_corroboration_weighted_confidence():
    hi = dd.cluster_maturity([_member("2026-06-01", 0.9, 4)], now=_now())
    md = dd.cluster_maturity([_member("2026-06-01", 0.6, 2)], now=_now())
    lo = dd.cluster_maturity([_member("2026-06-01", 0.3, 1)], now=_now())
    assert dd.corroboration_weighted_confidence(hi) == "high"
    assert dd.corroboration_weighted_confidence(md) == "medium"
    assert dd.corroboration_weighted_confidence(lo) == "low"
