"""supersede_decision: the shared trust-margin contradiction resolver."""

from __future__ import annotations

from types import SimpleNamespace

from memo import belief


def _mem(health, support):
    store = SimpleNamespace(
        get_health_batch=lambda ids: {i: health[i] for i in ids if i in health},
        get_support_batch=lambda ids: {i: support.get(i, 0) for i in ids},
    )
    return SimpleNamespace(store=store)


def test_flags_off_is_legacy_recency_archive(monkeypatch):
    monkeypatch.setenv("MEMO_BELIEF_COMPETING", "0")
    monkeypatch.setenv("MEMO_SUPERSEDE_SUPPORT_GATE", "0")
    mem = _mem({}, {})
    d = belief.supersede_decision(mem, older_id="OLD", newer_id="NEW")
    assert d.action == "archive"
    assert d.dominated_id == "OLD"  # recency: newer wins


def test_flags_off_support_gate_holds_older(monkeypatch):
    monkeypatch.setenv("MEMO_BELIEF_COMPETING", "0")
    monkeypatch.setenv("MEMO_SUPERSEDE_SUPPORT_GATE", "3")
    mem = _mem({}, {"OLD": 5})
    d = belief.supersede_decision(mem, older_id="OLD", newer_id="NEW")
    assert d.action == "hold_open"
    assert d.support_dominated == 5


def test_competing_within_margin(monkeypatch):
    monkeypatch.setenv("MEMO_BELIEF_COMPETING", "1")
    monkeypatch.setenv("MEMO_SUPERSEDE_MARGIN", "0.15")
    # scores: OLD 0.9*1.0=0.9 ; NEW 0.85*1.0=0.85 ; |diff|=0.05 <= 0.15
    mem = _mem(
        {
            "OLD": {"confidence": 0.9, "roi_score": 1.0},
            "NEW": {"confidence": 0.85, "roi_score": 1.0},
        },
        {},
    )
    d = belief.supersede_decision(mem, older_id="OLD", newer_id="NEW")
    assert d.action == "competing"


def test_trust_dominance_archives_weaker(monkeypatch):
    monkeypatch.setenv("MEMO_BELIEF_COMPETING", "1")
    monkeypatch.setenv("MEMO_SUPERSEDE_MARGIN", "0.05")
    monkeypatch.setenv("MEMO_SUPERSEDE_SUPPORT_GATE", "0")
    # OLD 0.95 strongly dominates NEW 0.40 ; diff 0.55 > margin
    mem = _mem(
        {
            "OLD": {"confidence": 0.95, "roi_score": 1.0},
            "NEW": {"confidence": 0.40, "roi_score": 1.0},
        },
        {},
    )
    d = belief.supersede_decision(mem, older_id="OLD", newer_id="NEW")
    assert d.action == "archive"
    assert d.dominated_id == "NEW"  # the weaker NEW is archived, NOT the older
    assert d.dominant_id == "OLD"


def test_nway_competing_pairs_flags_triangle():
    # A-B, B-C, C-A : component {A,B,C} size 3 -> all three pairs competing
    pairs = [(1, "A", "B"), (2, "B", "C"), (3, "C", "A")]
    assert belief.nway_competing_pairs(pairs) == {1, 2, 3}


def test_nway_competing_pairs_ignores_lone_pair():
    pairs = [(1, "A", "B"), (2, "C", "D")]  # two separate 2-node components
    assert belief.nway_competing_pairs(pairs) == set()
