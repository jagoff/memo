from dataclasses import dataclass, field

import pytest

from memo import recall_logic as rl
from memo.recall_logic import RankKnobs


@dataclass(frozen=True)
class _Hit:
    id: str
    score: float | None
    type: str = "note"
    extra: dict = field(default_factory=dict)
    title: str = "t"
    body: str = "body body body body body"
    tags: list = field(default_factory=list)


def _distilled(id_, score):
    return _Hit(id_, score, type="synthesis", extra={"synthesis_kind": "distillation"})


def test_is_broad_query():
    assert rl._is_broad_query("what about auth") is True
    assert (
        rl._is_broad_query("MEMO_RECALL_MIN_SIM default value here today") is False
    )  # long + identifier
    assert rl._is_broad_query("commit a1b2c3d4") is False  # id token
    assert rl._is_broad_query(None) is False
    assert rl._is_broad_query("") is False


def test_altitude_boosts_distilled_on_broad_query():
    hits = [_Hit("src", 0.80), _distilled("dist", 0.70)]
    out = rl._apply_altitude_boost(hits, 0.25, broad=True)
    # distilled 0.70 + 0.25 = 0.95 now outranks the 0.80 source
    assert out[0].id == "dist"
    assert out[0].score == pytest.approx(0.95)


def test_altitude_noop_on_specific_query():
    hits = [_Hit("src", 0.80), _distilled("dist", 0.70)]
    out = rl._apply_altitude_boost(hits, 0.25, broad=False)
    assert [h.id for h in out] == ["src", "dist"]  # order unchanged
    assert out[1].score == pytest.approx(0.70)  # not boosted


def test_altitude_ignores_non_distillation_synthesis():
    community = _Hit("comm", 0.70, type="synthesis", extra={"synthesis_kind": "community"})
    out = rl._apply_altitude_boost([community], 0.25, broad=True)
    assert out[0].score == pytest.approx(0.70)  # only distillation is lifted


def test_rank_hits_applies_altitude_when_knob_set():
    hits = [_Hit("src", 0.80), _distilled("dist", 0.70)]
    knobs = RankKnobs(top_k=5, min_sim=0.0, min_body_chars=0, altitude=0.25)
    out = rl.rank_hits(hits, knobs, query="what about auth")
    assert out[0].id == "dist"


def test_rank_hits_altitude_off_by_default():
    hits = [_Hit("src", 0.80), _distilled("dist", 0.70)]
    knobs = RankKnobs(top_k=5, min_sim=0.0, min_body_chars=0)  # altitude defaults 0.0
    out = rl.rank_hits(hits, knobs, query="what about auth")
    assert out[0].id == "src"  # no boost => source stays on top
