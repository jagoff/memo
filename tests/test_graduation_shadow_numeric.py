from dataclasses import dataclass

import pytest

from memo.graduation import shadow_numeric
from memo.graduation.registry import NumericCandidate


@dataclass
class _Row:  # stand-in for eval_recall.Row
    precision_at_k: float
    noise_at_k: float


CAND = NumericCandidate(
    flag="MEMO_RECALL_MMR_LAMBDA", field="mmr_lambda",
    off_value=0.0, on_value=0.3, grid=(0.0, 0.3, 0.5),
)


def test_shadow_numeric_pins_field_via_knob_overrides(monkeypatch):
    seen: list[dict] = []

    def fake_run_config(mem, cfg, k, labels, *, progress=None):
        seen.append(dict(cfg.knob_overrides or {}))
        val = (cfg.knob_overrides or {}).get("mmr_lambda")
        # a knob value of 0.3 helps; everything else is baseline.
        return _Row(0.30 if val == 0.3 else 0.20, 0.0)

    monkeypatch.setattr(shadow_numeric, "run_config", fake_run_config)
    res = shadow_numeric.shadow_eval_numeric(object(), CAND, k=5, labels=object())
    assert res["win"] is True
    assert res["best_value"] == 0.3
    assert res["delta_prec"] == pytest.approx(0.10)
    # OFF uses the REAL default (0.0), not a blind "0" flag_override.
    assert {"mmr_lambda": 0.0} in seen
    assert {"mmr_lambda": 0.3} in seen


def test_shadow_numeric_loses_when_no_grid_value_beats_baseline(monkeypatch):
    def flat(mem, cfg, k, labels, *, progress=None):
        return _Row(0.20, 0.0)  # nothing beats baseline

    monkeypatch.setattr(shadow_numeric, "run_config", flat)
    res = shadow_numeric.shadow_eval_numeric(object(), CAND, k=5, labels=object())
    assert res["win"] is False
    assert res["best_value"] == CAND.off_value  # stays at the baseline


def test_shadow_numeric_rejects_when_noise_rises(monkeypatch):
    def noisy(mem, cfg, k, labels, *, progress=None):
        val = (cfg.knob_overrides or {}).get("mmr_lambda")
        return _Row(0.30 if val == 0.3 else 0.20, 0.05 if val == 0.3 else 0.0)

    monkeypatch.setattr(shadow_numeric, "run_config", noisy)
    res = shadow_numeric.shadow_eval_numeric(object(), CAND, k=5, labels=object())
    assert res["win"] is False  # precision up but noise rose => not a win
