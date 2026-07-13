from dataclasses import dataclass

import pytest

from memo.graduation import shadow
from memo.graduation.registry import Candidate


@dataclass
class _Row:  # minimal stand-in for eval_recall.Row
    precision_at_k: float
    noise_at_k: float


def test_decide_win_true_when_precision_up_and_noise_flat():
    win, deltas = shadow.decide_win(_Row(0.20, 0.00), _Row(0.30, 0.00), epsilon=0.0)
    assert win is True
    assert deltas["delta_prec"] == pytest.approx(0.10)
    assert deltas["delta_noise"] == pytest.approx(0.0)


def test_decide_win_false_when_noise_rises():
    win, _ = shadow.decide_win(_Row(0.20, 0.00), _Row(0.30, 0.05), epsilon=0.0)
    assert win is False  # precision up but noise rose => not a win


def test_decide_win_false_below_epsilon():
    win, _ = shadow.decide_win(_Row(0.20, 0.00), _Row(0.21, 0.00), epsilon=0.05)
    assert win is False


def test_shadow_eval_runs_off_then_on(monkeypatch):
    # run_config returns low precision for OFF, high for ON -> win.
    calls: list[dict] = []

    def fake_run_config(mem, cfg, k, labels, *, progress=None):
        calls.append(dict(cfg.flag_overrides or {}))
        on = (cfg.flag_overrides or {}).get("MEMO_GRAPH_SIGNAL_ENABLED") == "1"
        return _Row(0.30 if on else 0.20, 0.0)

    monkeypatch.setattr(shadow, "run_config", fake_run_config)
    cand = Candidate(flag="MEMO_GRAPH_SIGNAL_ENABLED",
                     on_flags={"MEMO_GRAPH_SIGNAL_ENABLED": "1"})
    res = shadow.shadow_eval(object(), cand, k=5, labels=object())
    assert res["win"] is True
    assert res["delta_prec"] == pytest.approx(0.10)
    # OFF config zeroes every on_flags key; ON uses on_flags verbatim.
    assert {"MEMO_GRAPH_SIGNAL_ENABLED": "0"} in calls
    assert {"MEMO_GRAPH_SIGNAL_ENABLED": "1"} in calls
