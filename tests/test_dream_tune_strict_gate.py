"""Fase 4 — the tuner's six-condition graduation gate is recorded always and
enforced only behind MEMO_DREAM_TUNE_STRICT_GATE_ENABLED (default off).

The gate verdict must appear in the receipt every night (shadow evidence), but
an un-graduated (default-off) gate must NEVER change the tuner's existing apply
decision. Once the flag is on, a failing gate blocks the apply.

The apply path is reached via the MMR rank knob (same clean, non-leaky stubbing
as test_dream_tune_knobs — module-attr patches only, no global flag_float
monkeypatch, which leaks across tests).
"""

from __future__ import annotations

from memo import dream_metrics
from memo import dream_tune as dt
from memo.eval_recall import LabelSet, Prompt

_MMR = "MEMO_RECALL_MMR_LAMBDA"


class _StubMem:
    def search(self, query, limit, mode="vec", budget_ms=None, exclude_types=None, exclude_tags=None):
        return []


def _cfg(tmp_path):
    from memo.config import Config

    return Config.from_env()


def _metrics(prec, noise=0.0, lat=10.0):
    return {"precision_at_k": prec, "noise_at_k": noise, "latency_ms_p50": lat}


def _one_label():
    return LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["aaaa1111"])]), True


def _reach_apply(monkeypatch, tmp_path):
    """Stub the search so exactly the MMR knob improves — the tuner reaches the
    apply path where the Fase-4 graduation gate sits. All patches are on the dt
    module (reverted per-test) — no global flag_float monkeypatch."""
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(dt, "build_labels", lambda cfg, **k: _one_label())
    monkeypatch.setattr(dt, "load_baseline", lambda sd: None)
    monkeypatch.setattr(dt, "search_min_sim", lambda *a, **k: (0.5, _metrics(0.2), _metrics(0.2)))

    def _fake_search(mem, labels, *, k, floor, knob, current, grid, max_evals):
        if knob == _MMR:
            return 0.5, _metrics(0.2), _metrics(0.4), []
        return current, _metrics(0.2), _metrics(0.2), []

    monkeypatch.setattr(dt, "search_rank_knob", _fake_search)
    monkeypatch.setattr(dt, "curated_gate", lambda *a, **k: {"ok": True})


def test_gate_recorded_but_not_enforced_when_flag_off(tmp_path, monkeypatch):
    _reach_apply(monkeypatch, tmp_path)
    # Even a FAILING gate must not block the apply while the flag is off.
    monkeypatch.setattr(
        dream_metrics, "graduation_gate", lambda *a, **k: {"ok": False, "reasons": ["stub"]}
    )
    monkeypatch.delenv("MEMO_DREAM_TUNE_STRICT_GATE_ENABLED", raising=False)

    res = dt.run_tuning_pass(_cfg(tmp_path), _StubMem(), k=5)
    assert res["status"] == "applied"  # existing behavior preserved
    assert res["applied_knob"] == _MMR
    assert res["graduation_gate"] == {"ok": False, "reasons": ["stub"]}  # recorded


def test_gate_enforced_blocks_apply_when_flag_on(tmp_path, monkeypatch):
    _reach_apply(monkeypatch, tmp_path)
    monkeypatch.setattr(
        dream_metrics,
        "graduation_gate",
        lambda *a, **k: {"ok": False, "reasons": ["insufficient_labels"]},
    )
    monkeypatch.setenv("MEMO_DREAM_TUNE_STRICT_GATE_ENABLED", "1")

    res = dt.run_tuning_pass(_cfg(tmp_path), _StubMem(), k=5)
    assert res["status"] == "gate_rejected"
    assert res["graduation_gate"]["reasons"] == ["insufficient_labels"]
    assert dt.read_overlay(tmp_path) == {}  # nothing applied


def test_gate_ok_applies_when_flag_on(tmp_path, monkeypatch):
    _reach_apply(monkeypatch, tmp_path)
    monkeypatch.setattr(
        dream_metrics, "graduation_gate", lambda *a, **k: {"ok": True, "reasons": []}
    )
    monkeypatch.setenv("MEMO_DREAM_TUNE_STRICT_GATE_ENABLED", "1")

    res = dt.run_tuning_pass(_cfg(tmp_path), _StubMem(), k=5)
    assert res["status"] == "applied"
    assert res["graduation_gate"] == {"ok": True, "reasons": []}
