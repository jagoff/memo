"""Unit tests for the pure gate functions + learning-metric helpers in
``memo.dream_metrics``.

Scope is the fully-deterministic surface: the pre-apply graduation gate and its
sub-gates, plus the two pure learning-metric helpers. ``learning_metrics``
reaches many live subsystems, so it is exercised only for its total contract
(returns a dict with an ``errors`` list, never raises) on a minimal fake
cfg/mem. Every test is hermetic — no MLX, no network, no vault, temp dirs only.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from memo import dream_metrics as mod

# --- min_labels_gate ---------------------------------------------------------


def test_min_labels_gate_passes_when_at_or_above_floor() -> None:
    assert mod.min_labels_gate(12, 12) is True  # equal is enough
    assert mod.min_labels_gate(20, 12) is True


def test_min_labels_gate_fails_below_floor() -> None:
    assert mod.min_labels_gate(11, 12) is False
    assert mod.min_labels_gate(0, 1) is False


# --- margin_gate -------------------------------------------------------------


def test_margin_gate_zero_margin_accepts_non_regression() -> None:
    # min_margin=0.0 -> any non-regression passes (equality included).
    assert mod.margin_gate(0.5, 0.5, 0.0) is True
    assert mod.margin_gate(0.5, 0.6, 0.0) is True


def test_margin_gate_zero_margin_rejects_regression() -> None:
    assert mod.margin_gate(0.5, 0.49, 0.0) is False


def test_margin_gate_deadband_rejects_trivial_epsilon_churn() -> None:
    # With min_margin>0, an unchanged (or barely-nudged) precision is rejected.
    assert mod.margin_gate(0.5, 0.5, 0.02) is False
    assert mod.margin_gate(0.5, 0.515, 0.02) is False


def test_margin_gate_deadband_accepts_real_improvement() -> None:
    assert mod.margin_gate(0.5, 0.53, 0.02) is True
    # Exactly at the deadband boundary passes (within _EPS).
    assert mod.margin_gate(0.5, 0.52, 0.02) is True


# --- noise_gate --------------------------------------------------------------


def test_noise_gate_accepts_flat_or_decreasing_noise() -> None:
    assert mod.noise_gate(0.1, 0.1) is True  # unchanged
    assert mod.noise_gate(0.1, 0.05) is True  # improved


def test_noise_gate_rejects_increased_noise() -> None:
    assert mod.noise_gate(0.1, 0.2) is False


def test_noise_gate_tolerates_epsilon_representation_noise() -> None:
    # A rise strictly inside _EPS is not a real regression.
    assert mod.noise_gate(0.1, 0.1 + 1e-10) is True


# --- latency_gate ------------------------------------------------------------


def test_latency_gate_passes_within_both_ratio_and_budget() -> None:
    assert mod.latency_gate(100.0, 120.0, headroom=1.25, hook_budget_ms=3000.0) is True


def test_latency_gate_rejects_over_absolute_budget() -> None:
    # Ratio would pass (baseline huge) but the absolute hook budget is blown.
    assert mod.latency_gate(100.0, 4000.0, headroom=1000.0, hook_budget_ms=3000.0) is False


def test_latency_gate_rejects_ratio_regression_within_budget() -> None:
    # Under the absolute budget, but 200ms > 1.25 * 100ms baseline.
    assert mod.latency_gate(100.0, 200.0, headroom=1.25, hook_budget_ms=3000.0) is False


def test_latency_gate_skips_ratio_when_baseline_rounds_to_zero() -> None:
    # round(0.04, 1) == 0.0 -> the ratio is meaningless and skipped; only the
    # absolute budget matters, so a large-but-under-budget latency passes.
    assert mod.latency_gate(0.04, 500.0, headroom=1.25, hook_budget_ms=3000.0) is True


def test_latency_gate_enforces_budget_even_when_ratio_skipped() -> None:
    # Zero-ish baseline skips the ratio, but the absolute budget still bites.
    assert mod.latency_gate(0.04, 4000.0, headroom=1.25, hook_budget_ms=3000.0) is False


def test_latency_gate_boundary_at_exact_budget_passes() -> None:
    assert mod.latency_gate(100.0, 3000.0, headroom=100.0, hook_budget_ms=3000.0) is True


# --- graduation_gate ---------------------------------------------------------


def _grad_kwargs(**overrides: Any) -> dict[str, Any]:
    """A fully-passing kwargs baseline for graduation_gate; override to break one."""
    kwargs: dict[str, Any] = {
        "before": {"precision_at_k": 0.5, "noise_at_k": 0.1},
        "after": {"precision_at_k": 0.6, "noise_at_k": 0.1},
        "n_labels": 20,
        "latency_before": 100.0,
        "latency_after": 110.0,
        "min_labels": 12,
        "latency_headroom": 1.25,
        "hook_budget_ms": 3000.0,
        "min_margin": 0.0,
        "experiment_recorded": True,
        "rollback_exists": True,
    }
    kwargs.update(overrides)
    before = kwargs.pop("before")
    after = kwargs.pop("after")
    return {"before": before, "after": after, **kwargs}


def test_graduation_gate_all_conditions_pass() -> None:
    kw = _grad_kwargs()
    verdict = mod.graduation_gate(kw.pop("before"), kw.pop("after"), **kw)
    assert verdict["ok"] is True
    assert verdict["reasons"] == []


def test_graduation_gate_fails_on_insufficient_labels() -> None:
    kw = _grad_kwargs(n_labels=5)
    verdict = mod.graduation_gate(kw.pop("before"), kw.pop("after"), **kw)
    assert verdict["ok"] is False
    assert any("insufficient_labels" in r for r in verdict["reasons"])


def test_graduation_gate_fails_on_precision_not_improved() -> None:
    kw = _grad_kwargs(after={"precision_at_k": 0.4, "noise_at_k": 0.1})
    verdict = mod.graduation_gate(kw.pop("before"), kw.pop("after"), **kw)
    assert verdict["ok"] is False
    assert any("precision_not_improved" in r for r in verdict["reasons"])


def test_graduation_gate_fails_on_noise_regressed() -> None:
    kw = _grad_kwargs(after={"precision_at_k": 0.6, "noise_at_k": 0.3})
    verdict = mod.graduation_gate(kw.pop("before"), kw.pop("after"), **kw)
    assert verdict["ok"] is False
    assert any("noise_regressed" in r for r in verdict["reasons"])


def test_graduation_gate_fails_on_latency_over_budget() -> None:
    kw = _grad_kwargs(latency_after=4000.0)
    verdict = mod.graduation_gate(kw.pop("before"), kw.pop("after"), **kw)
    assert verdict["ok"] is False
    assert any("latency_over_budget" in r for r in verdict["reasons"])


def test_graduation_gate_fails_on_latency_ratio_regression() -> None:
    # Under the absolute budget but well over the headroom ratio.
    kw = _grad_kwargs(latency_after=200.0)
    verdict = mod.graduation_gate(kw.pop("before"), kw.pop("after"), **kw)
    assert verdict["ok"] is False
    assert any("latency_regressed" in r for r in verdict["reasons"])


def test_graduation_gate_fails_when_experiment_not_recorded() -> None:
    kw = _grad_kwargs(experiment_recorded=False)
    verdict = mod.graduation_gate(kw.pop("before"), kw.pop("after"), **kw)
    assert verdict["ok"] is False
    assert "experiment_not_recorded" in verdict["reasons"]


def test_graduation_gate_fails_when_rollback_missing() -> None:
    kw = _grad_kwargs(rollback_exists=False)
    verdict = mod.graduation_gate(kw.pop("before"), kw.pop("after"), **kw)
    assert verdict["ok"] is False
    assert "rollback_missing" in verdict["reasons"]


def test_graduation_gate_accumulates_every_failure_reason() -> None:
    kw = _grad_kwargs(
        after={"precision_at_k": 0.4, "noise_at_k": 0.3},
        n_labels=1,
        latency_after=4000.0,
        experiment_recorded=False,
        rollback_exists=False,
    )
    verdict = mod.graduation_gate(kw.pop("before"), kw.pop("after"), **kw)
    assert verdict["ok"] is False
    reasons = " ".join(verdict["reasons"])
    for token in (
        "insufficient_labels",
        "precision_not_improved",
        "noise_regressed",
        "latency_over_budget",
        "experiment_not_recorded",
        "rollback_missing",
    ):
        assert token in reasons


# --- correction_rate ---------------------------------------------------------


def test_correction_rate_empty_returns_none_zero() -> None:
    assert mod.correction_rate([]) == (None, 0)


def test_correction_rate_ignores_rows_without_a_verdict() -> None:
    # Falsy/absent verdicts are filtered out; over zero real verdicts -> None.
    assert mod.correction_rate([{"foo": 1}, {"verdict": ""}, {"verdict": None}]) == (
        None,
        0,
    )


def test_correction_rate_counts_negative_and_correction_as_bad() -> None:
    rows = [
        {"verdict": "negative"},
        {"verdict": "positive"},
        {"verdict": "correction"},
        {"verdict": "neutral"},
    ]
    rate, sample = mod.correction_rate(rows)
    assert rate == 0.5
    assert sample == 4


def test_correction_rate_all_positive_is_zero() -> None:
    rate, sample = mod.correction_rate([{"verdict": "positive"}, {"verdict": "neutral"}])
    assert rate == 0.0
    assert sample == 2


def test_correction_rate_all_bad_is_one() -> None:
    rate, sample = mod.correction_rate([{"verdict": "negative"}, {"verdict": "correction"}])
    assert rate == 1.0
    assert sample == 2


# --- created_used_ratio ------------------------------------------------------


def test_created_used_ratio_none_when_nothing_surfaced() -> None:
    assert mod.created_used_ratio(5, 0) is None
    assert mod.created_used_ratio(5, -1) is None


def test_created_used_ratio_computes_fraction() -> None:
    assert mod.created_used_ratio(5, 10) == 0.5
    assert mod.created_used_ratio(0, 10) == 0.0


# --- learning_metrics (contract only) ----------------------------------------


def test_learning_metrics_returns_dict_with_errors_list_and_never_raises(
    tmp_path: Any,
) -> None:
    # Minimal fake cfg/mem against an empty temp state dir: every source is
    # wrapped in _safe, so the snapshot must come back as a dict carrying an
    # ``errors`` list rather than raising.
    cfg = SimpleNamespace(state_dir=tmp_path)
    mem = SimpleNamespace()
    result = mod.learning_metrics(cfg, mem, params_version="test-v0", k=5)

    assert isinstance(result, dict)
    assert isinstance(result["errors"], list)
    assert result["params_version"] == "test-v0"
    assert result["k"] == 5
