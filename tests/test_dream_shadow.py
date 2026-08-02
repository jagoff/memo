"""Tests for dream_shadow — measure-only ("shadow") evaluation of opt-in dream phases.

Targets the pure / fully-deterministic surface: the latency ceiling gate, the
recall verdict axis, observation recording + the consecutive-clean streak, the
review-ready rollup, and the human promote/reject decisions. Hermetic: temp
state dirs only, no MLX, no network, no vault. Gate-kind lookups are stubbed via
``mod._gate`` and the markdown-key inverse via ``mod._env_to_path``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from memo import dream_shadow as mod

# --- helpers --------------------------------------------------------------------

_OFF = {"precision_at_k": 0.5, "noise_at_k": 0.2, "latency_ms_p50": 40.0}
_WIN_ON = {"precision_at_k": 0.7, "noise_at_k": 0.1, "latency_ms_p50": 50.0}
_LOSE_ON = {"precision_at_k": 0.3, "noise_at_k": 0.3, "latency_ms_p50": 50.0}


class _StubGate:
    """Minimal stand-in for a ``dream_flags`` GateSpec (kind + shadow_metric)."""

    def __init__(self, kind: str = "shadow", shadow_metric: str = "") -> None:
        self.kind = kind
        self.shadow_metric = shadow_metric


def _cfg(state_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(state_dir=state_dir)


def _seed_nights(state_dir: Path, flag: str, nights: list[str], on: dict) -> None:
    for night in nights:
        mod.record_recall_shadow(state_dir, flag=flag, off=_OFF, on=on, night=night)


# --- latency_ceiling_gate -------------------------------------------------------


def test_latency_ceiling_gate_disabled_when_ceiling_non_positive() -> None:
    # A non-positive ceiling disables the gate: always passes, however slow.
    assert mod.latency_ceiling_gate(99_999.0, 0) is True
    assert mod.latency_ceiling_gate(99_999.0, -1.0) is True


def test_latency_ceiling_gate_passes_when_within_ceiling() -> None:
    assert mod.latency_ceiling_gate(48.0, 1500.0) is True
    assert mod.latency_ceiling_gate(1500.0, 1500.0) is True  # boundary: equal passes


def test_latency_ceiling_gate_fails_when_over_ceiling() -> None:
    assert mod.latency_ceiling_gate(6272.0, 1500.0) is False


# --- _recall_verdict ------------------------------------------------------------


def test_recall_verdict_win_on_precision_gain() -> None:
    assert mod._recall_verdict(0.1, 0.0) == "win"
    # positive Δprecision wins even when noise also rises
    assert mod._recall_verdict(0.1, 0.5) == "win"


def test_recall_verdict_win_on_noise_drop_at_flat_precision() -> None:
    assert mod._recall_verdict(0.0, -0.1) == "win"


def test_recall_verdict_lose_on_precision_drop() -> None:
    assert mod._recall_verdict(-0.1, -0.5) == "lose"


def test_recall_verdict_lose_on_noise_rise_at_flat_precision() -> None:
    assert mod._recall_verdict(0.0, 0.1) == "lose"


def test_recall_verdict_neutral_when_both_flat() -> None:
    assert mod._recall_verdict(0.0, 0.0) == "neutral"


# --- record_recall_shadow -------------------------------------------------------


def test_record_recall_shadow_appends_observation(tmp_path: Path) -> None:
    obs = mod.record_recall_shadow(
        tmp_path, flag="MEMO_X", off=_OFF, on=_WIN_ON, night="2026-08-01"
    )
    assert obs is not None
    assert obs["kind"] == "recall"
    assert obs["flag"] == "MEMO_X"
    assert obs["night"] == "2026-08-01"
    assert obs["delta_precision"] == 0.2  # 0.7 - 0.5
    assert obs["delta_noise"] == -0.1  # 0.1 - 0.2
    assert obs["cost_ms"] == 50.0  # ON p50
    assert obs["verdict"] == "win"

    rows = mod.read_observations(tmp_path, flag="MEMO_X")
    assert len(rows) == 1
    assert rows[0]["night"] == "2026-08-01"


def test_streak_increments_over_consecutive_clean_nights(tmp_path: Path) -> None:
    _seed_nights(tmp_path, "MEMO_X", ["d1", "d2", "d3"], _WIN_ON)
    entry = mod._load_state(tmp_path)["flags"]["MEMO_X"]
    assert entry["nights"] == 3
    assert entry["streak"] == 3
    assert entry["last_verdict"] == "win"


def test_non_clean_night_resets_streak(tmp_path: Path) -> None:
    _seed_nights(tmp_path, "MEMO_X", ["d1", "d2"], _WIN_ON)
    mod.record_recall_shadow(tmp_path, flag="MEMO_X", off=_OFF, on=_LOSE_ON, night="d3")
    entry = mod._load_state(tmp_path)["flags"]["MEMO_X"]
    assert entry["nights"] == 3  # nights keep counting
    assert entry["streak"] == 0  # a lose resets the consecutive-clean streak
    assert entry["last_verdict"] == "lose"


def test_neutral_verdict_counts_as_clean_night(tmp_path: Path) -> None:
    same = {"precision_at_k": 0.5, "noise_at_k": 0.2}
    obs = mod.record_recall_shadow(
        tmp_path,
        flag="MEMO_X",
        off={**same, "latency_ms_p50": 40.0},
        on={**same, "latency_ms_p50": 50.0},
        night="d1",
    )
    assert obs is not None and obs["verdict"] == "neutral"
    assert mod._load_state(tmp_path)["flags"]["MEMO_X"]["streak"] == 1


def test_same_night_remeasurement_is_idempotent_for_state(tmp_path: Path) -> None:
    mod.record_recall_shadow(tmp_path, flag="MEMO_X", off=_OFF, on=_WIN_ON, night="d1")
    mod.record_recall_shadow(tmp_path, flag="MEMO_X", off=_OFF, on=_WIN_ON, night="d1")
    entry = mod._load_state(tmp_path)["flags"]["MEMO_X"]
    assert entry["nights"] == 1  # same night never double-counts
    assert entry["streak"] == 1


# --- read_observations ----------------------------------------------------------


def test_read_observations_filters_by_flag(tmp_path: Path) -> None:
    mod.record_recall_shadow(tmp_path, flag="MEMO_A", off=_OFF, on=_WIN_ON, night="d1")
    mod.record_recall_shadow(tmp_path, flag="MEMO_B", off=_OFF, on=_WIN_ON, night="d1")
    a_rows = mod.read_observations(tmp_path, flag="MEMO_A")
    assert len(a_rows) == 1 and a_rows[0]["flag"] == "MEMO_A"
    assert len(mod.read_observations(tmp_path)) == 2


def test_read_observations_excludes_event_rows(tmp_path: Path) -> None:
    mod.record_recall_shadow(tmp_path, flag="MEMO_X", off=_OFF, on=_WIN_ON, night="d1")
    mod.reject(_cfg(tmp_path), "MEMO_X", "nope")  # appends a lifecycle "event" row
    rows = mod.read_observations(tmp_path)
    assert len(rows) == 1  # the event row is excluded
    assert rows[0]["kind"] == "recall"
    assert all(r["kind"] in ("recall", "pass") for r in rows)


# --- shadow_summary / review-ready ----------------------------------------------


def test_shadow_summary_not_review_ready_before_enough_nights(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MEMO_SHADOW_REVIEW_NIGHTS", "3")
    _seed_nights(tmp_path, "MEMO_X", ["d1", "d2"], _WIN_ON)
    summary = mod.shadow_summary(tmp_path, "MEMO_X")
    assert summary["streak"] == 2
    assert summary["review_nights"] == 3
    assert summary["review_ready"] is False


def test_shadow_summary_review_ready_after_enough_clean_nights(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MEMO_SHADOW_REVIEW_NIGHTS", "3")
    _seed_nights(tmp_path, "MEMO_X", ["d1", "d2", "d3"], _WIN_ON)
    summary = mod.shadow_summary(tmp_path, "MEMO_X")
    assert summary["streak"] == 3
    assert summary["review_ready"] is True


def test_shadow_summary_not_review_ready_after_decision(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MEMO_SHADOW_REVIEW_NIGHTS", "3")
    _seed_nights(tmp_path, "MEMO_X", ["d1", "d2", "d3"], _WIN_ON)
    assert mod.shadow_summary(tmp_path, "MEMO_X")["review_ready"] is True
    mod.reject(_cfg(tmp_path), "MEMO_X", "not worth it")
    summary = mod.shadow_summary(tmp_path, "MEMO_X")
    assert summary["decision"] == "rejected"
    assert summary["review_ready"] is False  # a decision drops it out of the pool


# --- promote --------------------------------------------------------------------


def test_promote_refuses_when_not_review_ready(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_SHADOW_REVIEW_NIGHTS", "3")
    _seed_nights(tmp_path, "MEMO_X", ["d1"], _WIN_ON)  # only 1 clean night
    result = mod.promote(_cfg(tmp_path), "MEMO_X")
    assert result["ok"] is False
    assert result["applied"] is False
    assert "not review-ready" in result["error"]


def test_promote_returns_config_set_command_when_review_ready(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MEMO_SHADOW_REVIEW_NIGHTS", "3")
    monkeypatch.setattr(mod, "_gate", lambda f: _StubGate(shadow_metric="precision"))
    monkeypatch.setattr(mod, "_env_to_path", lambda: {"MEMO_X": "recall.graph_boost"})
    _seed_nights(tmp_path, "MEMO_X", ["d1", "d2", "d3"], _WIN_ON)

    result = mod.promote(_cfg(tmp_path), "MEMO_X", apply=False)
    assert result["ok"] is True
    assert result["applied"] is False  # apply=False writes nothing
    assert result["path"] == "recall.graph_boost"
    assert result["command"] == "memo config set recall.graph_boost true"


def test_promote_export_fallback_when_no_markdown_key(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MEMO_SHADOW_REVIEW_NIGHTS", "2")
    monkeypatch.setattr(mod, "_gate", lambda f: _StubGate(shadow_metric="precision"))
    monkeypatch.setattr(mod, "_env_to_path", lambda: {})  # no markdown-config surface
    _seed_nights(tmp_path, "MEMO_X", ["d1", "d2"], _WIN_ON)

    result = mod.promote(_cfg(tmp_path), "MEMO_X")
    assert result["ok"] is True
    assert result["instruction"] == "export MEMO_X=1"
    assert "command" not in result


def test_promote_refuses_latency_metric_over_ceiling(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MEMO_SHADOW_REVIEW_NIGHTS", "3")
    monkeypatch.setenv("MEMO_FLAG_GRADUATION_LATENCY_CEILING_MS", "100")
    monkeypatch.setattr(mod, "_gate", lambda f: _StubGate(shadow_metric="latency_ms"))
    monkeypatch.setattr(mod, "_env_to_path", lambda: {"MEMO_X": "recall.graph_boost"})
    slow_on = {"precision_at_k": 0.7, "noise_at_k": 0.1, "latency_ms_p50": 6000.0}
    _seed_nights(tmp_path, "MEMO_X", ["d1", "d2", "d3"], slow_on)

    result = mod.promote(_cfg(tmp_path), "MEMO_X")
    assert result["ok"] is False
    assert "exceeds ceiling" in result["error"]


def test_promote_latency_over_ceiling_passes_with_force(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MEMO_SHADOW_REVIEW_NIGHTS", "3")
    monkeypatch.setenv("MEMO_FLAG_GRADUATION_LATENCY_CEILING_MS", "100")
    monkeypatch.setattr(mod, "_gate", lambda f: _StubGate(shadow_metric="latency_ms"))
    monkeypatch.setattr(mod, "_env_to_path", lambda: {"MEMO_X": "recall.graph_boost"})
    slow_on = {"precision_at_k": 0.7, "noise_at_k": 0.1, "latency_ms_p50": 6000.0}
    _seed_nights(tmp_path, "MEMO_X", ["d1", "d2", "d3"], slow_on)

    result = mod.promote(_cfg(tmp_path), "MEMO_X", force_latency=True)
    assert result["ok"] is True
    assert result["command"] == "memo config set recall.graph_boost true"


# --- reject ---------------------------------------------------------------------


def test_reject_marks_decision(tmp_path: Path) -> None:
    assert mod.reject(_cfg(tmp_path), "MEMO_X", "too slow") is True
    entry = mod._load_state(tmp_path)["flags"]["MEMO_X"]
    assert entry["decision"] == "rejected"
    assert entry["decision_reason"] == "too slow"
    # reject records a lifecycle event, never an observation row
    assert mod.read_observations(tmp_path, flag="MEMO_X") == []
