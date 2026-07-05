"""Nightly HyDE A/B pass — offline gates + overlay apply, all stubs, no MLX."""

from __future__ import annotations

from memo import dream_tune
from memo.eval_recall import LabelSet, Prompt


def _labels() -> tuple[LabelSet, bool]:
    return LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["aaaabbbb"])]), True


def _metrics(prec: float, noise: float, lat: float) -> dict[str, float]:
    return {"precision_at_k": prec, "noise_at_k": noise, "latency_ms_p50": lat}


def test_hyde_pass_applies_when_on_wins(tmp_cfg, monkeypatch) -> None:
    monkeypatch.setattr(dream_tune, "build_labels", lambda cfg, **k: _labels())
    monkeypatch.setattr(dream_tune, "_curated_label_set", lambda sd: None)
    monkeypatch.setattr(
        dream_tune, "measure_hyde",
        lambda mem, labels, *, k, enabled: _metrics(0.4 if enabled else 0.2, 0.0, 50.0),
    )
    res = dream_tune.run_hyde_pass(tmp_cfg, mem=object())
    assert res["status"] == "applied"
    from memo.tuned_overlay import read_overlay

    doc = read_overlay(tmp_cfg.state_dir)
    assert doc["MEMO_HYDE_ENABLED"] is True


def test_hyde_pass_noop_when_it_loses(tmp_cfg, monkeypatch) -> None:
    monkeypatch.setattr(dream_tune, "build_labels", lambda cfg, **k: _labels())
    monkeypatch.setattr(
        dream_tune, "measure_hyde",
        lambda mem, labels, *, k, enabled: _metrics(0.2 if enabled else 0.4, 0.0, 50.0),
    )
    res = dream_tune.run_hyde_pass(tmp_cfg, mem=object())
    assert res["status"] == "hyde_loses"


def test_hyde_pass_rejects_latency_blowup(tmp_cfg, monkeypatch) -> None:
    monkeypatch.setattr(dream_tune, "build_labels", lambda cfg, **k: _labels())
    monkeypatch.setattr(
        dream_tune, "measure_hyde",
        lambda mem, labels, *, k, enabled: _metrics(
            0.4 if enabled else 0.2, 0.0, 10_000.0 if enabled else 100.0
        ),
    )
    res = dream_tune.run_hyde_pass(tmp_cfg, mem=object())
    assert res["status"] == "rejected_latency"


def test_hyde_pass_vetoed_when_live_mode_is_hybrid(tmp_cfg, monkeypatch) -> None:
    # An overlay-applied MEMO_HYDE_ENABLED under a hybrid recall hook would put
    # an MLX chat call in the 5s hook path — the pass must refuse outright.
    monkeypatch.setenv("MEMO_RECALL_MODE", "hybrid")
    res = dream_tune.run_hyde_pass(tmp_cfg, mem=object())
    assert res["status"] == "skipped_hook_mode_hybrid"


def test_hyde_pass_dry_run_does_not_write_overlay(tmp_cfg, monkeypatch) -> None:
    monkeypatch.setattr(dream_tune, "build_labels", lambda cfg, **k: _labels())
    monkeypatch.setattr(dream_tune, "_curated_label_set", lambda sd: None)
    monkeypatch.setattr(
        dream_tune, "measure_hyde",
        lambda mem, labels, *, k, enabled: _metrics(0.4 if enabled else 0.2, 0.0, 50.0),
    )
    res = dream_tune.run_hyde_pass(tmp_cfg, mem=object(), dry_run=True)
    assert res["status"] == "would_apply"
    from memo.tuned_overlay import read_overlay

    assert "MEMO_HYDE_ENABLED" not in read_overlay(tmp_cfg.state_dir)
