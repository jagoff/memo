"""floor-calibration: noise-quantile min_sim from shuffled probes."""

from __future__ import annotations

import numpy as np

from memo import floor_calibration
from memo.cli_dream_passes import _run_floor_calibration


class _StubEmbedder:
    """Deterministic 8-dim embedder: random-but-seeded unit vectors per text."""

    def embed(self, texts):
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(8)
            v = v / (np.linalg.norm(v) or 1.0)
            out.append(v.tolist())
        return out


def test_estimate_noise_floor_is_sane():
    floor = floor_calibration.estimate_noise_floor(
        _StubEmbedder(),
        probes=["alpha beta", "gamma delta", "epsilon zeta", "eta theta"],
        quantile=0.95,
        dims=8,
    )
    assert 0.0 <= floor <= 1.0


def test_estimate_noise_floor_empty_returns_none():
    assert (
        floor_calibration.estimate_noise_floor(_StubEmbedder(), probes=[], quantile=0.95, dims=8)
        is None
    )


def test_run_floor_calibration_noop_when_flag_off(mock_memory, monkeypatch):
    """Flag off -> complete no-op: no embed call, no overlay write, inert receipt."""
    monkeypatch.setattr("memo.flags.flag_bool", lambda name, **kw: False)
    embed_calls = []
    orig_embed = mock_memory.embedder.embed

    def _tracking_embed(inputs):
        embed_calls.append(list(inputs))
        return orig_embed(inputs)

    mock_memory.embedder.embed = _tracking_embed

    result = _run_floor_calibration(mock_memory, dry_run=False)

    assert result["floor_calibration"]["applied"] is False
    assert embed_calls == []
    from memo.tuned_overlay import overlay_path

    assert not overlay_path(mock_memory.cfg.state_dir).exists()


def _flag_stub(min_sim=0.0):
    def _stub(name, **kw):
        if name == "MEMO_FLOOR_CALIBRATION":
            return True
        if name == "MEMO_RECALL_MIN_SIM":
            return min_sim
        return kw.get("default")

    return _stub


def _seed_probes(mem):
    for i in range(6):
        mem.save(content=f"probe body {i}", title=f"probe title {i} alpha beta gamma", type="note")


def test_run_floor_calibration_applies_and_raises_min_sim(mock_memory, monkeypatch):
    """Flag on, curated gate passes, floor above the pinned-low current min_sim
    -> overlay is written and only ever raises."""
    _seed_probes(mock_memory)

    monkeypatch.setattr("memo.flags.flag_bool", _flag_stub())
    monkeypatch.setattr(
        "memo.flags.flag_float", lambda name, **kw: 0.0 if name == "MEMO_RECALL_MIN_SIM" else None
    )
    monkeypatch.setattr("memo.dream_tune._curated_label_set", lambda state_dir: object())
    monkeypatch.setattr(
        "memo.dream_tune.measure",
        lambda *a, **kw: {"precision_at_k": 1.0, "noise_at_k": 0.0},
    )

    result = _run_floor_calibration(mock_memory, dry_run=False)

    frag = result["floor_calibration"]
    assert frag["gate"] == "ok"
    assert frag["proposed"] is not None
    assert frag["applied"] is True

    from memo.tuned_overlay import read_overlay

    overlay = read_overlay(mock_memory.cfg.state_dir)
    assert overlay["MEMO_RECALL_MIN_SIM"] == frag["proposed"]
    assert overlay["MEMO_RECALL_MIN_SIM"] > 0.0


def test_run_floor_calibration_without_curated_labels_does_not_apply(mock_memory, monkeypatch):
    """Fails closed. The curated set ships in the wheel, so its absence means a
    damaged install — raising the recall floor unverified would silently bury
    memories with nothing able to prove or revert it."""
    _seed_probes(mock_memory)

    monkeypatch.setattr("memo.flags.flag_bool", _flag_stub())
    monkeypatch.setattr(
        "memo.flags.flag_float", lambda name, **kw: 0.0 if name == "MEMO_RECALL_MIN_SIM" else None
    )
    monkeypatch.setattr("memo.dream_tune._curated_label_set", lambda state_dir: None)

    result = _run_floor_calibration(mock_memory, dry_run=False)

    frag = result["floor_calibration"]
    assert frag["gate"] == "no_curated_labels"
    assert frag.get("applied") is not True

    from memo.tuned_overlay import overlay_path

    assert not overlay_path(mock_memory.cfg.state_dir).exists()


def test_run_floor_calibration_never_lowers_below_current(mock_memory, monkeypatch):
    """current min_sim pinned ABOVE any achievable cosine floor (unit vectors
    can't exceed 1.0) -> floor <= current -> no overlay write."""
    for i in range(6):
        mock_memory.save(
            content=f"probe body {i}", title=f"probe title {i} alpha beta gamma", type="note"
        )

    monkeypatch.setattr("memo.flags.flag_bool", _flag_stub())
    monkeypatch.setattr(
        "memo.flags.flag_float", lambda name, **kw: 1.0 if name == "MEMO_RECALL_MIN_SIM" else None
    )

    result = _run_floor_calibration(mock_memory, dry_run=False)

    frag = result["floor_calibration"]
    assert frag["gate"] == "not_above_current"
    assert frag["applied"] is False

    from memo.tuned_overlay import overlay_path

    assert not overlay_path(mock_memory.cfg.state_dir).exists()


def test_run_floor_calibration_curated_rejected_blocks_write(mock_memory, monkeypatch):
    """Proposed floor regresses the curated set -> gate rejects, no overlay write."""
    for i in range(6):
        mock_memory.save(
            content=f"probe body {i}", title=f"probe title {i} alpha beta gamma", type="note"
        )

    monkeypatch.setattr("memo.flags.flag_bool", _flag_stub())
    monkeypatch.setattr(
        "memo.flags.flag_float", lambda name, **kw: 0.0 if name == "MEMO_RECALL_MIN_SIM" else None
    )
    monkeypatch.setattr("memo.dream_tune._curated_label_set", lambda state_dir: object())
    monkeypatch.setattr(
        "memo.dream_tune.measure",
        lambda mem, labels, *, k, floor: (
            {"precision_at_k": 0.5, "noise_at_k": 0.0}
            if floor == 0.0
            else {"precision_at_k": 0.0, "noise_at_k": 0.5}
        ),
    )

    result = _run_floor_calibration(mock_memory, dry_run=False)

    frag = result["floor_calibration"]
    assert frag["gate"] == "curated_rejected"
    assert frag["applied"] is False

    from memo.tuned_overlay import overlay_path

    assert not overlay_path(mock_memory.cfg.state_dir).exists()


def test_run_floor_calibration_dry_run_writes_nothing(mock_memory, monkeypatch):
    """dry_run=True computes/gates but never writes the overlay."""
    for i in range(6):
        mock_memory.save(
            content=f"probe body {i}", title=f"probe title {i} alpha beta gamma", type="note"
        )

    monkeypatch.setattr("memo.flags.flag_bool", _flag_stub())
    monkeypatch.setattr(
        "memo.flags.flag_float", lambda name, **kw: 0.0 if name == "MEMO_RECALL_MIN_SIM" else None
    )
    monkeypatch.setattr("memo.dream_tune._curated_label_set", lambda state_dir: None)

    result = _run_floor_calibration(mock_memory, dry_run=True)

    frag = result["floor_calibration"]
    assert frag["applied"] is False

    from memo.tuned_overlay import overlay_path

    assert not overlay_path(mock_memory.cfg.state_dir).exists()
