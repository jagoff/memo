"""The eval grid must include a config that ranks the way the hook ranks.

Grid configs A-D pin a chosen floor and `min_body_chars=0`. That is right for
comparing knobs, but it means a green gate can report precision for a
configuration nobody runs — the same failure shape as a pre-push gate that
measures the installed binary instead of the diff. `live_config()` closes that
by inheriting every knob it can from the live flag chain.
"""

from __future__ import annotations

import pytest

from memo.eval_recall import Cfg, default_configs, live_config
from memo.recall_logic import knobs_from_flags


def test_the_default_grid_contains_exactly_one_live_config() -> None:
    live = [c for c in default_configs() if c.live]

    assert len(live) == 1, [c.name for c in default_configs()]


def test_live_config_applies_the_hooks_injection_filters_and_archive_exclusion() -> None:
    """Without these it would still not be the hook: the hook drops a weak top
    hit (skip-below) and trims on a score gap before injecting anything."""
    cfg = live_config()

    assert cfg.live is True
    assert cfg.injection_fidelity is True
    assert cfg.exclude_archived is True


def test_grid_configs_are_not_live() -> None:
    for cfg in default_configs():
        if cfg.name.startswith(("A ", "B ", "C ", "D ")):
            assert cfg.live is False, cfg.name
            assert cfg.injection_fidelity is False, cfg.name


def test_live_config_name_reports_the_resolved_mode_and_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The name has to carry the resolved values — a fixed label would hide the
    tuner moving the floor underneath the gate."""
    monkeypatch.setenv("MEMO_RECALL_MODE", "hybrid")
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.83")

    cfg = live_config()

    assert cfg.mode == "hybrid"
    assert cfg.name == "L live/hybrid/0.83"
    assert cfg.floor == pytest.approx(0.83)


def test_live_config_tracks_a_floor_change(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pinned grid config cannot do this; that is the whole point."""
    monkeypatch.setenv("MEMO_RECALL_MODE", "vec")
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.60")
    low = live_config()
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.90")
    high = live_config()

    assert low.floor != high.floor
    assert (low.name, high.name) == ("L live/vec/0.6", "L live/vec/0.9")


def _resolve(cfg: Cfg, k: int):
    """Mirror run_config's knob construction for the fields under test."""
    return knobs_from_flags(
        top_k=k,
        min_sim=None if cfg.live else cfg.floor,
        min_body_chars=None if cfg.live else 0,
        mode=None if cfg.live else cfg.mode,
        overrides={"code_proximity": False},
    )


def test_live_config_inherits_the_knobs_a_grid_config_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression this guards: the live hook ranked at min_sim 0.8835 with
    min_body_chars 40 while every grid config measured 0.60/0.72/0.40 and 0."""
    monkeypatch.setenv("MEMO_RECALL_MODE", "vec")
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.8835")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "40")

    live = _resolve(live_config(), k=5)
    grid = _resolve(Cfg("A vec/0.60/keep", "vec", 0.60, exclude_archived=False), k=5)

    assert live.min_sim == pytest.approx(0.8835)
    assert live.min_body_chars == 40
    # The grid config keeps its pins — it is not silently made live.
    assert grid.min_sim == pytest.approx(0.60)
    assert grid.min_body_chars == 0


def test_live_config_matches_the_hooks_own_knob_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equality with a bare knobs_from_flags(top_k=k) is the definition of
    'ranks like the hook' for every knob except top_k."""
    monkeypatch.setenv("MEMO_RECALL_MODE", "vec")
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.77")

    live = _resolve(live_config(), k=3)
    hook = knobs_from_flags(top_k=3, overrides={"code_proximity": False})

    assert live == hook
