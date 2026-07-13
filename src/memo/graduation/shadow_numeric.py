"""Offline shadow evaluator for NUMERIC ranking knobs. Unlike the boolean
``shadow.shadow_eval`` (which derives OFF by zeroing flag_overrides), a numeric
knob's OFF is its current DEFAULT (``off_value``), pinned through
``eval_recall.Cfg.knob_overrides`` — the recall-faithful seam ``rank_hits`` reads
(same as ``dream_tune.measure_rank_knob``). Line-searches ``cand.grid`` for the
best value that wins vs the baseline. No behavior change — only reads the index
through eval_recall."""
from __future__ import annotations

import dataclasses
from typing import Any

from memo.eval_recall import Cfg, LabelSet, default_configs, run_config
from memo.graduation.registry import NumericCandidate
from memo.graduation.shadow import decide_win


def _with_knob(base: Cfg, name: str, field: str, value: float) -> Cfg:
    """Create a new Cfg by overriding a knob on a base config.

    Args:
        base: Base Cfg to clone from
        name: New name for the config
        field: RankKnobs field name to override (e.g., "mmr_lambda")
        value: Numeric value to set

    Returns:
        New Cfg instance
    """
    return dataclasses.replace(base, name=name, knob_overrides={field: value})


def best_on_value(
    mem: Any, cand: NumericCandidate, *, k: int, labels: LabelSet
) -> tuple[float, dict[str, float]]:
    """Line-search the grid (or the single on_value) for the value that best
    beats the off_value baseline on (precision, -noise). Returns
    (best_value, deltas) where deltas is the winning value's delta vs baseline;
    best_value == off_value with zeroed deltas when nothing beats it.

    Args:
        mem: Memory instance (passed to run_config)
        cand: NumericCandidate with field, off_value, on_value, grid, epsilon
        k: top-k for eval
        labels: LabelSet for eval

    Returns:
        (best_value, deltas) tuple where deltas has delta_prec and delta_noise
    """
    base = default_configs()[0]  # A vec/0.60/keep — the live default shape
    off = run_config(mem, _with_knob(base, f"{cand.flag}-off", cand.field, cand.off_value), k, labels)
    grid = cand.grid or (cand.on_value,)
    best_value = cand.off_value
    best_deltas = {"delta_prec": 0.0, "delta_noise": 0.0}
    best_key = (0.0, 0.0)  # (delta_prec, -delta_noise); baseline is the floor
    for value in grid:
        if value == cand.off_value:
            continue
        on = run_config(mem, _with_knob(base, f"{cand.flag}={value}", cand.field, value), k, labels)
        win, deltas = decide_win(off, on, cand.epsilon)
        if not win:
            continue
        key = (deltas["delta_prec"], -deltas["delta_noise"])
        if key > best_key:
            best_key, best_value, best_deltas = key, value, deltas
    return best_value, best_deltas


def shadow_eval_numeric(
    mem: Any, cand: NumericCandidate, *, k: int, labels: LabelSet
) -> dict[str, Any]:
    """Run the candidate knob on best grid value vs OFF, attribute the
    precision/noise delta to it. Searches the grid maximizing (precision, -noise)
    vs the baseline off_value.

    Args:
        mem: Memory instance (passed to run_config)
        cand: NumericCandidate with field, off_value, on_value, grid, epsilon
        k: top-k for eval
        labels: LabelSet for eval

    Returns:
        dict with win, best_value, delta_prec, delta_noise (call-compatible
        with the boolean shadow.shadow_eval)
    """
    value, deltas = best_on_value(mem, cand, k=k, labels=labels)
    win = value != cand.off_value
    return {"win": win, "best_value": value, **deltas}
