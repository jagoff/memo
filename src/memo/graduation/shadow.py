"""Offline shadow evaluator: measure a candidate ON vs OFF on the eval corpus,
attributing the precision/noise delta to the candidate alone. No behavior
change — this only reads the index through eval_recall."""
from __future__ import annotations

import dataclasses
from typing import Any

from memo.eval_recall import Cfg, LabelSet, Row, default_configs, run_config
from memo.graduation.registry import Candidate


def decide_win(off: Row, on: Row, epsilon: float) -> tuple[bool, dict[str, float]]:
    """Determine if a flag ON is a win: precision gain >= epsilon AND noise does not rise.

    Args:
        off: Row with metrics for flag OFF
        on: Row with metrics for flag ON
        epsilon: minimum precision gain required to consider it a win

    Returns:
        (win, deltas) where win is bool and deltas is dict with delta_prec and delta_noise
    """
    delta_prec = round(on.precision_at_k - off.precision_at_k, 4)
    delta_noise = round(on.noise_at_k - off.noise_at_k, 4)
    win = delta_prec >= epsilon and delta_noise <= 0.0
    return win, {"delta_prec": delta_prec, "delta_noise": delta_noise}


def shadow_eval(mem: Any, cand: Candidate, *, k: int, labels: LabelSet) -> dict[str, Any]:
    """Run the candidate flag ON vs OFF, attribute the precision/noise delta to it.

    Args:
        mem: Memory instance (passed to run_config)
        cand: Candidate with flag, on_flags, and epsilon
        k: top-k for eval
        labels: LabelSet for eval

    Returns:
        dict with win, delta_prec, delta_noise
    """
    base = default_configs()[0]  # A vec/0.60/keep — the live default shape
    off_flags = {key: "0" for key in cand.on_flags}
    off_cfg = _with_flags(base, f"{cand.flag}-off", off_flags)
    on_cfg = _with_flags(base, f"{cand.flag}-on", dict(cand.on_flags))
    off = run_config(mem, off_cfg, k, labels)
    on = run_config(mem, on_cfg, k, labels)
    win, deltas = decide_win(off, on, cand.epsilon)
    return {"win": win, **deltas}


def _with_flags(base: Cfg, name: str, flag_overrides: dict[str, str]) -> Cfg:
    """Create a new Cfg by overriding flags on a base config.

    Args:
        base: Base Cfg to clone from
        name: New name for the config
        flag_overrides: Dict of flag name -> value to override

    Returns:
        New Cfg instance
    """
    return dataclasses.replace(base, name=name, flag_overrides=flag_overrides)
