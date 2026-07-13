"""Flip a proven candidate ON (or revert it) by merging a single boolean into
the tuned overlay. flags.flag() precedence is env > markdown > overlay > default,
so a flip is picked up unless a human pinned the flag by env/markdown, and a
revert is one key deleted."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from memo.tuned_overlay import read_overlay, write_overlay


def _scalars(state_dir: Path) -> dict[str, Any]:
    return {k: v for k, v in read_overlay(state_dir).items() if k != "_meta"}


def is_flipped_on(state_dir: Path, flag: str) -> bool:
    return _scalars(state_dir).get(flag) is True


def flip_on(state_dir: Path, flag: str, *, evidence: dict[str, Any]) -> None:
    params = _scalars(state_dir)
    params[flag] = True
    write_overlay(state_dir, params, {"set_by": "graduation-controller", "evidence": evidence})


def revert(state_dir: Path, flag: str) -> None:
    params = _scalars(state_dir)
    params.pop(flag, None)
    write_overlay(state_dir, params, {"set_by": "graduation-revert", "flag": flag})


def overlay_value(state_dir: Path, flag: str) -> Any | None:
    """The current overlay scalar for ``flag`` (numeric or bool), or None."""
    return _scalars(state_dir).get(flag)


def flip_numeric(state_dir: Path, flag: str, value: float, *, evidence: dict[str, Any]) -> None:
    """Flip a numeric knob ON by merging its proven value into the overlay
    (preserving other tuned knobs). Revert = drop the key via ``revert`` →
    the built-in default returns."""
    params = _scalars(state_dir)
    params[flag] = value
    write_overlay(state_dir, params, {"set_by": "graduation-controller", "evidence": evidence})
