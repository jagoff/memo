"""Per-type save-gate presets (gate-presets).

A preset maps a memory `type_` to a GatePolicy that tunes how strict the save
path is about near-duplicates. Seeded so an unset MEMO_SAVE_GATE_PRESETS ⇒ every
type resolves to `balanced` ⇒ today's behavior (dedup warns, never refuses).
Pure config resolution; no I/O.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from memo.flags import flag_str


@dataclass(frozen=True)
class GatePolicy:
    dedup_mode: str   # "warn" (log, admit — current) | "refuse" (raise) | "off" (skip check)
    quality_mode: str  # "warn" | "strict"  (reserved; wired later if a quality-refuse is added)
    ground: bool       # reserved: consult grounding at save (future)


PRESETS: dict[str, GatePolicy] = {
    "balanced": GatePolicy(dedup_mode="warn", quality_mode="warn", ground=False),
    "strict": GatePolicy(dedup_mode="refuse", quality_mode="strict", ground=True),
    "permissive": GatePolicy(dedup_mode="off", quality_mode="warn", ground=False),
}

_DEFAULT = PRESETS["balanced"]


def _preset_map() -> dict[str, str]:
    raw = flag_str("MEMO_SAVE_GATE_PRESETS") or ""
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def resolve_gate(type_: str) -> GatePolicy:
    """GatePolicy for a memory type. Unset/unlisted/unknown ⇒ `balanced` (no-op)."""
    name = _preset_map().get(type_, "balanced")
    return PRESETS.get(name, _DEFAULT)
