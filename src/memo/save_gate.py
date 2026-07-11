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


def _preset_map() -> tuple[str | None, dict[str, str]]:
    """Parse MEMO_SAVE_GATE_PRESETS → (global_default, per_type_map).

    Accepts two forms: a bare preset name (e.g. ``strict``) applies to ALL types,
    or a JSON object (``{"note": "strict"}``) sets per-type presets. A bare name
    is the natural/documented usage — parsing it as the global default avoids the
    silent no-op where ``json.loads("strict")`` failed and everything defaulted.
    """
    raw = (flag_str("MEMO_SAVE_GATE_PRESETS") or "").strip()
    if not raw:
        return None, {}
    if raw in PRESETS:  # bare preset name → global default for every type
        return raw, {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None, {}
    if not isinstance(data, dict):
        return None, {}
    return None, {str(k): str(v) for k, v in data.items()}


def resolve_gate(type_: str) -> GatePolicy:
    """GatePolicy for a memory type. Unset/unlisted/unknown ⇒ `balanced` (no-op)."""
    global_default, per_type = _preset_map()
    name = per_type.get(type_, global_default or "balanced")
    return PRESETS.get(name, _DEFAULT)
