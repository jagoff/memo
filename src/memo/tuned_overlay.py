"""Auto-tuned MEMO_* params overlay — written by `memo dream tune`, read by
`flags.flag()` with precedence env > overlay > default.

Machine-local, never committed. Deleting the file restores pure defaults. The
overlay holds scalar tuning params (float ranking knobs like
``MEMO_RECALL_MIN_SIM``, plus boolean/string levers the retrieval tuner is
allowed to flip, e.g. ``MEMO_GRAPH_RETRIEVAL_ENABLED`` / ``MEMO_RECALL_MODE``);
`_meta` carries provenance + the previous values for one-step rollback.
Every value round-trips through ``flag()``'s per-kind coercion, so it is stored
as its native JSON scalar and surfaced to ``flag()`` as a coercible string.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_FILENAME = "tuned_params.json"

# path -> (mtime, param->str-value) — keeps `flag()` off disk on the hot recall path.
_cache: dict[str, tuple[float, dict[str, str]]] = {}

# JSON scalar types the overlay is allowed to carry (bool is a subclass of int,
# so it is covered, but we branch on it first when stringifying for `flag()`).
_SCALAR = (bool, int, float, str)


def overlay_path(state_dir: Path) -> Path:
    return Path(state_dir) / _FILENAME


def read_overlay(state_dir: Path) -> dict[str, Any]:
    """Full overlay document (incl. `_meta`); {} when missing or corrupt."""
    try:
        doc = json.loads(overlay_path(state_dir).read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def params_version(state_dir: Path) -> str:
    """Stable short hash of the active scalar overlay params. Returns ``"base"``
    when the overlay is empty/missing (the identity config). Order-independent,
    so re-serialising the same params never changes the version."""
    params = _scalar_params(read_overlay(Path(state_dir)))
    if not params:
        return "base"
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _scalar_params(doc: dict[str, Any]) -> dict[str, Any]:
    """Every non-`_meta` scalar param, native type preserved. This is the full
    set the overlay carries (floats + bools + strings)."""
    return {k: v for k, v in doc.items() if k != "_meta" and isinstance(v, _SCALAR)}


def _params_only(doc: dict[str, Any]) -> dict[str, float]:
    """Numeric-only params (the float ranking knobs). Kept for callers that do
    float math on the overlay and must ignore boolean/string levers."""
    return {
        k: float(v)
        for k, v in doc.items()
        if k != "_meta" and isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def _to_flag_str(v: Any) -> str:
    """Stringify a scalar for `flag()` per-kind coercion. Booleans map to the
    canonical ``1``/``0`` that the bool coercer accepts (``str(True)`` == "True"
    would be rejected)."""
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)


def overlay_values(src: Mapping[str, str]) -> dict[str, str]:
    """param-name -> string value, for `flag()` resolution. Resolved from
    ``src["MEMO_STATE_DIR"]``, mtime-cached. {} when unset/missing/corrupt."""
    sd = src.get("MEMO_STATE_DIR")
    if not sd:
        return {}
    p = overlay_path(Path(sd))
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return {}
    cached = _cache.get(str(p))
    if cached and cached[0] == mtime:
        return cached[1]
    vals = {k: _to_flag_str(v) for k, v in _scalar_params(read_overlay(Path(sd))).items()}
    _cache[str(p)] = (mtime, vals)
    return vals


def write_overlay(state_dir: Path, params: dict[str, Any], meta: dict[str, Any]) -> None:
    """Write the overlay, stashing the current params under ``_meta.prev``.

    ``params`` may hold any JSON scalar (float knobs and/or boolean/string
    levers); the previous scalar set is preserved for one-step rollback.
    """
    sd = Path(state_dir)
    sd.mkdir(parents=True, exist_ok=True)
    prev = _scalar_params(read_overlay(sd))
    doc: dict[str, Any] = dict(params)
    doc["_meta"] = {**meta, "prev": prev}
    overlay_path(sd).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    _cache.pop(str(overlay_path(sd)), None)


def rollback_overlay(state_dir: Path) -> dict[str, Any] | None:
    """Restore ``_meta.prev``; returns the restored params, or None if none.
    Native scalar types are preserved (a rolled-back boolean stays a boolean)."""
    sd = Path(state_dir)
    prev = (read_overlay(sd).get("_meta") or {}).get("prev")
    if not isinstance(prev, dict) or not prev:
        return None
    restored = dict(_scalar_params(prev))
    write_overlay(sd, restored, {"set_by": "rollback"})
    return restored
