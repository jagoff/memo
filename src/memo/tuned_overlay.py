"""Auto-tuned MEMO_* params overlay — written by `memo dream tune`, read by
`flags.flag()` with precedence env > overlay > default.

Machine-local, never committed. Deleting the file restores pure defaults. The
overlay only ever holds numeric ranking params the nightly tuner is allowed to
move; `_meta` carries provenance + the previous values for one-step rollback.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_FILENAME = "tuned_params.json"

# path -> (mtime, param->str-value) — keeps `flag()` off disk on the hot recall path.
_cache: dict[str, tuple[float, dict[str, str]]] = {}


def overlay_path(state_dir: Path) -> Path:
    return Path(state_dir) / _FILENAME


def read_overlay(state_dir: Path) -> dict[str, Any]:
    """Full overlay document (incl. `_meta`); {} when missing or corrupt."""
    try:
        doc = json.loads(overlay_path(state_dir).read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _params_only(doc: dict[str, Any]) -> dict[str, float]:
    return {
        k: float(v)
        for k, v in doc.items()
        if k != "_meta" and isinstance(v, (int, float)) and not isinstance(v, bool)
    }


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
    vals = {k: str(v) for k, v in _params_only(read_overlay(Path(sd))).items()}
    _cache[str(p)] = (mtime, vals)
    return vals


def write_overlay(state_dir: Path, params: dict[str, float], meta: dict[str, Any]) -> None:
    """Write the overlay, stashing the current params under ``_meta.prev``."""
    sd = Path(state_dir)
    sd.mkdir(parents=True, exist_ok=True)
    prev = _params_only(read_overlay(sd))
    doc: dict[str, Any] = dict(params)
    doc["_meta"] = {**meta, "prev": prev}
    overlay_path(sd).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    _cache.pop(str(overlay_path(sd)), None)


def rollback_overlay(state_dir: Path) -> dict[str, float] | None:
    """Restore ``_meta.prev``; returns the restored params, or None if none."""
    sd = Path(state_dir)
    prev = (read_overlay(sd).get("_meta") or {}).get("prev")
    if not isinstance(prev, dict) or not prev:
        return None
    restored = {k: float(v) for k, v in prev.items()}
    write_overlay(sd, restored, {"set_by": "rollback"})
    return restored
