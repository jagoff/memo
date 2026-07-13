"""Confidence calibration: bin memories by PREDICTED confidence
(``memory_health.confidence`` band), compare to grounding-observed usefulness,
and emit a monotonic recalibration map. The render layer (the recall confidence
gate) looks the map up by a hit's SCORE-derived band — one file-cached read, no
store read, no MLX on the 5s hot path. The offline join (this module's
``build_calibration``, run nightly in dream) is the only place the stored
confidence is consulted."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_FILENAME = "confidence_calibration.json"
_BANDS = ("low", "med", "high")

# path -> (mtime, doc) — keeps the recall path off disk after the first read.
_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def predicted_band(confidence: float) -> str:
    """Coarse band from a stored confidence — same thresholds as
    ``recall_logic._conf_band`` so calibration + render speak one vocabulary."""
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.5:
        return "med"
    return "low"


def _calibration_path(state_dir: Path) -> Path:
    return Path(state_dir) / _FILENAME


def build_calibration(state_dir: Path, mem: Any, *, min_bin: int = 5) -> dict[str, Any]:
    """Join grounding uses to each recalled memory's stored confidence band and
    compute observed usefulness per band; derive a monotonic remap. Bands with
    fewer than ``min_bin`` observations are identity-mapped."""
    from memo.dashboard import grounding_used, read_grounding_log

    rows = read_grounding_log(Path(state_dir))
    # recall_id is stored as an 8-char prefix; map it back to a full id.
    try:
        all_ids = mem.store.all_ids()
    except Exception:
        all_ids = []
    by_prefix = {i[:8]: i for i in all_ids}

    # (used?, confidence) per grounding row that resolves to a known memory.
    totals: dict[str, list[int]] = {b: [0, 0] for b in _BANDS}  # [grounded, total]
    ids_needed = {
        by_prefix.get(str(r.get("recall_id") or "")[:8])
        for r in rows
        if by_prefix.get(str(r.get("recall_id") or "")[:8])
    }
    health = mem.store.get_health_batch([i for i in ids_needed if i]) if ids_needed else {}
    for r in rows:
        full = by_prefix.get(str(r.get("recall_id") or "")[:8])
        if not full:
            continue
        conf = float((health.get(full) or {}).get("confidence", 1.0))
        band = predicted_band(conf)
        totals[band][1] += 1
        if grounding_used(r):
            totals[band][0] += 1

    bins = {
        b: {
            "predicted": {"low": 0.25, "med": 0.6, "high": 0.9}[b],
            "observed": (g / t if t else 0.0),
            "n": t,
        }
        for b, (g, t) in totals.items()
    }
    return {"bins": bins, "map": _monotonic_map(bins, min_bin=min_bin)}


def _monotonic_map(bins: dict[str, dict[str, Any]], *, min_bin: int) -> dict[str, str]:
    """Isotonic-style remap: a band whose observed usefulness is out of order
    (e.g. 'high' grounds LESS than 'med') is demoted so observed rates rise
    monotonically low<=med<=high. Under-observed bands (n < min_bin) stay
    identity — insufficient evidence to correct."""
    remap: dict[str, str] = {b: b for b in _BANDS}
    observed = {b: bins[b]["observed"] for b in _BANDS}
    n = {b: bins[b]["n"] for b in _BANDS}
    # demote 'high' to 'med' when it grounds no better than 'med' (both observed).
    if n["high"] >= min_bin and n["med"] >= min_bin and observed["high"] < observed["med"]:
        remap["high"] = "med"
    # demote 'med' to 'low' when it grounds no better than 'low'.
    if n["med"] >= min_bin and n["low"] >= min_bin and observed["med"] < observed["low"]:
        remap["med"] = "low"
    # a 'high' that even falls below 'low' collapses to 'low'.
    if n["high"] >= min_bin and n["low"] >= min_bin and observed["high"] < observed["low"]:
        remap["high"] = "low"
    return remap


def save_calibration(state_dir: Path, doc: dict[str, Any]) -> None:
    sd = Path(state_dir)
    sd.mkdir(parents=True, exist_ok=True)
    _calibration_path(sd).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    _cache.pop(str(_calibration_path(sd)), None)


def load_calibration(state_dir: Path) -> dict[str, Any]:
    p = _calibration_path(Path(state_dir))
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return {}
    cached = _cache.get(str(p))
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
        doc = doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError):
        doc = {}
    _cache[str(p)] = (mtime, doc)
    return doc


def recalibrated_band(state_dir: Path, score_band: str) -> str:
    """Render-time lookup: map a hit's score-derived band through the calibration
    map (identity when no map/entry). One mtime-cached file read; hot-path safe."""
    return str((load_calibration(state_dir).get("map") or {}).get(score_band, score_band))
