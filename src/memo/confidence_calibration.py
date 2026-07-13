"""Confidence calibration: bin memories by PREDICTED confidence
(``memory_health.confidence`` band), compare to grounding-observed usefulness,
and emit a monotonic recalibration map. The render layer (the recall confidence
gate) looks the map up by a hit's SCORE-derived band — one file-cached read, no
store read, no MLX on the 5s hot path. The offline join (this module's
``build_calibration``, run nightly in dream) is the only place the stored
confidence is consulted."""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

_FILENAME = "confidence_calibration.json"
_BANDS = ("low", "med", "high")
# Canonical iteration order for the monotone projection — low -> med -> high.
# Correctness of _monotonic_map depends on this order, not on dict insertion
# order (dicts happen to preserve insertion order in Python, but that is not
# the contract this module relies on).
_BAND_ORDER: tuple[str, ...] = ("low", "med", "high")

_log = logging.getLogger(__name__)

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
    except (AttributeError, OSError, sqlite3.Error) as exc:
        # A broken join must be observable, not silently degraded to a
        # plausible-looking identity map — log so this shows up, don't guess.
        _log.warning("confidence_calibration: mem.store.all_ids() failed: %s", exc)
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
    """Pool-Adjacent-Violators (isotonic regression) over the bands in their
    canonical low -> med -> high order, producing a map whose *effective*
    observed rate (``bins[remap[b]]["observed"]``) is provably non-decreasing
    for ANY input — including an adversarial middle spike
    (e.g. low=0.3, med=0.9, high=0.2) or a fully decreasing sequence.

    Under-observed bands (n < min_bin) are substituted with their nearest
    already-decided neighbor's value *before* the PAVA pass runs (not
    corrected after the fact per-pair), so a sparse band can never introduce
    a violation of its own. With no neighbor yet decided, a sparse band
    falls back to its own predicted (identity) rate.

    PAVA works by pooling adjacent bands whose raw values would otherwise
    decrease into a single block, averaging within the block, and repeating
    until the whole sequence is non-decreasing block-by-block. Each band is
    then mapped to the *lowest-order* band name within its own block — since
    blocks are non-decreasing and same-block bands share one pooled value,
    this keeps ``remap`` monotone by construction.
    """
    n = {b: bins[b]["n"] for b in _BAND_ORDER}

    # Sparse-band fallback: substitute values for under-observed bands with
    # the nearest preceding well-observed band's value (falling back to that
    # band's own predicted/identity rate when no preceding value exists yet).
    # This runs BEFORE PAVA so a sparse band cannot be the source of a
    # violation the pooling pass would otherwise have to "fix".
    values: list[float] = []
    last_good: float | None = None
    for b in _BAND_ORDER:
        if n[b] >= min_bin:
            v = bins[b]["observed"]
            last_good = v
        else:
            v = last_good if last_good is not None else bins[b]["predicted"]
        values.append(v)

    # Pool-Adjacent-Violators: iterative merge of adjacent blocks whose
    # averages violate non-decreasing order. Each block is (sum, count,
    # start_index); the pooled value of a block is sum/count.
    blocks: list[list[float | int]] = [[v, 1, i] for i, v in enumerate(values)]
    i = 0
    while i < len(blocks) - 1:
        avg_cur = blocks[i][0] / blocks[i][1]
        avg_next = blocks[i + 1][0] / blocks[i + 1][1]
        if avg_cur > avg_next:
            merged_sum = blocks[i][0] + blocks[i + 1][0]
            merged_count = blocks[i][1] + blocks[i + 1][1]
            blocks[i] = [merged_sum, merged_count, blocks[i][2]]
            del blocks[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1

    # Expand blocks back to per-band pooled values, then map every band to
    # the name of the FIRST (lowest-order) band in its block.
    band_of_block_start: list[str] = []
    for block in blocks:
        _, block_count, start_idx = block
        band_of_block_start.extend([_BAND_ORDER[int(start_idx)]] * int(block_count))

    remap: dict[str, str] = {
        band: band_of_block_start[idx] for idx, band in enumerate(_BAND_ORDER)
    }
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
