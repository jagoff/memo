"""Citation-type feedback — nightly grounding stats → capture-time type weights.

If grounding data shows ``type=decision`` gets cited 3x more often than
``type=note``, capture should favor the higher-value type when a candidate's
classification is genuinely ambiguous. Two halves:

- ``compute_type_citation_stats`` (the dream ``capture_weights`` pass) joins
  ``grounding.log`` rows (``recall_id`` = 8-char id prefix) to each memory's
  type via the store, derives a per-type citation rate (rows with
  ``used_score >= _STRONG`` over that type's recalled observations), and turns
  the rates into multipliers normalized around 1.0 (rate / overall rate),
  clamped to ``[0.5, 2.0]``. Types with fewer than ``_MIN_OBSERVATIONS``
  recalled rows get weight 1.0 — no signal, no bias. Result persists to
  ``state_dir/capture/type_weights.json``.
- ``load_type_weights`` is the cheap capture-time reader: one small JSON read,
  ``{}`` on missing/corrupt, values re-clamped defensively. Consumed by
  ``capture.reweight_ambiguous_type`` (gated by MEMO_CAPTURE_TYPE_FEEDBACK).
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from memo.config import Config

_log = logging.getLogger(__name__)

# used_score at/above this counts as a citation — the same "actually USED in
# the answer" convention as eval_recall.harvest_labels(strong=0.5).
_STRONG = 0.5
# Types with fewer recalled observations than this get weight 1.0 (no signal).
_MIN_OBSERVATIONS = 5
_WEIGHT_MIN = 0.5
_WEIGHT_MAX = 2.0


def weights_path(cfg: Config) -> Path:
    return cfg.state_dir / "capture" / "type_weights.json"


def _resolve_type(mem: Any, prefix: str) -> str | None:
    """Memory type for an 8-char id prefix, or None (deleted / ambiguous)."""
    try:
        ids = mem.store.find_by_prefix(prefix, limit=2)
        if len(ids) != 1:
            return None  # missing or (vanishingly unlikely) ambiguous prefix
        row = mem.store.get(ids[0])
        t = (row or {}).get("type")
        return str(t) if t else None
    except Exception as exc:
        _log.debug("capture_weights: type lookup failed for %s: %s", prefix, exc)
        return None


def compute_type_citation_stats(cfg: Config, mem: Any) -> dict[str, Any]:
    """Join grounding.log cited ids → memory type; write the weights file.

    Returns the persisted payload
    ``{"computed_ts", "weights": {type: float}, "stats": {type: {...}}}``.
    Raises on write failure (the dream caller records it in
    ``receipt["errors"]``); per-row lookup failures are skipped (no signal).
    """
    from datetime import UTC, datetime

    from memo.dashboard_logs import read_grounding_log

    rows = read_grounding_log(cfg.state_dir, limit=4000)
    type_by_prefix: dict[str, str | None] = {}
    recalled: dict[str, int] = {}
    cited: dict[str, int] = {}
    for r in rows:
        rid = str(r.get("recall_id") or "")
        if len(rid) < 8:
            continue
        if rid not in type_by_prefix:
            type_by_prefix[rid] = _resolve_type(mem, rid)
        t = type_by_prefix[rid]
        if not t:
            continue
        recalled[t] = recalled.get(t, 0) + 1
        try:
            used = float(r.get("used_score") or 0.0)
        except (TypeError, ValueError):
            used = 0.0
        if used >= _STRONG:
            cited[t] = cited.get(t, 0) + 1

    stats: dict[str, dict[str, Any]] = {}
    for t, n in recalled.items():
        c = cited.get(t, 0)
        stats[t] = {"recalled": n, "cited": c, "rate": round(c / n, 4)}

    # Normalization center: the overall citation rate across types with
    # enough observations. weight = type_rate / center, so a type cited at
    # the corpus-average rate lands exactly on 1.0.
    qualified = {t: s for t, s in stats.items() if int(s["recalled"]) >= _MIN_OBSERVATIONS}
    total_recalled = sum(int(s["recalled"]) for s in qualified.values())
    total_cited = sum(int(s["cited"]) for s in qualified.values())
    center = (total_cited / total_recalled) if total_recalled else 0.0

    weights: dict[str, float] = {}
    for t, s in stats.items():
        if t not in qualified or center <= 0.0:
            weights[t] = 1.0  # <5 observations (or zero citations anywhere): no bias
            continue
        w = float(s["rate"]) / center
        weights[t] = round(min(_WEIGHT_MAX, max(_WEIGHT_MIN, w)), 4)

    payload: dict[str, Any] = {
        "computed_ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "weights": weights,
        "stats": stats,
    }
    dest = weights_path(cfg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write — capture reads this file concurrently with the nightly
    # pass. The tmp name is pid-suffixed so two concurrent dream runs can't
    # interleave writes into one tmp file (same pattern as presence.py).
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, dest)
    return payload


def load_type_weights(cfg: Config) -> dict[str, float]:
    """Capture-time reader: ``{type: multiplier}``, ``{}`` on missing/corrupt.

    One small JSON read, no store/MLX. Values are re-clamped to
    ``[0.5, 2.0]`` so a hand-edited or stale file can never bias harder than
    the nightly pass would."""
    try:
        raw = json.loads(weights_path(cfg).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    w = raw.get("weights") if isinstance(raw, dict) else None
    if not isinstance(w, dict):
        return {}
    out: dict[str, float] = {}
    for t, v in w.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[str(t)] = min(_WEIGHT_MAX, max(_WEIGHT_MIN, float(v)))
    return out
