"""dream_shadow — measure-only ("shadow") evaluation of opt-in dream phases.

Generalises the single-flag ``ask_gaps.py`` shadow pattern ("report-only: log
what it WOULD do; a human flips the flag after review") into a reusable harness
for every ``shadow``-kind gate (see ``dream_flags.GATES``). A shadow phase runs
in *measure-only* mode: we compute what it WOULD change and its cost into a
durable, comparable ledger **without mutating production** (no overlay write, no
markdown-config write, no corpus/index change). Evidence accrues per night; a
human promotes a flag only after ``MEMO_SHADOW_REVIEW_NIGHTS`` of consecutive
clean nights (:func:`promote`). Auto-graduation is deliberately never reached.

Two measurement entry points:

- :func:`record_recall_shadow` — for recall-measurable shadow flags (e.g. the
  graph-ranking signal). The caller supplies the offline ON/OFF eval metrics
  (measured non-invasively through ``measure_flag``'s ``flag_overrides`` env-pin
  seam, which restores the env); we derive the comparable axis
  (Δprecision / −Δnoise) and cost (ON p50) and append.
- :func:`maybe_shadow` — for pass-behavior shadow flags (phases whose impact is
  not recall-eval-measurable). Runs the pass's ``runner(dry_run=True)`` — which
  MUST be side-effect-free — records the declared ``shadow_metric`` plus wall
  cost, and asserts non-mutation via the corpus fingerprint + overlay version.

Also home to :func:`latency_ceiling_gate` (the absolute p50 guard used by
auto-graduation and shadow promotion — defence against the graph-signal
~6272ms-vs-48ms recall-hook latency catastrophe that the *relative* headroom
gate cannot catch) and :func:`migrate_reclassified_overlay` (one-time cleanup
that un-graduates a shadow-reclassified flag the flag-graduation pass had
previously auto-set in the overlay).

Best-effort throughout: every public helper swallows and degrades rather than
raising into the nightly pipeline. No MLX imports.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from memo.flags import flag_bool, flag_int

_log = logging.getLogger(__name__)

_SHADOW_DIR = "shadow"
_LEDGER = "shadow_ledger.jsonl"
_STATE = "shadow_state.json"
_LEDGER_CAP = 2000
_LEDGER_SIZE_LIMIT = 2_000_000

# Verdicts that count as a "clean" night for the consecutive-clean review streak.
_CLEAN = frozenset({"win", "neutral", "clean"})
# Observation kinds (as opposed to lifecycle "event" rows) in the ledger.
_OBS_KINDS = frozenset({"recall", "pass"})


# --- flag accessors -------------------------------------------------------------


def _review_nights() -> int:
    v = flag_int("MEMO_SHADOW_REVIEW_NIGHTS")
    return 5 if v is None else max(1, v)


def _latency_ceiling() -> float:
    v = flag_int("MEMO_FLAG_GRADUATION_LATENCY_CEILING_MS")
    return 1500.0 if v is None else float(v)


def _shadow_enabled() -> bool:
    return flag_bool("MEMO_DREAM_SHADOW_ENABLED")


# --- pure gate ------------------------------------------------------------------


def latency_ceiling_gate(measured_ms: float, ceiling_ms: float) -> bool:
    """True when ``measured_ms`` is within the absolute latency ceiling (i.e. it
    is safe to graduate on latency grounds).

    A non-positive ``ceiling_ms`` disables the gate (always passes) — matching
    the ``ceil > 0`` guard the auto-graduation win loop applies. This is the
    absolute backstop the relative headroom ratio cannot provide: a flag whose
    p50 balloons from ~48ms to ~6272ms on the recall hook passes any relative
    check against an equally-slow OFF baseline but must never graduate.
    """
    if ceiling_ms <= 0:
        return True
    return measured_ms <= ceiling_ms


# --- paths + io -----------------------------------------------------------------


def _shadow_root(state_dir: Path) -> Path:
    return Path(state_dir) / "dream" / _SHADOW_DIR


def _ledger_path(state_dir: Path) -> Path:
    return _shadow_root(state_dir) / _LEDGER


def _state_path(state_dir: Path) -> Path:
    return _shadow_root(state_dir) / _STATE


def _append_ledger(state_dir: Path, entry: dict[str, Any]) -> None:
    from memo.dashboard_logs import _write_jsonl_entry

    row = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **entry}
    _write_jsonl_entry(
        _ledger_path(state_dir), row, cap=_LEDGER_CAP, size_limit=_LEDGER_SIZE_LIMIT
    )


def read_observations(
    state_dir: Path, *, flag: str | None = None, limit: int = _LEDGER_CAP
) -> list[dict[str, Any]]:
    """Ledger observation rows (kind ``recall``/``pass``), oldest first, optionally
    filtered to one ``flag``. Lifecycle ``event`` rows are excluded."""
    from memo.dashboard_logs import _read_jsonl

    rows = _read_jsonl(_ledger_path(state_dir), limit=limit)
    out = [r for r in rows if r.get("kind") in _OBS_KINDS]
    if flag is not None:
        out = [r for r in out if r.get("flag") == flag]
    return out


def _load_state(state_dir: Path) -> dict[str, Any]:
    try:
        doc = json.loads(_state_path(state_dir).read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(state_dir: Path, state: dict[str, Any]) -> None:
    p = _state_path(state_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")


# --- small pure helpers ---------------------------------------------------------


def _today() -> str:
    return date.today().isoformat()


def _p50(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2:
        return round(float(s[mid]), 1)
    return round((float(s[mid - 1]) + float(s[mid])) / 2.0, 1)


def _recall_verdict(delta_precision: float, delta_noise: float) -> str:
    """win / lose / neutral on the (Δprecision, −Δnoise) comparable axis."""
    if delta_precision > 0 or (delta_precision == 0 and delta_noise < 0):
        return "win"
    if delta_precision < 0 or (delta_precision == 0 and delta_noise > 0):
        return "lose"
    return "neutral"


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


# --- gate lookups (deferred; dream_flags imports this module) --------------------


def _gate(flag: str) -> Any:
    try:
        from memo.dream_flags import GATES

        return GATES.get(flag)
    except Exception as exc:  # pragma: no cover - import guard
        _log.debug("dream_shadow: gate lookup failed for %s: %s", flag, exc)
        return None


def _is_shadow_flag(flag: str) -> bool:
    return getattr(_gate(flag), "kind", "") == "shadow"


def _shadow_metric(flag: str) -> str:
    return str(getattr(_gate(flag), "shadow_metric", "") or "")


def _is_latency_metric(metric: str) -> bool:
    return "latency" in metric.lower()


def _env_to_path() -> dict[str, str]:
    """Inverse of ``catalog.path_to_env`` — env-var name -> dotted config key."""
    try:
        from memo.tui.config.catalog import path_to_env

        return {env: path for path, env in path_to_env().items()}
    except Exception as exc:
        _log.debug("dream_shadow: path_to_env inverse failed: %s", exc)
        return {}


def _fingerprint(mem: Any) -> str | None:
    if mem is None:
        return None
    try:
        from memo.dream_utils import _corpus_fingerprint

        return _corpus_fingerprint(mem)
    except Exception as exc:
        _log.debug("dream_shadow: corpus fingerprint failed: %s", exc)
        return None


def _human_or_live(cfg: Any, flag: str) -> bool:
    """A flag the human already pinned (env/config) or that is live in the
    overlay is out of the shadow pool — we only shadow dark, unpinned flags."""
    try:
        from memo.dream_flags import human_owned
        from memo.tuned_overlay import read_overlay

        if human_owned(flag):
            return True
        return bool(read_overlay(cfg.state_dir).get(flag))
    except Exception as exc:
        _log.debug("dream_shadow: ownership check failed for %s: %s", flag, exc)
        return False


# --- recording ------------------------------------------------------------------


def _bump_state(state_dir: Path, obs: dict[str, Any]) -> None:
    """Update the per-flag rollup: total nights + the consecutive-clean streak.

    Idempotent within a night — a re-measurement of the same ``night`` updates
    the last verdict but never double-counts nights or the streak."""
    state = _load_state(state_dir)
    flags = state.setdefault("flags", {})
    flag = obs["flag"]
    entry = flags.setdefault(flag, {"first_night": obs["night"], "nights": 0, "streak": 0})
    clean = obs.get("verdict") in _CLEAN
    if obs["night"] != entry.get("last_night"):
        entry["nights"] = int(entry.get("nights", 0)) + 1
        entry["streak"] = int(entry.get("streak", 0)) + 1 if clean else 0
        entry["last_night"] = obs["night"]
    entry["last_verdict"] = obs.get("verdict")
    _save_state(state_dir, state)


def _record(state_dir: Path, obs: dict[str, Any]) -> None:
    _append_ledger(state_dir, obs)
    _bump_state(state_dir, obs)


def record_recall_shadow(
    state_dir: Path,
    *,
    flag: str,
    off: dict[str, float],
    on: dict[str, float],
    night: str | None = None,
) -> dict[str, Any] | None:
    """Record one recall-shadow observation from precomputed ON/OFF eval metrics.

    Comparable axis: Δprecision (primary) and Δnoise (tiebreak, lower better);
    cost is the ON p50. Never mutates anything but the shadow ledger/state.
    Returns the observation fragment (for the caller's ``receipt['shadow']``)."""
    try:
        d_prec = round(float(on.get("precision_at_k", 0.0)) - float(off.get("precision_at_k", 0.0)), 4)
        d_noise = round(float(on.get("noise_at_k", 0.0)) - float(off.get("noise_at_k", 0.0)), 4)
        cost_ms = round(float(on.get("latency_ms_p50", 0.0)), 1)
        obs: dict[str, Any] = {
            "kind": "recall",
            "flag": flag,
            "night": night or _today(),
            "delta": d_prec,
            "delta_precision": d_prec,
            "delta_noise": d_noise,
            "cost_ms": cost_ms,
            "off": dict(off),
            "on": dict(on),
            "verdict": _recall_verdict(d_prec, d_noise),
        }
        _record(state_dir, obs)
        return obs
    except Exception as exc:
        _log.debug("dream_shadow: record_recall_shadow failed for %s: %s", flag, exc)
        return None


def maybe_shadow(
    cfg: Any,
    mem: Any,
    *,
    flag: str,
    runner: Callable[..., dict[str, Any] | None],
    receipt: dict[str, Any],
    metric: str,
    baseline_getter: Callable[[Any, Any], Any] | None = None,
    night: str | None = None,
) -> dict[str, Any] | None:
    """Shadow-measure a pass-behavior flag: run ``runner(dry_run=True)`` and
    record its declared ``metric`` + wall cost WITHOUT mutating production.

    Gate: ``MEMO_DREAM_SHADOW_ENABLED`` on, ``flag`` is shadow-kind, and the
    human has not pinned/graduated it. ``runner(dry_run=True)`` MUST be
    side-effect-free; non-mutation is asserted via the corpus fingerprint +
    overlay ``params_version`` — a run that moved either is recorded as a
    ``lose`` (leak). Stashes the fragment under ``receipt['shadow'][flag]``.
    Best-effort: returns None (and records nothing) when gated off or on error.
    """
    try:
        if not _shadow_enabled() or not _is_shadow_flag(flag) or _human_or_live(cfg, flag):
            return None
        from memo.tuned_overlay import params_version

        fp_before = _fingerprint(mem)
        ver_before = params_version(cfg.state_dir)
        t0 = time.perf_counter()
        fragment = runner(dry_run=True) or {}
        cost_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        fp_after = _fingerprint(mem)
        ver_after = params_version(cfg.state_dir)
        mutated = (fp_before is not None and fp_before != fp_after) or (ver_before != ver_after)

        value = fragment.get(metric) if isinstance(fragment, dict) else None
        baseline: Any = None
        if baseline_getter is not None:
            try:
                baseline = baseline_getter(cfg, mem)
            except Exception as exc:
                _log.debug("dream_shadow: baseline_getter failed for %s: %s", flag, exc)
                baseline = None
        v_num, b_num = _num(value), _num(baseline)
        if v_num is not None and b_num is not None:
            delta: float | None = round(v_num - b_num, 4)
        else:
            delta = v_num
        obs: dict[str, Any] = {
            "kind": "pass",
            "flag": flag,
            "night": night or _today(),
            "metric": metric,
            "value": value,
            "baseline": baseline,
            "delta": delta,
            "cost_ms": cost_ms,
            "mutated": bool(mutated),
            "verdict": "lose" if mutated else "clean",
        }
        _record(cfg.state_dir, obs)
        receipt.setdefault("shadow", {})[flag] = obs
        return obs
    except Exception as exc:
        _log.debug("dream_shadow: maybe_shadow failed for %s: %s", flag, exc)
        return None


# --- review / summary -----------------------------------------------------------


def shadow_summary(state_dir: Path, flag: str) -> dict[str, Any]:
    """Per-flag review rollup: nights of evidence, consecutive-clean streak, mean
    Δ, cost p50, last verdict, and whether the flag is review-ready (enough
    consecutive-clean nights and no decision yet)."""
    obs = read_observations(state_dir, flag=flag)
    entry = _load_state(state_dir).get("flags", {}).get(flag, {})
    review_nights = _review_nights()
    deltas = [d for d in (_num(o.get("delta")) for o in obs) if d is not None]
    costs = [c for c in (_num(o.get("cost_ms")) for o in obs) if c is not None]
    nights = int(entry.get("nights", 0))
    streak = int(entry.get("streak", 0))
    decision = entry.get("decision")
    return {
        "flag": flag,
        "kind": getattr(_gate(flag), "kind", ""),
        "metric": _shadow_metric(flag),
        "nights": nights,
        "streak": streak,
        "review_nights": review_nights,
        "mean_delta": round(sum(deltas) / len(deltas), 4) if deltas else 0.0,
        "cost_p50": _p50(costs),
        "last_verdict": entry.get("last_verdict"),
        "decision": decision,
        "review_ready": decision is None and streak >= review_nights,
    }


def review_rows(state_dir: Path, gates: dict[str, Any]) -> list[dict[str, Any]]:
    """One :func:`shadow_summary` per shadow-kind gate — the ``memo dream shadow
    --status`` table source. Pure over the passed ``gates`` mapping."""
    shadow_flags = sorted(
        name for name, g in gates.items() if getattr(g, "kind", "") == "shadow"
    )
    return [shadow_summary(state_dir, name) for name in shadow_flags]


# --- decisions ------------------------------------------------------------------


def _mark_decision(state_dir: Path, flag: str, decision: str, *, reason: str = "") -> None:
    state = _load_state(state_dir)
    entry = state.setdefault("flags", {}).setdefault(flag, {})
    entry["decision"] = decision
    entry["decision_reason"] = reason
    entry["decided_at"] = _today()
    _save_state(state_dir, state)


def migrate_reclassified_overlay(state_dir: Path, gates: dict[str, Any]) -> list[str]:
    """One-time cleanup: strip any shadow-kind flag the flag-graduation pass had
    auto-set in the overlay (``_meta.set_by == 'dream-flag-graduation'``).

    Scoped to that provenance so it never fights the graph-weight tuner's own
    online-revert (``set_by == 'dream-curated-graph'``) — an overlay last written
    by any other writer is left entirely alone. Each removal is logged to the
    ledger as ``auto_reverted_on_reclassify``. Returns the removed flag names.
    """
    try:
        from memo.tuned_overlay import _scalar_params, read_overlay, write_overlay

        doc = read_overlay(state_dir)
        if not doc:
            return []
        set_by = str((doc.get("_meta") or {}).get("set_by") or "")
        if set_by != "dream-flag-graduation":
            return []
        shadow_flags = {name for name, g in gates.items() if getattr(g, "kind", "") == "shadow"}
        params = _scalar_params(doc)
        removed = [name for name in params if name in shadow_flags]
        if not removed:
            return []
        for name in removed:
            params.pop(name, None)
        write_overlay(state_dir, params, {"set_by": "dream-shadow-reclassify"})
        for name in removed:
            _append_ledger(
                state_dir,
                {"kind": "event", "event": "auto_reverted_on_reclassify", "flag": name},
            )
        return removed
    except Exception as exc:
        _log.debug("dream_shadow: overlay reclassify migration failed: %s", exc)
        return []


def promote(
    cfg: Any, flag: str, *, force_latency: bool = False, apply: bool = False
) -> dict[str, Any]:
    """Human graduation of a review-ready shadow flag (never automatic).

    Refuses unless the flag is review-ready. For a latency-metric shadow flag,
    refuses when the recorded cost p50 exceeds ``MEMO_FLAG_GRADUATION_LATENCY_CEILING_MS``
    unless ``force_latency`` — the offline p50 is a LOWER bound on the real
    recall-hook cost, so a passing offline ceiling is not proof of hook safety.
    With ``apply=False`` (default) nothing is written: returns the exact ``memo
    config set`` command (or an ``export`` fallback when the flag has no markdown
    key). With ``apply=True`` persists via ``config_md.set_value`` and marks the
    decision. Best-effort: returns a result dict with ``ok``/``error``.
    """
    result: dict[str, Any] = {"flag": flag, "ok": False, "applied": False}
    try:
        summary = shadow_summary(cfg.state_dir, flag)
        if not summary["review_ready"]:
            result["error"] = (
                f"not review-ready: {summary['streak']}/{summary['review_nights']} "
                "consecutive-clean nights"
            )
            return result

        metric = _shadow_metric(flag)
        if _is_latency_metric(metric):
            ceiling = _latency_ceiling()
            cost_p50 = float(summary["cost_p50"])
            if not force_latency and not latency_ceiling_gate(cost_p50, ceiling):
                result["error"] = (
                    f"latency p50 {cost_p50}ms exceeds ceiling {ceiling}ms; the offline "
                    "eval is a lower bound on the recall-hook cost — confirm "
                    "hook-faithfully or pass force_latency"
                )
                return result

        path = _env_to_path().get(flag)
        if path is None:
            # Pure meta flag with no markdown-config surface: instruct via env.
            result["instruction"] = f"export {flag}=1"
            if apply:
                result["error"] = "no markdown-config key; set via env/overlay, cannot config-set"
                return result
            result["ok"] = True
            return result

        result["path"] = path
        if not apply:
            result["command"] = f"memo config set {path} true"
            result["ok"] = True
            return result

        from memo.config_md import set_value

        set_value(path, "true")
        _mark_decision(cfg.state_dir, flag, "promoted", reason=f"config set {path}=true")
        result["ok"] = True
        result["applied"] = True
        return result
    except Exception as exc:
        _log.debug("dream_shadow: promote failed for %s: %s", flag, exc)
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def reject(cfg: Any, flag: str, reason: str) -> bool:
    """Record a human rejection of a shadowed flag and mark its decision so it
    drops out of the review-ready pool. Best-effort; returns success."""
    try:
        _append_ledger(
            cfg.state_dir,
            {"kind": "event", "event": "rejected", "flag": flag, "reason": reason[:200]},
        )
        _mark_decision(cfg.state_dir, flag, "rejected", reason=reason[:200])
        return True
    except Exception as exc:
        _log.debug("dream_shadow: reject failed for %s: %s", flag, exc)
        return False


__all__ = [
    "latency_ceiling_gate",
    "maybe_shadow",
    "migrate_reclassified_overlay",
    "promote",
    "read_observations",
    "record_recall_shadow",
    "reject",
    "review_rows",
    "shadow_summary",
]
