"""`memo dream tune` — self-improving recall tuner.

Mines ground-truth labels from ``grounding.log`` (a memory actually USED in an
answer is a positive label, by construction — no hand-labeling), measures
retrieval over the live index, line-searches ``MEMO_RECALL_MIN_SIM``, and
applies the winner via the tuned-params overlay. Every apply must not regress
the curated regression set; a later night whose live config regresses vs the
saved baseline rolls back. OFF by default (``MEMO_DREAM_TUNE_ENABLED``).

Three tuner passes, each gated + reversible, sharing the mined∪curated label
set and the (precision@K, -noise@K) objective:
  - ``run_tuning_pass``          — line-searches ``MEMO_RECALL_MIN_SIM`` plus
    the Fase-3 rank knobs (``MEMO_RECALL_MMR_LAMBDA``,
    ``MEMO_RECALL_SYNTHESIS_BOOST``) through the recall-faithful
    ``eval_recall.Cfg.knob_overrides`` seam, applying at most ONE knob change
    per night (curated no-regression + latency gated).
  - ``run_graph_weight_pass``    — grid-searches the graph-proximity boost
    weight (``MEMO_RECALL_GRAPH_PROXIMITY_WEIGHT``) via the recall-faithful
    ``rank_hits()`` seam.
  - ``run_graph_retrieval_pass`` — selects among candidate recall CONFIGS
    (whether to inject entity-graph candidates as a retrieval source and/or
    expand 1-hop, incl. the ``MEMO_RECALL_MODE`` flip retrieval needs),
    applying the winner only when it beats the plain-vec baseline within the
    recall-hook latency budget.

Each writes only its own key(s) into the shared overlay and preserves the
others, so the passes coexist. A later night whose live config regresses vs the
saved baseline rolls back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memo import dream_tune_online
from memo.eval_recall import (
    Cfg,
    LabelSet,
    Prompt,
    evaluate,
    gate_metrics,
    harvest_labels,
    harvest_negative_labels,
    limit_label_set,
    merge_label_prompts,
)
from memo.tuned_overlay import (
    params_version,
    pin_prev_to_current,
    read_overlay,
    rollback_overlay,
    write_overlay,
)

_MIN_SIM = "MEMO_RECALL_MIN_SIM"
_BASELINE = "dream_baseline.json"
_FLOOR_LO, _FLOOR_HI, _FLOOR_STEP = 0.40, 0.85, 0.05

# Graph-proximity weight tuning (Phase 2). Separate knob + baseline file so it
# tunes independently of min_sim and never clobbers it in the overlay.
_GRAPH_WEIGHT = "MEMO_RECALL_GRAPH_PROXIMITY_WEIGHT"
_GRAPH_BASELINE = "dream_graph_baseline.json"
GRAPH_WEIGHT_GRID: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2)

# Online-only project-boost tuning (F3 boosts). Distinct from the offline knob
# tuners above: the project-affinity boost fires only for hits in the cwd
# project, which the offline label eval does NOT model (labels carry no project
# context). So boosts are NOT offline-measurable — the online proof loop is the
# sole judge. A small hill-climb nudge is applied; if grounding improves it is
# confirmed; if it regresses the online loop reverts it. Gated by
# MEMO_DREAM_TUNE_BOOST_ENABLED (separate opt-in, default OFF).
_PROJECT_BOOST = "MEMO_RECALL_PROJECT_BOOST"
_BOOST_LO, _BOOST_HI = 0.0, 0.5

# Fase 3 — rank-knob tuning (MMR diversity + synthesis boost). Offline-
# measurable through the recall-faithful eval seam
# (eval_recall.Cfg.knob_overrides -> knobs_from_flags(overrides=...) ->
# rank_hits), so they join run_tuning_pass's nightly search. Both flags are
# registered FlagSpecs and flag() consults the tuned overlay generically, so
# an overlay-applied value reaches knobs_from_flags at hook time. Per-knob
# baseline files so an online revert restores only the reverted knob's own
# offline baseline.
_MMR_LAMBDA = "MEMO_RECALL_MMR_LAMBDA"
_SYNTHESIS_BOOST = "MEMO_RECALL_SYNTHESIS_BOOST"
RANK_KNOB_GRIDS: dict[str, tuple[float, ...]] = {
    _MMR_LAMBDA: (0.0, 0.3, 0.5, 0.7),
    _SYNTHESIS_BOOST: (0.0, 0.05, 0.10),
}
# RankKnobs field each flag pins through Cfg.knob_overrides.
_KNOB_TO_FIELD = {_MMR_LAMBDA: "mmr_lambda", _SYNTHESIS_BOOST: "synthesis_boost"}
_RANK_KNOB_BASELINE_FILES = {
    _MMR_LAMBDA: "dream_mmr_baseline.json",
    _SYNTHESIS_BOOST: "dream_synth_baseline.json",
}
# Latency gate: a candidate whose eval p50 exceeds the current config's p50 by
# more than this factor is rejected (MMR is O(K^2) — cheap, but gate anyway).
RANK_KNOB_LATENCY_HEADROOM = 1.25


# --- measurement -------------------------------------------------------------


def measure(mem: Any, labels: LabelSet, *, k: int, floor: float) -> dict[str, float]:
    """precision@K / noise@K for a single vec config at ``floor``."""
    cfg = Cfg(name=f"vec/{floor}", mode="vec", floor=floor, exclude_archived=True)
    rows = evaluate(mem, k=k, labels=labels, configs=[cfg])
    return gate_metrics(rows)


def _baseline_path(state_dir: Path) -> Path:
    return Path(state_dir) / "eval" / _BASELINE


def load_baseline(state_dir: Path) -> dict[str, float] | None:
    try:
        return json.loads(_baseline_path(state_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_baseline(state_dir: Path, metrics: dict[str, float]) -> None:
    p = _baseline_path(state_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


# --- labels ------------------------------------------------------------------


def _curated_raw(state_dir: Path) -> dict[str, Any]:
    """Parsed curated regression-labels document — state_dir first (where the
    daemon reaches), repo-committed file second (dev). {} when neither has
    prompts."""
    candidates = [
        Path(state_dir) / "eval" / "regression_labels.json",
        Path(__file__).resolve().parent.parent.parent / "eval" / "regression_labels.json",
    ]
    for cp in candidates:
        try:
            raw = json.loads(cp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        prompts = raw.get("prompts")
        if isinstance(prompts, list) and prompts:
            return raw
    return {}


def _curated_prompts(state_dir: Path) -> list[dict[str, Any]]:
    """Curated regression prompts — [] when no curated document is present."""
    return list(_curated_raw(state_dir).get("prompts") or [])


def build_labels(
    cfg: Any, *, min_used_score: float = 0.5, limit: int | None = None
) -> tuple[LabelSet, bool]:
    """Mined (grounding) ∪ curated labels. Returns (label_set, curated_used)."""
    if limit is None:
        from memo.flags import flag_int

        limit = flag_int("MEMO_DREAM_MINE_LIMIT") or 200
    mined = harvest_labels(cfg.state_dir, strong=min_used_score, max_labels=limit)
    curated = _curated_prompts(cfg.state_dir)
    merged = merge_label_prompts(curated, mined)
    negatives = harvest_negative_labels(cfg.state_dir, max_labels=100)
    if negatives:
        merged = merge_label_prompts(merged, negatives)
    prompts = [
        Prompt(
            text=str(p["text"]),
            relevant=bool(p.get("relevant", False)),
            expect_ids=[str(x) for x in (p.get("expect_ids") or [])],
            avoid_ids=[str(x) for x in (p.get("avoid_ids") or [])],
        )
        for p in merged
        if p.get("text")
    ]
    # Same noise pass-through as _curated_label_set — without it the tuner's
    # noise@K objective is a vacuous 0.0 (see the curated-gate fix).
    raw = _curated_raw(cfg.state_dir) if curated else {}
    return LabelSet(
        prompts=prompts,
        noise_tags={str(t).lower() for t in (raw.get("noise_tags") or [])},
        noise_path_fragments=tuple(str(f) for f in (raw.get("noise_path_fragments") or [])),
    ), bool(curated)


# --- tuning ------------------------------------------------------------------


def search_min_sim(
    mem: Any,
    labels: LabelSet,
    *,
    k: int,
    current: float,
    lo: float,
    hi: float,
    step: float,
    max_evals: int,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Line-search the floor that maximises (precision, -noise). Returns
    ``(best_floor, metrics_before, metrics_best)``."""
    before = measure(mem, labels, k=k, floor=current)
    best_floor, best = current, before
    evals = 0
    f = lo
    while f <= hi + 1e-9 and evals < max_evals:
        cand = round(f, 4)
        m = measure(mem, labels, k=k, floor=cand)
        evals += 1
        if (m["precision_at_k"], -m["noise_at_k"]) > (best["precision_at_k"], -best["noise_at_k"]):
            best_floor, best = cand, m
        f += step
    return best_floor, before, best


def _regressed(live: dict[str, float], baseline: dict[str, float]) -> bool:
    return (
        live["precision_at_k"] < baseline["precision_at_k"] - 1e-9
        or live["noise_at_k"] > baseline["noise_at_k"] + 1e-9
    )


def run_tuning_pass(
    cfg: Any,
    mem: Any,
    *,
    k: int = 5,
    max_evals: int = 20,
    min_used_score: float = 0.5,
    dry_run: bool = False,
) -> dict[str, Any]:
    """One nightly tuning pass over min_sim + the Fase-3 rank knobs
    (mmr_lambda / synthesis_boost). Searches every knob against the same
    mined∪curated corpus, applies at most ONE knob change (the best), and
    records it for the online proof loop. Per-knob verdicts land in
    ``res["knobs"]``. Returns a receipt fragment; never raises."""
    from memo.flags import flag_float

    res: dict[str, Any] = {"status": "noop"}
    try:
        labels, curated_used = build_labels(cfg, min_used_score=min_used_score)
        res["n_labels"] = len(labels.prompts)
        res["curated_used"] = curated_used
        if not labels.prompts:
            return res

        # Phase-1 online proof loop: resolve a prior night's applied change
        # against its out-of-sample grounding cohort BEFORE searching again.
        # Skipped under dry_run — a preview must not mutate the sidecars.
        from memo import dream_tune_online

        if not dry_run:
            # New tuning cycle: clear last cycle's revert-cooldown marker.
            dream_tune_online.clear_revert_cooldown(cfg.state_dir)

            from memo.flags import flag_int

            _mc = flag_int("MEMO_DREAM_TUNE_MIN_COHORT")
            min_cohort = 20 if _mc is None else _mc
            eps = flag_float("MEMO_DREAM_TUNE_ONLINE_EPS")
            eps = 0.02 if eps is None else eps
            resolution = dream_tune_online.resolve_pending(
                cfg.state_dir,
                min_cohort=min_cohort,
                eps=eps,
                live_version=params_version(cfg.state_dir),
            )
            res["online"] = resolution
            if resolution["status"] == "reverted":
                # Self-contained, knob-generic revert: restore the reverted knob to
                # its pre-apply value by merging into the CURRENT overlay (not the
                # shared one-step _meta.prev), and restore that knob's own offline
                # baseline file.
                params = _scalar_overlay(cfg.state_dir)
                params[resolution["knob"]] = resolution["floor_before"]
                write_overlay(cfg.state_dir, params, {"set_by": "dream-online-revert"})
                _saver = _KNOB_BASELINE_SAVERS.get(resolution["knob"])
                if _saver is not None:
                    _saver(cfg.state_dir, resolution["offline_before"])
                dream_tune_online.set_revert_cooldown(cfg.state_dir)
                # Self-heal _meta.prev so a later offline rollback-guard can't
                # resurrect the config the online loop just reverted away.
                pin_prev_to_current(cfg.state_dir)
                res["status"] = "online_reverted"
                return res
            if resolution["status"] == "waiting":
                res["status"] = "awaiting_online"  # one change per proof cycle
                return res
            # "none"/"expired" → fall through and search a new change.

        current = flag_float(_MIN_SIM)
        current = 0.5 if current is None else current

        # rollback guard: if the LIVE config already regressed vs baseline, revert first.
        baseline = load_baseline(cfg.state_dir)
        if baseline is not None:
            live = measure(mem, labels, k=k, floor=current)
            if _regressed(live, baseline) and not dry_run:
                rolled = rollback_overlay(cfg.state_dir)
                if rolled is not None:
                    res["status"] = "rolled_back"
                    res["restored"] = rolled
                    return res

        # Fase-3 rollback guard: if a previously-applied rank knob (mmr /
        # synthesis) regressed vs its own saved baseline, revert first —
        # mirrors the min_sim guard above; measured only once that knob has a
        # baseline (i.e. was applied at least once).
        for guard_knob in RANK_KNOB_GRIDS:
            knob_baseline = load_rank_knob_baseline(cfg.state_dir, guard_knob)
            if knob_baseline is None:
                continue
            knob_live = flag_float(guard_knob) or 0.0
            live_m = measure_rank_knob(
                mem, labels, k=k, floor=current, knob=guard_knob, value=knob_live
            )
            if _regressed(live_m, knob_baseline) and not dry_run:
                rolled = rollback_overlay(cfg.state_dir)
                if rolled is not None:
                    res["status"] = "rolled_back"
                    res["restored"] = rolled
                    return res

        best_floor, before, after = search_min_sim(
            mem,
            labels,
            k=k,
            current=current,
            lo=_FLOOR_LO,
            hi=_FLOOR_HI,
            step=_FLOOR_STEP,
            max_evals=max_evals,
        )
        res.update(
            {"before": before, "after": after, "floor_before": current, "floor_after": best_floor}
        )

        # Per-knob search results. Candidate tuple =
        # (knob, value_before, value_after, metrics_before, metrics_after);
        # list order (min_sim first) is the tie-break on equal metrics.
        knob_results: dict[str, dict[str, Any]] = {}
        candidates: list[tuple[str, float, float, dict[str, float], dict[str, float]]] = []

        min_sim_improved = best_floor != current and (
            after["precision_at_k"],
            -after["noise_at_k"],
        ) > (before["precision_at_k"], -before["noise_at_k"])
        knob_results[_MIN_SIM] = {
            "value_before": current,
            "value_best": best_floor,
            "before": before,
            "best": after,
            "verdict": "improved" if min_sim_improved else "noop",
        }
        if min_sim_improved:
            # Curated no-regression gate — same bar the rank knobs pass below.
            gate = curated_gate_min_sim(
                mem, cfg.state_dir, k=k, floor_before=current, floor_after=best_floor
            )
            knob_results[_MIN_SIM]["curated"] = gate
            if gate["ok"]:
                candidates.append((_MIN_SIM, current, best_floor, before, after))
            else:
                knob_results[_MIN_SIM]["verdict"] = "curated_rejected"

        # Fase 3 — line-search the rank knobs (mmr_lambda / synthesis_boost)
        # via Cfg.knob_overrides against the SAME label corpus. A per-knob
        # failure lands in the receipt and never aborts the min_sim tune.
        for knob, grid in RANK_KNOB_GRIDS.items():
            try:
                knob_current = flag_float(knob) or 0.0
                best_value, k_before, k_best, lat_rejected = search_rank_knob(
                    mem,
                    labels,
                    k=k,
                    floor=current,
                    knob=knob,
                    current=knob_current,
                    grid=grid,
                    max_evals=max_evals,
                )
                entry: dict[str, Any] = {
                    "value_before": knob_current,
                    "value_best": best_value,
                    "before": k_before,
                    "best": k_best,
                    "latency_rejected": lat_rejected,
                }
                k_improved = best_value != knob_current and (
                    k_best["precision_at_k"],
                    -k_best["noise_at_k"],
                ) > (k_before["precision_at_k"], -k_before["noise_at_k"])
                if k_improved:
                    # Curated no-regression gate: a change that helps mined
                    # labels but hurts the curated regression set is rejected.
                    gate = curated_gate(
                        mem,
                        cfg.state_dir,
                        k=k,
                        floor=current,
                        knob=knob,
                        value_before=knob_current,
                        value_after=best_value,
                    )
                    entry["curated"] = gate
                    if gate["ok"]:
                        entry["verdict"] = "improved"
                        candidates.append((knob, knob_current, best_value, k_before, k_best))
                    else:
                        entry["verdict"] = "curated_rejected"
                else:
                    entry["verdict"] = "noop"
                knob_results[knob] = entry
            except Exception as exc:  # per-knob, surfaced — never kills the pass
                knob_results[knob] = {
                    "verdict": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        res["knobs"] = knob_results

        if not candidates:
            res["status"] = "noop"
            return res

        # Single-apply guard: at most ONE knob change per night. Winner = best
        # (precision, -noise); max() keeps the first on ties (min_sim first).
        knob, value_before, value_after, w_before, w_after = max(
            candidates,
            key=lambda c: (c[4]["precision_at_k"], -c[4]["noise_at_k"]),
        )
        for other, entry in knob_results.items():
            if other != knob and entry.get("verdict") == "improved":
                entry["verdict"] = "deferred_single_apply"
        if dry_run:
            res["status"] = "would_apply"
            res["would_apply"] = {"knob": knob, "value": value_after}
            knob_results[knob]["verdict"] = "would_apply"
            return res

        # Merge, don't clobber: preserve every param a prior pass set (float
        # knobs AND the retrieval pass's bool/str levers) so the tuners coexist.
        version_before = params_version(cfg.state_dir)
        params = _scalar_overlay(cfg.state_dir)
        params[knob] = value_after
        write_overlay(
            cfg.state_dir,
            params,
            {
                "set_by": "dream",
                "knob": knob,
                "baseline_prec": w_after["precision_at_k"],
                "baseline_noise": w_after["noise_at_k"],
            },
        )
        saver = _KNOB_BASELINE_SAVERS.get(knob)
        if saver is not None:
            saver(cfg.state_dir, w_after)
        dream_tune_online.record_pending(
            cfg.state_dir,
            knob=knob,
            value_before=value_before,
            value_after=value_after,
            offline_before=w_before,
            offline_after=w_after,
            version_before=version_before,
        )
        res["status"] = "applied"
        res["applied_knob"] = knob
        knob_results[knob]["verdict"] = "applied"
    except Exception as exc:  # surfaced into the receipt, never silent
        res["status"] = "error"
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res


# --- Fase-3 rank-knob tuning (mmr_lambda / synthesis_boost) -------------------


def _rank_knob_baseline_path(state_dir: Path, knob: str) -> Path:
    return Path(state_dir) / "eval" / _RANK_KNOB_BASELINE_FILES[knob]


def load_rank_knob_baseline(state_dir: Path, knob: str) -> dict[str, float] | None:
    try:
        return json.loads(_rank_knob_baseline_path(state_dir, knob).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def save_rank_knob_baseline(state_dir: Path, knob: str, metrics: dict[str, float]) -> None:
    p = _rank_knob_baseline_path(state_dir, knob)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def measure_rank_knob(
    mem: Any, labels: LabelSet, *, k: int, floor: float, knob: str, value: float
) -> dict[str, float]:
    """precision@K / noise@K / latency p50 with ``knob`` pinned to ``value``.

    Goes through the recall-faithful seam: ``Cfg.knob_overrides`` ->
    ``knobs_from_flags(overrides=...)`` -> ``rank_hits`` — every OTHER knob
    inherits the live flag/overlay resolution, so the delta vs the current
    value is attributable to this knob alone."""
    cfg = Cfg(
        name=f"{knob}={value}",
        mode="vec",
        floor=floor,
        exclude_archived=True,
        knob_overrides={_KNOB_TO_FIELD[knob]: value},
    )
    rows = evaluate(mem, k=k, labels=labels, configs=[cfg])
    metrics = gate_metrics(rows)
    metrics["latency_ms_p50"] = round(rows[0].latency_ms_p50, 1) if rows else 0.0
    return metrics


def search_rank_knob(
    mem: Any,
    labels: LabelSet,
    *,
    k: int,
    floor: float,
    knob: str,
    current: float,
    grid: tuple[float, ...],
    max_evals: int,
    latency_headroom: float = RANK_KNOB_LATENCY_HEADROOM,
) -> tuple[float, dict[str, float], dict[str, float], list[float]]:
    """Line-search ``knob`` over ``grid`` maximising (precision, -noise).

    Latency gate: a candidate whose eval p50 exceeds the current config's p50
    by more than ``latency_headroom`` is rejected regardless of precision
    (skipped when the current p50 rounds to 0 — stub/tiny corpora). Returns
    ``(best_value, metrics_before, metrics_best, latency_rejected)``."""
    before = measure_rank_knob(mem, labels, k=k, floor=floor, knob=knob, value=current)
    budget = before.get("latency_ms_p50", 0.0) * latency_headroom
    best_value, best = current, before
    rejected: list[float] = []
    evals = 0
    for v in grid:
        cand = round(float(v), 4)
        if cand == round(current, 4) or evals >= max_evals:
            continue
        m = measure_rank_knob(mem, labels, k=k, floor=floor, knob=knob, value=cand)
        evals += 1
        if budget > 0 and m.get("latency_ms_p50", 0.0) > budget:
            rejected.append(cand)
            continue
        if (m["precision_at_k"], -m["noise_at_k"]) > (best["precision_at_k"], -best["noise_at_k"]):
            best_value, best = cand, m
    return best_value, before, best, rejected


def _curated_label_set(state_dir: Path) -> LabelSet | None:
    """The curated regression prompts as a LabelSet (same resolution as
    ``build_labels``: state_dir first, repo file second). None when absent.

    The document-level ``noise_tags``/``noise_path_fragments`` are passed
    through (same parsing as ``eval_recall.load_labels``) so the gate's
    noise@K dimension measures real noise instead of a vacuous 0.0 —
    ``_regressed`` then rejects a candidate that RAISES curated noise."""
    prompts = [
        Prompt(
            text=str(p["text"]),
            relevant=bool(p.get("relevant", False)),
            expect_ids=[str(x) for x in (p.get("expect_ids") or [])],
        )
        for p in _curated_prompts(state_dir)
        if p.get("text")
    ]
    if not prompts:
        return None
    raw = _curated_raw(state_dir)
    return LabelSet(
        prompts=prompts,
        noise_tags={str(t).lower() for t in (raw.get("noise_tags") or [])},
        noise_path_fragments=tuple(str(f) for f in (raw.get("noise_path_fragments") or [])),
    )


def curated_gate(
    mem: Any,
    state_dir: Path,
    *,
    k: int,
    floor: float,
    knob: str,
    value_before: float,
    value_after: float,
) -> dict[str, Any]:
    """Curated-labels no-regression gate for a winning rank-knob candidate.

    Measures candidate vs current on the CURATED set only: a change that helps
    the mined labels but drops precision (or raises noise) on the curated
    regression set is rejected. Vacuously passes when no curated set exists."""
    curated = _curated_label_set(state_dir)
    if curated is None:
        return {"ok": True, "reason": "no_curated_labels"}
    cur_before = measure_rank_knob(mem, curated, k=k, floor=floor, knob=knob, value=value_before)
    cur_after = measure_rank_knob(mem, curated, k=k, floor=floor, knob=knob, value=value_after)
    return {"ok": not _regressed(cur_after, cur_before), "before": cur_before, "after": cur_after}


def curated_gate_min_sim(
    mem: Any,
    state_dir: Path,
    *,
    k: int,
    floor_before: float,
    floor_after: float,
) -> dict[str, Any]:
    """Curated-labels no-regression gate for the min_sim candidate — same
    contract as :func:`curated_gate`, but the knob under test IS the floor.
    Without this, a floor that wins on mined labels could bury a curated
    must-surface memory and still be applied."""
    curated = _curated_label_set(state_dir)
    if curated is None:
        return {"ok": True, "reason": "no_curated_labels"}
    cur_before = measure(mem, curated, k=k, floor=floor_before)
    cur_after = measure(mem, curated, k=k, floor=floor_after)
    return {"ok": not _regressed(cur_after, cur_before), "before": cur_before, "after": cur_after}


# --- HyDE A/B (MEMO_HYDE_ENABLED — shipped default-off, never measured) ------

_HYDE_FLAG = "MEMO_HYDE_ENABLED"
_HYDE_FLOOR = 0.40  # same floor as grid configs C/J (hybrid)
_HYDE_MAX_PROMPTS = 40  # cap: HyDE costs one MLX chat call PER PROMPT
# HyDE adds an LLM call by construction; it is ask-path-only (never the vec
# hook), so the headroom is looser than RANK_KNOB_LATENCY_HEADROOM.
HYDE_LATENCY_HEADROOM = 3.0


def measure_hyde(mem: Any, labels: LabelSet, *, k: int, enabled: bool) -> dict[str, float]:
    """prec@K / noise@K / p50 for hybrid retrieval with HyDE pinned on/off,
    through the Cfg.flag_overrides env-pin seam (HyDE is read by flag_bool
    inside Memory.search — RankKnobs can't reach it)."""
    cfg = Cfg(
        name=f"hyde={'on' if enabled else 'off'}",
        mode="hybrid",
        floor=_HYDE_FLOOR,
        exclude_archived=True,
        flag_overrides={_HYDE_FLAG: "1" if enabled else "0"},
    )
    rows = evaluate(mem, k=k, labels=labels, configs=[cfg])
    metrics = gate_metrics(rows)
    metrics["latency_ms_p50"] = round(rows[0].latency_ms_p50, 1) if rows else 0.0
    return metrics


def run_hyde_pass(
    cfg: Any,
    mem: Any,
    *,
    k: int = 5,
    min_used_score: float = 0.5,
    dry_run: bool = False,
) -> dict[str, Any]:
    """One nightly HyDE on/off A/B. Applies MEMO_HYDE_ENABLED=1 to the tuned
    overlay only when ALL gates pass. Returns a receipt fragment; never raises
    past its caller's try (cli_dream catches into receipt["errors"])."""
    from memo.flags import flag_bool, flag_str

    res: dict[str, Any] = {"status": "noop", "knob": _HYDE_FLAG}
    if flag_bool(_HYDE_FLAG):
        res["status"] = "already_on"
        return res
    # One overlay change per proof cycle: an ungated write here would bump
    # params_version and expire the proof loop's pending verification.
    if dream_tune_online.has_unresolved_pending(
        cfg.state_dir
    ) or dream_tune_online.in_revert_cooldown(cfg.state_dir):
        res["status"] = "deferred_pending"
        return res
    live_mode = (flag_str("MEMO_RECALL_MODE") or "vec").strip().lower()
    if live_mode == "hybrid":
        # Overlay-applied HyDE would inject an MLX chat call into the recall
        # hook's hybrid search — never risk the 5s budget. Hard veto.
        res["status"] = "skipped_hook_mode_hybrid"
        return res
    labels, curated_used = build_labels(cfg, min_used_score=min_used_score)
    labels = limit_label_set(labels, _HYDE_MAX_PROMPTS)
    if not labels.prompts:
        res["status"] = "no_labels"
        return res
    off = measure_hyde(mem, labels, k=k, enabled=False)
    on = measure_hyde(mem, labels, k=k, enabled=True)
    res.update({"off": off, "on": on, "curated_used": curated_used})
    wins = (on["precision_at_k"], -on["noise_at_k"]) > (off["precision_at_k"], -off["noise_at_k"])
    if not wins:
        res["status"] = "hyde_loses"
        return res
    budget = off.get("latency_ms_p50", 0.0) * HYDE_LATENCY_HEADROOM
    if budget > 0 and on.get("latency_ms_p50", 0.0) > budget:
        res["status"] = "rejected_latency"
        return res
    curated = _curated_label_set(cfg.state_dir)
    if curated is not None:
        c_off = measure_hyde(mem, curated, k=k, enabled=False)
        c_on = measure_hyde(mem, curated, k=k, enabled=True)
        res["curated"] = {"off": c_off, "on": c_on}
        if _regressed(c_on, c_off):
            res["status"] = "rejected_curated"
            return res
    if dry_run:
        res["status"] = "would_apply"
        return res
    params = {**_scalar_overlay(cfg.state_dir), _HYDE_FLAG: True}
    write_overlay(cfg.state_dir, params, {"set_by": "dream-hyde"})
    res["status"] = "applied"
    return res


# --- graph-proximity weight tuning -------------------------------------------


def measure_graph_weight(
    mem: Any, labels: LabelSet, *, k: int, weight: float, floor: float
) -> dict[str, float]:
    """precision@K / noise@K for the live index with the graph-proximity boost
    applied at ``weight`` (gated at ``floor`` = current ``min_sim``).

    Mirrors ``eval_recall.run_config``'s vec ranking but threads the Phase-2
    ``graph_boost`` seam through ``rank_hits`` so the measurement reflects the
    real recall path. ``eval_recall`` itself is not graph-boost aware, so this
    measure lives here rather than extending the shared harness.
    """
    from memo.eval_recall import _is_noise, _is_relevant
    from memo.graph_proximity import extract_query_entities, graph_boost_factory
    from memo.recall_logic import RankKnobs, rank_hits

    graph = getattr(mem, "graph", None)
    knobs = RankKnobs(top_k=k, min_sim=floor, min_body_chars=0, mode="vec")
    prec_hits = prec_total = noise_hits = 0
    n_prompts = len(labels.prompts) or 1
    for prompt in labels.prompts:
        hits = mem.search(prompt.text, limit=k * 4, mode="vec")
        graph_boost = None
        if weight > 0 and graph is not None:
            graph_boost = graph_boost_factory(
                graph, extract_query_entities(prompt.text, graph), weight=weight
            )
        ranked = rank_hits(hits, knobs, graph_boost=graph_boost)
        ranked = [h for h in ranked if not _is_noise(h, labels)]
        top = ranked[:k]
        noise_hits += sum(1 for h in top if _is_noise(h, labels))
        if prompt.relevant or prompt.expect_ids:
            prec_total += k
            prec_hits += sum(1 for h in top if _is_relevant(h, prompt, labels))
    return {
        "precision_at_k": round(prec_hits / prec_total, 3) if prec_total else 0.0,
        "noise_at_k": round(noise_hits / (n_prompts * k), 3) if (n_prompts * k) else 0.0,
    }


def search_graph_weight(
    mem: Any,
    labels: LabelSet,
    *,
    k: int,
    current: float,
    floor: float,
    grid: tuple[float, ...],
    max_evals: int,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Grid-search the graph-proximity weight maximising (precision, -noise).
    Returns ``(best_weight, metrics_before, metrics_best)``. Mirrors
    ``search_min_sim``."""
    before = measure_graph_weight(mem, labels, k=k, weight=current, floor=floor)
    best_weight, best = current, before
    for evals, w in enumerate(grid):
        if evals >= max_evals:
            break
        cand = round(w, 4)
        m = measure_graph_weight(mem, labels, k=k, weight=cand, floor=floor)
        if (m["precision_at_k"], -m["noise_at_k"]) > (best["precision_at_k"], -best["noise_at_k"]):
            best_weight, best = cand, m
    return best_weight, before, best


def _graph_baseline_path(state_dir: Path) -> Path:
    return Path(state_dir) / "eval" / _GRAPH_BASELINE


def load_graph_baseline(state_dir: Path) -> dict[str, float] | None:
    try:
        return json.loads(_graph_baseline_path(state_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_graph_baseline(state_dir: Path, metrics: dict[str, float]) -> None:
    p = _graph_baseline_path(state_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


# Which offline-baseline saver each tunable knob calls on an online revert.
# Lambda wrappers preserve late-binding: save_baseline / save_graph_baseline are
# looked up in the module's global namespace at call time, not at dict-construction
# time, so monkeypatching either function works correctly in tests.
_KNOB_BASELINE_SAVERS = {
    _MIN_SIM: lambda sd, m: save_baseline(sd, m),
    _GRAPH_WEIGHT: lambda sd, m: save_graph_baseline(sd, m),
    _MMR_LAMBDA: lambda sd, m: save_rank_knob_baseline(sd, _MMR_LAMBDA, m),
    _SYNTHESIS_BOOST: lambda sd, m: save_rank_knob_baseline(sd, _SYNTHESIS_BOOST, m),
}


def _overlay_params(state_dir: Path) -> dict[str, float]:
    """Current numeric overlay params (no ``_meta``) — so a graph-weight write
    preserves a min_sim value a prior pass set, instead of clobbering it."""
    return {
        key: float(val)
        for key, val in read_overlay(state_dir).items()
        if key != "_meta" and isinstance(val, (int, float)) and not isinstance(val, bool)
    }


def run_graph_weight_pass(
    cfg: Any,
    mem: Any,
    *,
    k: int = 5,
    max_evals: int = 20,
    min_used_score: float = 0.5,
    dry_run: bool = False,
) -> dict[str, Any]:
    """One nightly graph-proximity-weight tuning pass. Mirrors
    ``run_tuning_pass``: build labels, roll back a regressed live config first,
    grid-search the weight, apply the winner via the overlay (preserving other
    params), save the graph baseline. Never raises."""
    from memo.flags import flag_float

    res: dict[str, Any] = {"status": "noop"}
    try:
        labels, curated_used = build_labels(cfg, min_used_score=min_used_score)
        res["n_labels"] = len(labels.prompts)
        res["curated_used"] = curated_used
        if not labels.prompts:
            return res

        # One overlay change per proof cycle: if the proof loop has an
        # unresolved pending OR a revert just happened this cycle, hold the overlay
        # steady — skip the (expensive) search entirely and defer.
        if dream_tune_online.has_unresolved_pending(
            cfg.state_dir
        ) or dream_tune_online.in_revert_cooldown(cfg.state_dir):
            res["status"] = "deferred_pending"
            return res

        floor = flag_float(_MIN_SIM)
        floor = 0.5 if floor is None else floor
        current = flag_float(_GRAPH_WEIGHT) or 0.0

        # rollback guard: if the LIVE weight already regressed vs baseline, revert.
        baseline = load_graph_baseline(cfg.state_dir)
        if baseline is not None:
            live = measure_graph_weight(mem, labels, k=k, weight=current, floor=floor)
            if _regressed(live, baseline) and not dry_run:
                rolled = rollback_overlay(cfg.state_dir)
                if rolled is not None:
                    res["status"] = "rolled_back"
                    res["restored"] = rolled
                    return res

        best_weight, before, after = search_graph_weight(
            mem,
            labels,
            k=k,
            current=current,
            floor=floor,
            grid=GRAPH_WEIGHT_GRID,
            max_evals=max_evals,
        )
        res.update(
            {
                "before": before,
                "after": after,
                "weight_before": current,
                "weight_after": best_weight,
            }
        )

        improved = (after["precision_at_k"], -after["noise_at_k"]) > (
            before["precision_at_k"],
            -before["noise_at_k"],
        )
        if not improved or best_weight == current:
            res["status"] = "noop"
            return res
        # Curated no-regression gate — same bar as the rank knobs.
        curated = _curated_label_set(cfg.state_dir)
        if curated is not None:
            cur_before = measure_graph_weight(mem, curated, k=k, weight=current, floor=floor)
            cur_after = measure_graph_weight(mem, curated, k=k, weight=best_weight, floor=floor)
            res["curated"] = {"before": cur_before, "after": cur_after}
            if _regressed(cur_after, cur_before):
                res["status"] = "curated_rejected"
                return res
        if dry_run:
            res["status"] = "would_apply"
            return res

        version_before = params_version(cfg.state_dir)
        params = _overlay_params(cfg.state_dir)
        params[_GRAPH_WEIGHT] = best_weight
        write_overlay(
            cfg.state_dir,
            params,
            {
                "set_by": "dream-graph",
                "baseline_prec": after["precision_at_k"],
                "baseline_noise": after["noise_at_k"],
            },
        )
        save_graph_baseline(cfg.state_dir, after)
        # Join the online proof loop: this graph-weight change is now verified
        # out-of-sample next cycle (and reverted by knob if it regresses).
        dream_tune_online.record_pending(
            cfg.state_dir,
            knob=_GRAPH_WEIGHT,
            value_before=current,
            value_after=best_weight,
            offline_before=before,
            offline_after=after,
            version_before=version_before,
        )
        res["status"] = "applied"
    except Exception as exc:  # surfaced into the receipt, never silent
        res["status"] = "error"
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res


# --- graph-injection (retrieval / expansion) config tuning -------------------
#
# Distinct from the two knob tuners above: those move a scalar (min_sim /
# proximity weight); this one selects among a small set of *candidate recall
# configs* — whether to inject entity-graph candidates as a retrieval source
# (hybrid RRF) and/or expand 1-hop after ranking, including the recall mode flip
# that graph retrieval requires. It applies the winning config via the overlay
# only when it beats the plain-vec baseline AND stays within the recall-hook
# latency budget (a hybrid flip that helps precision but blows the 5s budget is
# rejected — see MEMO recall-hook budget in CLAUDE.md). Reversible like the
# others.

_RETRIEVAL_BASELINE = "dream_retrieval_baseline.json"
_RECALL_MODE = "MEMO_RECALL_MODE"
# Graph-injection env levers this pass owns (cleared/reset per measurement so a
# prior night's overlay never leaks into another config's numbers).
_MANAGED_RETRIEVAL_FLAGS = ("MEMO_GRAPH_RETRIEVAL_ENABLED", "MEMO_GRAPH_EXPANSION_ENABLED")
# All overlay keys this pass manages (so applying a winner clears a prior one).
_MANAGED_RETRIEVAL_KEYS = (_RECALL_MODE, *_MANAGED_RETRIEVAL_FLAGS)
# Search-latency ceiling (p50, ms) a config must respect to be eligible. Hybrid
# is materially slower; the recall hook has ~3s for embed+search+format after a
# ~2s cold MLX load, so a config whose search p50 exceeds this would risk the
# 5s budget and is refused regardless of precision.
_RETRIEVAL_LATENCY_BUDGET_MS = 2500.0

# name -> (recall mode, {env flag: "0"/"1"}, overlay-to-apply-if-it-wins).
# The plain-vec baseline carries an empty overlay (pure defaults).
RETRIEVAL_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "vec",
        "mode": "vec",
        "flags": {"MEMO_GRAPH_RETRIEVAL_ENABLED": "0", "MEMO_GRAPH_EXPANSION_ENABLED": "0"},
        "overlay": {},
    },
    {
        "name": "vec+expansion",
        "mode": "vec",
        "flags": {"MEMO_GRAPH_RETRIEVAL_ENABLED": "0", "MEMO_GRAPH_EXPANSION_ENABLED": "1"},
        "overlay": {"MEMO_GRAPH_EXPANSION_ENABLED": True},
    },
    {
        "name": "hybrid+retrieval",
        "mode": "hybrid",
        "flags": {"MEMO_GRAPH_RETRIEVAL_ENABLED": "1", "MEMO_GRAPH_EXPANSION_ENABLED": "0"},
        "overlay": {_RECALL_MODE: "hybrid", "MEMO_GRAPH_RETRIEVAL_ENABLED": True},
    },
    {
        "name": "hybrid+retrieval+expansion",
        "mode": "hybrid",
        "flags": {"MEMO_GRAPH_RETRIEVAL_ENABLED": "1", "MEMO_GRAPH_EXPANSION_ENABLED": "1"},
        "overlay": {
            _RECALL_MODE: "hybrid",
            "MEMO_GRAPH_RETRIEVAL_ENABLED": True,
            "MEMO_GRAPH_EXPANSION_ENABLED": True,
        },
    },
)


def _retrieval_baseline_path(state_dir: Path) -> Path:
    return Path(state_dir) / "eval" / _RETRIEVAL_BASELINE


def load_retrieval_baseline(state_dir: Path) -> dict[str, float] | None:
    try:
        return json.loads(_retrieval_baseline_path(state_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_retrieval_baseline(state_dir: Path, metrics: dict[str, float]) -> None:
    p = _retrieval_baseline_path(state_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def measure_retrieval_config(
    mem: Any, labels: LabelSet, *, k: int, mode: str, flags: dict[str, str]
) -> dict[str, float]:
    """precision@K / noise@K / latency_ms_p50 for one candidate recall config.

    The graph-injection flags are set in ``os.environ`` around a reuse of the
    shared ``evaluate()`` harness (which reads them inside ``mem.search`` and
    applies the real hybrid true-cosine gate), so the numbers reflect the
    production recall path. Both managed flags are always fully specified so a
    prior night's overlay cannot leak into the measurement. The environment is
    restored afterwards.
    """
    import os

    from memo.flags import flag_float

    saved = {kk: os.environ.get(kk) for kk in _MANAGED_RETRIEVAL_FLAGS}
    try:
        for kk, vv in flags.items():
            os.environ[kk] = vv
        floor = flag_float(_MIN_SIM)
        floor = 0.5 if floor is None else floor
        cfg = Cfg(name=mode, mode=mode, floor=floor, exclude_archived=True)
        rows = evaluate(mem, k=k, labels=labels, configs=[cfg])
        metrics = gate_metrics(rows)
        metrics["latency_ms_p50"] = round(rows[0].latency_ms_p50, 1) if rows else 0.0
        return metrics
    finally:
        for kk, prev in saved.items():
            if prev is None:
                os.environ.pop(kk, None)
            else:
                os.environ[kk] = prev


def _scalar_overlay(state_dir: Path) -> dict[str, Any]:
    """Current overlay params (no ``_meta``), native types preserved — so
    applying a retrieval winner keeps a float knob a prior pass set."""
    return {k: v for k, v in read_overlay(state_dir).items() if k != "_meta"}


def _live_retrieval_config(state_dir: Path) -> dict[str, Any]:
    """The RETRIEVAL_CONFIGS entry the overlay currently applies (for the
    rollback guard). Matches on the managed levers; defaults to plain vec."""
    ov = read_overlay(state_dir)
    live_mode = str(ov.get(_RECALL_MODE, "vec"))
    live_ret = bool(ov.get("MEMO_GRAPH_RETRIEVAL_ENABLED", False))
    live_exp = bool(ov.get("MEMO_GRAPH_EXPANSION_ENABLED", False))
    for c in RETRIEVAL_CONFIGS:
        if (
            c["mode"] == live_mode
            and (c["flags"]["MEMO_GRAPH_RETRIEVAL_ENABLED"] == "1") == live_ret
            and (c["flags"]["MEMO_GRAPH_EXPANSION_ENABLED"] == "1") == live_exp
        ):
            return c
    return RETRIEVAL_CONFIGS[0]


def run_graph_retrieval_pass(
    cfg: Any,
    mem: Any,
    *,
    k: int = 5,
    min_used_score: float = 0.5,
    dry_run: bool = False,
    latency_budget_ms: float = _RETRIEVAL_LATENCY_BUDGET_MS,
) -> dict[str, Any]:
    """One nightly graph-injection config pass. Grid the candidate recall
    configs, apply the best via the overlay when it beats the plain-vec
    baseline within the latency budget, revert a regressed live config first.
    Never raises — mirrors ``run_graph_weight_pass``."""
    res: dict[str, Any] = {"status": "noop"}
    try:
        labels, curated_used = build_labels(cfg, min_used_score=min_used_score)
        res["n_labels"] = len(labels.prompts)
        res["curated_used"] = curated_used
        if not labels.prompts:
            return res

        # One overlay change per proof cycle: if the proof loop has an
        # unresolved pending OR a revert just happened this cycle, hold the overlay
        # steady — skip the (expensive) search entirely and defer.
        if dream_tune_online.has_unresolved_pending(
            cfg.state_dir
        ) or dream_tune_online.in_revert_cooldown(cfg.state_dir):
            res["status"] = "deferred_pending"
            return res

        # rollback guard: if the LIVE overlay config already regressed vs the
        # saved baseline, revert before considering new configs.
        baseline = load_retrieval_baseline(cfg.state_dir)
        if baseline is not None:
            live_cfg = _live_retrieval_config(cfg.state_dir)
            live = measure_retrieval_config(
                mem, labels, k=k, mode=live_cfg["mode"], flags=live_cfg["flags"]
            )
            if _regressed(live, baseline) and not dry_run:
                rolled = rollback_overlay(cfg.state_dir)
                if rolled is not None:
                    res["status"] = "rolled_back"
                    res["restored"] = rolled
                    return res

        measured = []
        for c in RETRIEVAL_CONFIGS:
            m = measure_retrieval_config(mem, labels, k=k, mode=c["mode"], flags=c["flags"])
            measured.append((c, m))
        base = measured[0][1]
        res["baseline"] = base
        res["configs"] = [{"name": c["name"], **m} for c, m in measured]

        # Eligible = respects the latency budget. The baseline (plain vec) is
        # always eligible so we never get stuck with nothing to compare against.
        eligible = [
            (c, m)
            for c, m in measured
            if c["name"] == "vec" or m["latency_ms_p50"] <= latency_budget_ms
        ]
        res["latency_rejected"] = [
            c["name"]
            for c, m in measured
            if c["name"] != "vec" and m["latency_ms_p50"] > latency_budget_ms
        ]
        best_cfg, best = max(
            eligible,
            key=lambda cm: (cm[1]["precision_at_k"], -cm[1]["noise_at_k"]),
        )
        res["best"] = {"name": best_cfg["name"], **best}

        improved = (best["precision_at_k"], -best["noise_at_k"]) > (
            base["precision_at_k"],
            -base["noise_at_k"],
        )
        if not improved or best_cfg["name"] == "vec":
            res["status"] = "noop"
            return res
        # Curated no-regression gate — same bar as the rank knobs.
        curated = _curated_label_set(cfg.state_dir)
        if curated is not None:
            vec_cfg = RETRIEVAL_CONFIGS[0]
            cur_before = measure_retrieval_config(
                mem, curated, k=k, mode=vec_cfg["mode"], flags=vec_cfg["flags"]
            )
            cur_after = measure_retrieval_config(
                mem, curated, k=k, mode=best_cfg["mode"], flags=best_cfg["flags"]
            )
            res["curated"] = {"before": cur_before, "after": cur_after}
            if _regressed(cur_after, cur_before):
                res["status"] = "curated_rejected"
                return res
        if dry_run:
            res["status"] = "would_apply"
            res["would_apply"] = best_cfg["name"]
            return res

        # Apply: preserve prior scalar knobs, clear any retrieval levers we
        # manage, then set the winner's overlay.
        params = _scalar_overlay(cfg.state_dir)
        for kk in _MANAGED_RETRIEVAL_KEYS:
            params.pop(kk, None)
        params.update(best_cfg["overlay"])
        write_overlay(
            cfg.state_dir,
            params,
            {
                "set_by": "dream-retrieval",
                "config": best_cfg["name"],
                "baseline_prec": best["precision_at_k"],
                "baseline_noise": best["noise_at_k"],
            },
        )
        save_retrieval_baseline(cfg.state_dir, best)
        res["status"] = "applied"
        res["applied"] = best_cfg["name"]
    except Exception as exc:  # surfaced into the receipt, never silent
        res["status"] = "error"
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res


# --- online-only project-boost tuner ----------------------------------------


def _boost_direction(state_dir: Any, step: float) -> float:
    """Next nudge direction for the project-boost, learned from the ledger:
    repeat the direction that was confirmed, reverse the one that was reverted/
    expired, explore up (+step) when there is no history for this knob."""
    for e in reversed(dream_tune_online.read_ledger(state_dir, limit=50)):
        if e.get("knob") != _PROJECT_BOOST:
            continue
        last_up = float(e.get("floor_after", 0.0)) >= float(e.get("floor_before", 0.0))
        if e.get("verdict") == "confirmed":
            return step if last_up else -step
        if e.get("verdict") in ("reverted", "expired"):
            return -step if last_up else step
        break
    return step


def run_boost_pass(
    cfg: Any, mem: Any, *, step: float = 0.05, dry_run: bool = False
) -> dict[str, Any]:
    """One nightly ONLINE-ONLY project-boost exploration. Proposes a small nudge
    and lets the online proof loop verify it (no offline measure — boosts are not
    offline-measurable). Respects the one-change-per-cycle guard. Never raises."""
    from memo.flags import flag_float

    res: dict[str, Any] = {"status": "noop"}
    try:
        # one change per proof cycle: defer while any pending/cooldown is active
        if dream_tune_online.has_unresolved_pending(
            cfg.state_dir
        ) or dream_tune_online.in_revert_cooldown(cfg.state_dir):
            res["status"] = "deferred_pending"
            return res

        current = flag_float(_PROJECT_BOOST)
        current = 0.25 if current is None else current
        direction = _boost_direction(cfg.state_dir, step)
        proposed = round(current + direction, 4)
        res.update({"boost_before": current, "boost_after": proposed, "direction": direction})
        if proposed < _BOOST_LO or proposed > _BOOST_HI or proposed == current:
            res["status"] = "noop"
            res["reason"] = "boundary"
            return res
        if dry_run:
            res["status"] = "would_apply"
            return res

        version_before = params_version(cfg.state_dir)
        params = _overlay_params(cfg.state_dir)
        params[_PROJECT_BOOST] = proposed
        write_overlay(cfg.state_dir, params, {"set_by": "dream-boost"})
        # No offline metrics for boosts — the online proof loop is the sole judge.
        dream_tune_online.record_pending(
            cfg.state_dir,
            knob=_PROJECT_BOOST,
            value_before=current,
            value_after=proposed,
            offline_before={},
            offline_after={},
            version_before=version_before,
        )
        res["status"] = "applied"
    except Exception as exc:  # surfaced into the receipt, never silent
        res["status"] = "error"
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res


__all__ = [
    "GRAPH_WEIGHT_GRID",
    "RANK_KNOB_GRIDS",
    "RETRIEVAL_CONFIGS",
    "build_labels",
    "curated_gate",
    "load_baseline",
    "load_graph_baseline",
    "load_rank_knob_baseline",
    "load_retrieval_baseline",
    "measure",
    "measure_graph_weight",
    "measure_rank_knob",
    "measure_retrieval_config",
    "read_overlay",
    "run_boost_pass",
    "run_graph_retrieval_pass",
    "run_graph_weight_pass",
    "run_hyde_pass",
    "run_tuning_pass",
    "save_baseline",
    "save_graph_baseline",
    "save_rank_knob_baseline",
    "save_retrieval_baseline",
    "search_graph_weight",
    "search_min_sim",
    "search_rank_knob",
]
