"""`memo dream tune` — self-improving recall tuner.

Mines ground-truth labels from ``grounding.log`` (a memory actually USED in an
answer is a positive label, by construction — no hand-labeling), measures
retrieval over the live index, line-searches ``MEMO_RECALL_MIN_SIM``, and
applies the winner via the tuned-params overlay. Every apply must not regress
the curated regression set; a later night whose live config regresses vs the
saved baseline rolls back. OFF by default (``MEMO_DREAM_TUNE_ENABLED``).

Three tuner passes, each gated + reversible, sharing the mined∪curated label
set and the (precision@K, -noise@K) objective:
  - ``run_tuning_pass``          — line-searches ``MEMO_RECALL_MIN_SIM``.
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
    merge_label_prompts,
)
from memo.tuned_overlay import params_version, read_overlay, rollback_overlay, write_overlay

_MIN_SIM = "MEMO_RECALL_MIN_SIM"
_BASELINE = "dream_baseline.json"
_FLOOR_LO, _FLOOR_HI, _FLOOR_STEP = 0.40, 0.85, 0.05

# Graph-proximity weight tuning (Phase 2). Separate knob + baseline file so it
# tunes independently of min_sim and never clobbers it in the overlay.
_GRAPH_WEIGHT = "MEMO_RECALL_GRAPH_PROXIMITY_WEIGHT"
_GRAPH_BASELINE = "dream_graph_baseline.json"
GRAPH_WEIGHT_GRID: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2)


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


def _curated_prompts(state_dir: Path) -> list[dict[str, Any]]:
    """Curated regression prompts — state_dir first (where the daemon reaches),
    repo-committed file second (dev). [] when neither is present."""
    candidates = [
        Path(state_dir) / "eval" / "regression_labels.json",
        Path(__file__).resolve().parent.parent.parent / "eval" / "regression_labels.json",
    ]
    for cp in candidates:
        try:
            raw = json.loads(cp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        prompts = raw.get("prompts")
        if isinstance(prompts, list) and prompts:
            return list(prompts)
    return []


def build_labels(
    cfg: Any, *, min_used_score: float = 0.5, limit: int = 200
) -> tuple[LabelSet, bool]:
    """Mined (grounding) ∪ curated labels. Returns (label_set, curated_used)."""
    mined = harvest_labels(cfg.state_dir, strong=min_used_score, max_labels=limit)
    curated = _curated_prompts(cfg.state_dir)
    merged = merge_label_prompts(curated, mined)
    prompts = [
        Prompt(
            text=str(p["text"]),
            relevant=bool(p.get("relevant", False)),
            expect_ids=[str(x) for x in (p.get("expect_ids") or [])],
        )
        for p in merged
        if p.get("text")
    ]
    return LabelSet(prompts=prompts), bool(curated)


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
    """One nightly tuning pass. Returns a receipt fragment; never raises."""
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
                params = _overlay_params(cfg.state_dir)
                params[resolution["knob"]] = resolution["floor_before"]
                write_overlay(cfg.state_dir, params, {"set_by": "dream-online-revert"})
                _saver = _KNOB_BASELINE_SAVERS.get(resolution["knob"])
                if _saver is not None:
                    _saver(cfg.state_dir, resolution["offline_before"])
                dream_tune_online.set_revert_cooldown(cfg.state_dir)
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

        improved = (after["precision_at_k"], -after["noise_at_k"]) > (
            before["precision_at_k"],
            -before["noise_at_k"],
        )
        if not improved or best_floor == current:
            res["status"] = "noop"
            return res
        if dry_run:
            res["status"] = "would_apply"
            return res

        # Merge, don't clobber: preserve a graph-weight a prior pass set (symmetric
        # with run_graph_weight_pass) so the two tuners coexist in the overlay.
        version_before = params_version(cfg.state_dir)
        params = _overlay_params(cfg.state_dir)
        params[_MIN_SIM] = best_floor
        write_overlay(
            cfg.state_dir,
            params,
            {
                "set_by": "dream",
                "baseline_prec": after["precision_at_k"],
                "baseline_noise": after["noise_at_k"],
            },
        )
        save_baseline(cfg.state_dir, after)
        dream_tune_online.record_pending(
            cfg.state_dir,
            knob=_MIN_SIM,
            value_before=current,
            value_after=best_floor,
            offline_before=before,
            offline_after=after,
            version_before=version_before,
        )
        res["status"] = "applied"
    except Exception as exc:  # surfaced into the receipt, never silent
        res["status"] = "error"
        res["error"] = f"{type(exc).__name__}: {exc}"
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
        if dream_tune_online.has_unresolved_pending(cfg.state_dir) or dream_tune_online.in_revert_cooldown(cfg.state_dir):
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
    {"name": "vec", "mode": "vec", "flags": {"MEMO_GRAPH_RETRIEVAL_ENABLED": "0",
     "MEMO_GRAPH_EXPANSION_ENABLED": "0"}, "overlay": {}},
    {"name": "vec+expansion", "mode": "vec", "flags": {"MEMO_GRAPH_RETRIEVAL_ENABLED": "0",
     "MEMO_GRAPH_EXPANSION_ENABLED": "1"},
     "overlay": {"MEMO_GRAPH_EXPANSION_ENABLED": True}},
    {"name": "hybrid+retrieval", "mode": "hybrid", "flags": {"MEMO_GRAPH_RETRIEVAL_ENABLED": "1",
     "MEMO_GRAPH_EXPANSION_ENABLED": "0"},
     "overlay": {_RECALL_MODE: "hybrid", "MEMO_GRAPH_RETRIEVAL_ENABLED": True}},
    {"name": "hybrid+retrieval+expansion", "mode": "hybrid",
     "flags": {"MEMO_GRAPH_RETRIEVAL_ENABLED": "1", "MEMO_GRAPH_EXPANSION_ENABLED": "1"},
     "overlay": {_RECALL_MODE: "hybrid", "MEMO_GRAPH_RETRIEVAL_ENABLED": True,
                 "MEMO_GRAPH_EXPANSION_ENABLED": True}},
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
        if dream_tune_online.has_unresolved_pending(cfg.state_dir) or dream_tune_online.in_revert_cooldown(cfg.state_dir):
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
            c["name"] for c, m in measured
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


__all__ = [
    "GRAPH_WEIGHT_GRID",
    "RETRIEVAL_CONFIGS",
    "build_labels",
    "load_baseline",
    "load_graph_baseline",
    "load_retrieval_baseline",
    "measure",
    "measure_graph_weight",
    "measure_retrieval_config",
    "read_overlay",
    "run_graph_retrieval_pass",
    "run_graph_weight_pass",
    "run_tuning_pass",
    "save_baseline",
    "save_graph_baseline",
    "save_retrieval_baseline",
    "search_graph_weight",
    "search_min_sim",
]
