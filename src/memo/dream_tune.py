"""`memo dream tune` — self-improving recall tuner.

Mines ground-truth labels from ``grounding.log`` (a memory actually USED in an
answer is a positive label, by construction — no hand-labeling), measures
retrieval over the live index, line-searches ``MEMO_RECALL_MIN_SIM``, and
applies the winner via the tuned-params overlay. Every apply must not regress
the curated regression set; a later night whose live config regresses vs the
saved baseline rolls back. OFF by default (``MEMO_DREAM_TUNE_ENABLED``).

Scope note: only ``min_sim`` is tuned this phase — it is the one ranking knob
the existing eval harness measures faithfully (its ``Cfg.floor`` is the same
``h.score < min_sim`` gate the recall path applies in vec mode). Boosts /
rerank pool need a recall-faithful eval (a pure ``rank_hits()`` extracted from
``recall_logic``) and are deliberately deferred.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memo.eval_recall import (
    Cfg,
    LabelSet,
    Prompt,
    evaluate,
    gate_metrics,
    harvest_labels,
    merge_label_prompts,
)
from memo.tuned_overlay import read_overlay, rollback_overlay, write_overlay

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
        res["status"] = "applied"
    except Exception as exc:  # surfaced into the receipt, never silent
        res["status"] = "error"
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res


__all__ = [
    "GRAPH_WEIGHT_GRID",
    "build_labels",
    "load_baseline",
    "load_graph_baseline",
    "measure",
    "measure_graph_weight",
    "read_overlay",
    "run_graph_weight_pass",
    "run_tuning_pass",
    "save_baseline",
    "save_graph_baseline",
    "search_graph_weight",
    "search_min_sim",
]
