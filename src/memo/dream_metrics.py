"""Fase-4 learning metrics + the tuner graduation gate.

Two responsibilities, both nightly-only and MLX-free:

1. **Pure gate functions** (:func:`graduation_gate` + the sub-gates) the
   recall self-tuner calls before it applies a knob change. Today
   ``dream_tune.run_tuning_pass`` gates a change on only three of the six
   required conditions (offline improvement, curated no-regression, rank-knob
   latency headroom). :func:`graduation_gate` folds ALL six into one verdict:
   precision improves by ``>= min_margin``, noise not up, latency within both a
   headroom ratio and an absolute hook budget, a minimum label-sample count,
   plus the two structural invariants (experiment recorded, rollback exists).
   The functions are pure — the caller reads the flags and threads the
   thresholds in — so they never touch the store, the flags, or MLX.

2. **Read-only :func:`learning_metrics`** — assembles the eleven learning
   metrics from their existing sources (eval before/after, grounded_rate,
   the online cohort, the verdict log, consolidate-reuse, outcome utilities,
   recall-latency percentiles) into one snapshot the tuner receipt/ledger can
   record for before/after-by-cohort auditability. It NEVER re-runs the eval
   (it reuses the ``before``/``after`` the tuner already measured) and NEVER
   raises — every sub-failure lands in the returned ``errors`` list, mirroring
   the dream passes.

MLX-free: only reads jsonl logs + ``store.list_recent``; never calls
``embed()`` / ``chat()``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

# Float comparison slack — mirrors dream_tune._regressed's 1e-9 tolerance so
# the gate never trips on representation noise.
_EPS = 1e-9

# verdict.classify_reaction vocabulary that counts as a bad outcome.
_NEGATIVE_VERDICTS = frozenset({"negative", "correction"})


# --- pure gate functions -----------------------------------------------------


def min_labels_gate(n_labels: int, min_labels: int) -> bool:
    """Enough labeled samples to trust the offline apply (statistical floor)."""
    return n_labels >= min_labels


def margin_gate(prec_before: float, prec_after: float, min_margin: float) -> bool:
    """precision@k improved by at least ``min_margin`` (0.0 = any non-regression).

    Deadband: ``min_margin > 0`` rejects trivial +epsilon churn on the overlay.
    """
    return prec_after >= prec_before + min_margin - _EPS


def noise_gate(noise_before: float, noise_after: float) -> bool:
    """noise@k did not rise."""
    return noise_after <= noise_before + _EPS


def latency_gate(
    latency_before: float,
    latency_after: float,
    *,
    headroom: float,
    hook_budget_ms: float,
) -> bool:
    """Candidate latency respects BOTH the headroom ratio and the absolute budget.

    - Absolute: ``latency_after <= hook_budget_ms`` (protects the 5s recall-hook
      budget) — always enforced.
    - Ratio: ``latency_after <= headroom * latency_before`` — skipped when the
      baseline p50 rounds to 0.0 (stub/tiny corpora have no meaningful ratio).
    """
    if latency_after > hook_budget_ms + _EPS:
        return False
    if round(latency_before, 1) <= 0.0:
        return True
    return latency_after <= headroom * latency_before + _EPS


def graduation_gate(
    before: dict[str, float],
    after: dict[str, float],
    *,
    n_labels: int,
    latency_before: float,
    latency_after: float,
    min_labels: int,
    latency_headroom: float,
    hook_budget_ms: float,
    min_margin: float,
    experiment_recorded: bool = True,
    rollback_exists: bool = True,
) -> dict[str, Any]:
    """Hard pre-apply gate enforcing all six graduation conditions.

    ``before``/``after`` are the ``{"precision_at_k", "noise_at_k", ...}`` dicts
    the tuner already measured (``gate_metrics``). Returns a structured verdict
    ``{"ok": bool, "reasons": [...]}`` — ``ok`` is True only when every
    condition holds; ``reasons`` names each condition that failed. Pure and
    total: never reads flags, never touches the store, never raises.
    """
    reasons: list[str] = []

    prec_b = _as_float(before.get("precision_at_k"))
    prec_a = _as_float(after.get("precision_at_k"))
    noise_b = _as_float(before.get("noise_at_k"))
    noise_a = _as_float(after.get("noise_at_k"))

    # 1. minimum sample count
    if not min_labels_gate(n_labels, min_labels):
        reasons.append(f"insufficient_labels: n_labels={n_labels} < min={min_labels}")
    # 2. precision improves by >= min_margin
    if not margin_gate(prec_b, prec_a, min_margin):
        reasons.append(
            f"precision_not_improved: after={prec_a:.4f} < before={prec_b:.4f} + margin={min_margin}"
        )
    # 3. noise not up
    if not noise_gate(noise_b, noise_a):
        reasons.append(f"noise_regressed: after={noise_a:.4f} > before={noise_b:.4f}")
    # 4. latency budget (headroom ratio AND absolute hook budget)
    if not latency_gate(
        latency_before, latency_after, headroom=latency_headroom, hook_budget_ms=hook_budget_ms
    ):
        if latency_after > hook_budget_ms + _EPS:
            reasons.append(
                f"latency_over_budget: p50={latency_after:.1f}ms > {hook_budget_ms:.0f}ms"
            )
        else:
            reasons.append(
                f"latency_regressed: p50={latency_after:.1f}ms > "
                f"{latency_headroom:g}x baseline {latency_before:.1f}ms"
            )
    # 5. experiment recorded (structural invariant the caller guarantees)
    if not experiment_recorded:
        reasons.append("experiment_not_recorded")
    # 6. rollback exists (structural invariant the caller guarantees)
    if not rollback_exists:
        reasons.append("rollback_missing")

    return {"ok": not reasons, "reasons": reasons}


def tune_gate_thresholds() -> dict[str, Any]:
    """The four graduation thresholds, read from the flag registry.

    Convenience for the tuner call site — keeps every ``MEMO_DREAM_TUNE_*``
    read in one place so ``graduation_gate`` stays pure. Returns kwargs ready to
    splat into :func:`graduation_gate` (``min_labels``, ``latency_headroom``,
    ``hook_budget_ms``, ``min_margin``).
    """
    from memo.flags import flag_float, flag_int

    min_labels = flag_int("MEMO_DREAM_TUNE_MIN_LABELS")
    headroom = flag_float("MEMO_DREAM_TUNE_LATENCY_HEADROOM")
    budget = flag_float("MEMO_DREAM_TUNE_HOOK_BUDGET_MS")
    margin = flag_float("MEMO_DREAM_TUNE_MIN_MARGIN")
    return {
        "min_labels": 12 if min_labels is None else min_labels,
        "latency_headroom": 1.25 if headroom is None else headroom,
        "hook_budget_ms": 3000.0 if budget is None else budget,
        "min_margin": 0.0 if margin is None else margin,
    }


# --- pure learning-metric helpers --------------------------------------------


def correction_rate(verdict_rows: list[dict[str, Any]]) -> tuple[float | None, int]:
    """(fraction of next-turn verdicts that were negative/correction, sample).

    ``(None, 0)`` when no verdicts are present — a rate over zero turns is not
    meaningful, so it is reported as absent rather than 0.0.
    """
    verdicts = [str(r.get("verdict")) for r in verdict_rows if r.get("verdict")]
    total = len(verdicts)
    if total == 0:
        return None, 0
    bad = sum(1 for v in verdicts if v in _NEGATIVE_VERDICTS)
    return bad / total, total


def created_used_ratio(grounded_total: int, surfaced_total: int) -> float | None:
    """created->used: grounded observations / surfaced observations.

    ``None`` when nothing was surfaced (no denominator).
    """
    if surfaced_total <= 0:
        return None
    return grounded_total / surfaced_total


# --- learning-metrics snapshot -----------------------------------------------


def learning_metrics(
    cfg: Any,
    mem: Any,
    *,
    params_version: str,
    k: int = 5,
    before: dict[str, float] | None = None,
    after: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Read-only snapshot of the eleven learning metrics from existing sources.

    Reuses the ``before``/``after`` the tuner already measured (does NOT re-run
    ``evaluate``). Every metric is computed defensively: a missing log or a
    failing source yields ``None`` for that key and a note in ``errors`` —
    the function never raises. Runs only inside ``memo dream run`` (nightly).
    """
    errors: list[str] = []
    state_dir = Path(cfg.state_dir)
    headline = after or before or {}

    metrics: dict[str, Any] = {
        "params_version": params_version,
        "k": k,
        # 1-2. precision@k / noise@k — from the tuner's own measurement.
        "precision_at_k": _num(headline.get("precision_at_k")),
        "noise_at_k": _num(headline.get("noise_at_k")),
    }
    if before is not None:
        metrics["precision_before"] = _num(before.get("precision_at_k"))
        metrics["noise_before"] = _num(before.get("noise_at_k"))
    if after is not None:
        metrics["precision_after"] = _num(after.get("precision_at_k"))
        metrics["noise_after"] = _num(after.get("noise_at_k"))

    # 3-4. answerability + grounding coverage — from grounded_rate.
    grounded = _safe(errors, "grounded_rate", lambda: _grounded_rate(state_dir), {})
    # answerability is a log-derived proxy (answer_rate_knowledge), NOT the
    # eval's pack-answerability (which is unimplemented) — labelled as such.
    metrics["answerability"] = _num(grounded.get("answer_rate_knowledge"))
    metrics["answerability_note"] = "proxy: answer_rate_knowledge (log-derived, not pack)"
    metrics["grounding_coverage"] = _num(grounded.get("measurement_coverage"))
    metrics["grounded_rate"] = _num(grounded.get("grounded_rate"))

    # 5. later_usefulness — realized grounded fraction of the live cohort.
    lu_frac, lu_n = _safe(
        errors, "later_usefulness", lambda: _online_fraction(state_dir, params_version), (None, 0)
    )
    metrics["later_usefulness"] = _num(lu_frac) if lu_n else None
    metrics["later_usefulness_cohort"] = int(lu_n or 0)

    # 6. correction_rate — from the next-turn verdict log.
    cr_rate, cr_n = _safe(
        errors, "correction_rate", lambda: correction_rate(_verdict_rows(state_dir)), (None, 0)
    )
    metrics["correction_rate"] = _num(cr_rate)
    metrics["correction_sample"] = int(cr_n or 0)

    # 7. synthesis_acceptance — reuse fraction of consolidate's synthesis memos.
    metrics["synthesis_acceptance"] = _num(
        _safe(errors, "synthesis_acceptance", lambda: _reuse_fraction(mem), None)
    )

    # 8. created->used — grounded / surfaced over the outcome window.
    metrics["created_used_ratio"] = _num(
        _safe(errors, "created_used_ratio", lambda: _created_used(state_dir), None)
    )

    # 9-10. p50 / p95 recall latency — p95 lives ONLY here.
    p50, p95 = _safe(errors, "latency", lambda: _latency_percentiles(state_dir), (None, None))
    metrics["latency_p50_ms"] = p50
    metrics["latency_p95_ms"] = p95

    # 11. reopened_contradiction_rate — no faithful durable source today; the
    # reopen transition is not stamped, so expose None rather than fabricate.
    metrics["reopened_contradiction_rate"] = None
    metrics["reopened_contradiction_note"] = "needs_event_stamp: reopen() nulls resolved_at"

    metrics["errors"] = errors
    return metrics


# --- source adapters (thin, defensive) ---------------------------------------


def _grounded_rate(state_dir: Path) -> dict[str, Any]:
    from memo import dashboard_metrics

    return dashboard_metrics.grounded_rate(state_dir)


def _online_fraction(state_dir: Path, params_version: str) -> tuple[float, int]:
    from memo import dream_tune_online

    return dream_tune_online.online_fraction(state_dir, params_version)


def _verdict_rows(state_dir: Path) -> list[dict[str, Any]]:
    from memo.dashboard_logs import read_verdict_log

    return read_verdict_log(state_dir)


def _reuse_fraction(mem: Any) -> float | None:
    from memo import dream_reuse

    return dream_reuse.consolidated_reuse(mem).get("reuse_fraction")


def _created_used(state_dir: Path) -> float | None:
    from memo import outcome

    u = outcome.compute_utilities(state_dir)
    return created_used_ratio(int((u.get("surfaced_total", 0) and u.get("grounded_total", 0)) or 0), int(u.get("surfaced_total", 0) or 0))


def _latency_percentiles(state_dir: Path) -> tuple[float | None, float | None]:
    """p50/p95 of the dominant recall path (most-sampled) from recall_metrics."""
    from memo import recall_metrics

    summary = recall_metrics.summarize(state_dir)
    if not summary:
        return None, None
    # Dominant path = most samples; name breaks ties deterministically.
    path = max(summary, key=lambda p: (int(summary[p].get("count", 0)), p))
    row = summary[path]
    return _num(row.get("p50")), _num(row.get("p95"))


# --- small utilities ---------------------------------------------------------


def _safe[T](errors: list[str], name: str, fn: Callable[[], T], default: Any) -> T:
    """Run ``fn``; on any failure record it and return ``default`` (never raise).

    ``default`` is typed ``Any`` (not ``T``) so the type variable binds to the
    callable's return alone — a literal fallback like ``{}`` / ``(None, 0)`` must
    not narrow ``T`` below what ``fn`` actually yields.
    """
    try:
        return fn()
    except Exception as exc:  # defensive: a broken source must not kill the snapshot
        errors.append(f"{name}: {type(exc).__name__}: {exc}")
        return default


def _num(value: Any) -> float | None:
    """Coerce a numeric metric to float; None for bool/None/non-numeric."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_float(value: Any) -> float:
    """Coerce a gate input to float; 0.0 for None/non-numeric (defensive)."""
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0
