"""The graduation controller: for each candidate, shadow-eval → record → decide
flip / revert / accumulate. Pure over its inputs except the ledger + overlay
writes; the ``evaluator`` seam keeps it MLX-free under test."""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from memo.graduation import ledger, overlay_ops
from memo.graduation.registry import Candidate, NumericCandidate, default_candidates
from memo.graduation.shadow import shadow_eval
from memo.graduation.shadow_numeric import shadow_eval_numeric

if TYPE_CHECKING:
    from memo.eval_recall import LabelSet


def run_graduation_controller(
    cfg: Any,
    mem: Any,
    *,
    evaluator: Callable[..., dict[str, Any]] | None = None,
    candidates: list[Candidate | NumericCandidate] | None = None,
    k: int = 5,
    labels: LabelSet | None = None,
    dry_run: bool = False,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    cands = candidates if candidates is not None else default_candidates()
    env = os.environ if env is None else env
    state_dir = cfg.state_dir
    out: list[dict[str, Any]] = []

    if labels is None:
        from memo.dream_tune import build_labels

        labels, _ = build_labels(cfg)

    for cand in cands:
        if env.get(cand.flag) is not None:
            out.append({"flag": cand.flag, "status": "vetoed"})
            continue

        is_numeric = isinstance(cand, NumericCandidate)
        # evaluator seam: explicit stub wins for both kinds; else pick by type.
        if evaluator is not None:
            res = evaluator(mem, cand, k=cand.k, labels=labels)
        elif isinstance(cand, NumericCandidate):
            res = shadow_eval_numeric(mem, cand, k=cand.k, labels=labels)
        else:
            res = shadow_eval(mem, cand, k=cand.k, labels=labels)
        win = bool(res.get("win"))
        delta_prec = float(res.get("delta_prec", 0.0))
        if not dry_run:
            ledger.record(state_dir, cand.flag, {
                "verdict": "confirmed" if win else "reverted",
                "realized_delta": delta_prec,
                "delta_noise": float(res.get("delta_noise", 0.0)),
            })
        s = ledger.streak(state_dir, cand.flag)

        live = (
            overlay_ops.overlay_value(state_dir, cand.flag) is not None
            if is_numeric
            else overlay_ops.is_flipped_on(state_dir, cand.flag)
        )
        if live:
            if not win:
                if not dry_run:
                    overlay_ops.revert(state_dir, cand.flag)
                status = "reverted"
            else:
                status = "live"
        elif cand.auto_flip and s >= cand.k:
            if not dry_run:
                if is_numeric:
                    overlay_ops.flip_numeric(
                        state_dir, cand.flag, float(res["best_value"]),
                        evidence={"streak": s, **res})
                else:
                    overlay_ops.flip_on(state_dir, cand.flag, evidence={"streak": s, **res})
            status = "graduated"
        elif not cand.auto_flip:
            status = "report_only"
        else:
            status = "accumulating"

        out.append({"flag": cand.flag, "status": status, "streak": s, "k": cand.k, **res})

    return {"candidates": out}
