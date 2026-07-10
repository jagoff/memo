"""Unified capability taxonomy for the public long-memory benchmarks.

`memo eval bench` (eval_bench.py) reports retrieval + QA per *raw* dataset
category (LoCoMo `multi_hop`/`temporal_reasoning`/…, LongMemEval
`single-session-user`/`knowledge-update`/…). Those labels don't line up
across datasets, so cross-dataset and cross-run comparison is hard.

This module adds the *auxiliary* view Memoria's benchmark uses (see its
`docs/memory-ability-taxonomy.md`): a fixed 6-bucket capability taxonomy that
normalizes both datasets' labels onto one axis. The raw per-category numbers
stay the primary report — this is a secondary rollup for cross-dataset
comparison and regression tracking, never a replacement.

Leaf module: pure mapping + arithmetic, no memo imports, so eval_bench and
tests can depend on it without cycles.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Mapping

# The six unified capability buckets (Memoria's memory-ability taxonomy).
CAPABILITY_BUCKETS: tuple[str, ...] = (
    "single_session_grounding",  # fact/context extraction within one session
    "preference_understanding",  # user preferences, not keyword hits
    "multi_session_synthesis",  # cross-session integration / summarization
    "temporal_state_tracking",  # time, ordering, state evolution
    "knowledge_update_conflict",  # new-vs-old updates, contradiction handling
    "abstention_constraint",  # decline w/o evidence + follow constraints
)

# Fallback bucket for a raw category we don't recognize. Kept OUT of the
# canonical six so an unmapped label surfaces loudly in the rollup instead of
# being silently miscredited to grounding.
OTHER_BUCKET = "other"

# Raw dataset category (normalized) -> capability bucket. Covers LoCoMo
# category names (eval_bench._LOCOMO_CATEGORY_NAMES) and LongMemEval /
# BEAM question types. Abstention is handled structurally in `bucket_for`
# (the abstention flag wins over the category), matching Memoria: an
# abstention question belongs to the abstention bucket regardless of the
# base topic it was derived from.
_CATEGORY_TO_BUCKET: dict[str, str] = {
    # --- single-session grounding ---
    "single_hop": "single_session_grounding",
    "single_session_user": "single_session_grounding",
    "single_session_assistant": "single_session_grounding",
    "information_extraction": "single_session_grounding",
    "open_domain": "single_session_grounding",
    # --- preference ---
    "single_session_preference": "preference_understanding",
    "preference_following": "preference_understanding",
    "preference": "preference_understanding",
    # --- multi-session synthesis ---
    "multi_hop": "multi_session_synthesis",
    "multi_session": "multi_session_synthesis",
    "multi_session_reasoning": "multi_session_synthesis",
    "summarization": "multi_session_synthesis",
    # --- temporal ---
    "temporal_reasoning": "temporal_state_tracking",
    "event_ordering": "temporal_state_tracking",
    # --- knowledge update / conflict ---
    "knowledge_update": "knowledge_update_conflict",
    "contradiction_resolution": "knowledge_update_conflict",
    # --- abstention / constraint ---
    "adversarial": "abstention_constraint",
    "abstention": "abstention_constraint",
    "instruction_following": "abstention_constraint",
}


def _norm(category: str) -> str:
    """Fold a raw category label to the mapping's key form."""
    return (category or "").strip().lower().replace("-", "_").replace(" ", "_")


def bucket_for(category: str, abstention: bool = False) -> str:
    """Map a raw dataset category to its capability bucket.

    An abstention question routes to `abstention_constraint` regardless of the
    base category it was derived from (LongMemEval `*_abs` keeps its topic
    `question_type`, but the *ability* under test is declining). Unknown
    categories fall to `OTHER_BUCKET` so they stay visible.
    """
    if abstention:
        return "abstention_constraint"
    return _CATEGORY_TO_BUCKET.get(_norm(category), OTHER_BUCKET)


def rollup_weighted(
    by_category: Mapping[str, Mapping[str, float | int]],
    metrics: Iterable[str],
    *,
    weight_key: str = "n_questions",
    bucket_of: Callable[[str], str] = bucket_for,
) -> dict[str, dict[str, float | int]]:
    """Roll per-category metric dicts up into capability buckets.

    Each metric is a question-count-weighted mean across the categories that
    fall in the same bucket (so a bucket dominated by a large category isn't
    skewed by a tiny sibling). Categories missing/zero `weight_key` are
    skipped. Returns `{bucket: {metric: value, "n_questions": int}}`.
    """
    metric_list = list(metrics)
    acc: dict[str, dict[str, float]] = {}
    for cat, vals in by_category.items():
        try:
            weight = float(vals.get(weight_key, 0) or 0)
        except (TypeError, ValueError):
            weight = 0.0
        if weight <= 0:
            continue
        slot = acc.setdefault(bucket_of(cat), {**{m: 0.0 for m in metric_list}, "n": 0.0})
        for m in metric_list:
            with contextlib.suppress(TypeError, ValueError):
                slot[m] += float(vals.get(m, 0) or 0) * weight
        slot["n"] += weight
    out: dict[str, dict[str, float | int]] = {}
    for bucket, slot in acc.items():
        total = slot.pop("n")
        rolled: dict[str, float | int] = {m: round(slot[m] / total, 3) for m in metric_list}
        rolled["n_questions"] = int(total)
        out[bucket] = rolled
    return out


__all__ = [
    "CAPABILITY_BUCKETS",
    "OTHER_BUCKET",
    "bucket_for",
    "rollup_weighted",
]
