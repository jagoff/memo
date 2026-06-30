"""Recall-path adapter for associative recall: turn the top-K into a nudge.

Bridges the pure `associate()` engine to the recall hook — resolves titles,
applies the time guard, and returns objects shaped for `render_recall_context`'s
nudge slot (`.id`, `.title`). Degrades to `[]` on any error.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC
from typing import Any

from memo.associative import associate
from memo.flags import flag_bool, flag_float, flag_int


@dataclass(frozen=True)
class NudgeItem:
    id: str
    title: str
    via: str


def _codegraph_adj() -> dict[str, set[str]] | None:
    try:
        from memo import codegraph_loader

        return codegraph_loader.load()[0]
    except Exception:
        return None


def build_nudge(memory: Any, relevant: list[Any]) -> list[Any]:
    if not flag_bool("MEMO_RECALL_ASSOCIATIVE") or not relevant:
        return []
    budget_ms_v = flag_int("MEMO_ASSOCIATIVE_BUDGET_MS")
    budget_ms: int = 300 if budget_ms_v is None else budget_ms_v
    deadline = time.monotonic() + budget_ms / 1000.0
    try:
        from memo.lifecycle import IS_FORGOTTEN_KEY

        seed_ids = [r.id for r in relevant]
        hops: int = flag_int("MEMO_ASSOCIATIVE_HOPS") or 2
        limit: int = flag_int("MEMO_ASSOCIATIVE_LIMIT") or 2
        min_act_v = flag_float("MEMO_ASSOCIATIVE_MIN_ACTIVATION")
        min_act: float = 0.5 if min_act_v is None else min_act_v

        # Load codegraph first, before the deadline check
        cg = _codegraph_adj()

        # Check deadline BEFORE the potentially expensive associate() call
        if time.monotonic() > deadline:
            return []

        hits = associate(
            seed_ids,
            store=memory.graph,
            codegraph_adj=cg,
            hops=hops,
            limit=limit + 5,  # buffer: backfill after dropping forgotten/missing
            exclude_ids=frozenset(seed_ids),
            min_activation=min_act,
            deadline=deadline,
        )
        if time.monotonic() > deadline:
            return []
        # Resolve records, drop forgotten/missing, then re-rank with a mild
        # recency preference: among similarly-connected memories, a fresher one
        # is the better hint. Recency only modulates; graph activation dominates.
        scored: list[tuple[float, Any]] = []
        for h in hits:
            rec = memory.get(h.id)
            if rec is None or (getattr(rec, "extra", None) or {}).get(IS_FORGOTTEN_KEY):
                continue
            adj = h.activation * _recency_weight(getattr(rec, "updated", "") or "")
            scored.append((adj, NudgeItem(id=h.id, title=getattr(rec, "title", h.id), via=h.via)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]
    except Exception:
        return []


def _recency_weight(updated_iso: str) -> float:
    """Mild recency factor: today ~1.0, ~18mo ~0.5, older trends to 0. Unknown -> 1.0."""
    from datetime import datetime

    try:
        dt = datetime.fromisoformat(updated_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        days = (datetime.now(UTC) - dt).days
        return 1.0 / (1.0 + max(0, days) / 540.0)
    except Exception:
        return 1.0


def render_associative_line(context: str, nudge: list[Any], *, token_budget: int) -> str:
    """Append an associative nudge line to *context* if it fits the token budget.

    Format::

        _🔗 Also connected (via graph · unverified): [id8] title — via via; …_

    ``token_budget <= 0`` means no cap (always append).  Otherwise the line is
    skipped when ``len(context) + len(line) > token_budget * 4``.
    """
    if not nudge:
        return context
    parts = "; ".join(f"[{h.id[:8]}] {h.title} — via {h.via}" for h in nudge)
    line = f"\n_🔗 Also connected (via graph · unverified): {parts}._"
    if token_budget > 0 and len(context) + len(line) > token_budget * 4:
        return context
    return context + line
