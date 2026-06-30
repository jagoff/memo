"""Recall-path adapter for associative recall: turn the top-K into a nudge.

Bridges the pure `associate()` engine to the recall hook — resolves titles,
applies the time guard, and returns objects shaped for `render_recall_context`'s
nudge slot (`.id`, `.title`). Degrades to `[]` on any error.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
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
        )
        if time.monotonic() > deadline:
            return []
        out: list[Any] = []
        for h in hits:
            rec = memory.get(h.id)
            # Skip records that vanished or were soft-forgotten — a stale hint is
            # worse than none, and the recall block is meant to be trustworthy.
            if rec is None or (getattr(rec, "extra", None) or {}).get(IS_FORGOTTEN_KEY):
                continue
            out.append(NudgeItem(id=h.id, title=getattr(rec, "title", h.id), via=h.via))
            if len(out) >= limit:
                break
        return out
    except Exception:
        return []


def render_associative_line(context: str, nudge: list[Any], *, token_budget: int) -> str:
    """Append an associative nudge line to *context* if it fits the token budget.

    Format::

        _🔗 También conectado (vía grafo · no verificado): [id8] title — vía via; …_

    ``token_budget <= 0`` means no cap (always append).  Otherwise the line is
    skipped when ``len(context) + len(line) > token_budget * 4``.
    """
    if not nudge:
        return context
    parts = "; ".join(f"[{h.id[:8]}] {h.title} — vía {h.via}" for h in nudge)
    line = f"\n_🔗 También conectado (vía grafo · no verificado): {parts}._"
    if token_budget > 0 and len(context) + len(line) > token_budget * 4:
        return context
    return context + line
