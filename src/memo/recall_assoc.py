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
    budget_ms: int = flag_int("MEMO_ASSOCIATIVE_BUDGET_MS") or 300
    deadline = time.monotonic() + budget_ms / 1000.0
    try:
        seed_ids = [r.id for r in relevant]
        hops: int = flag_int("MEMO_ASSOCIATIVE_HOPS") or 2
        limit: int = flag_int("MEMO_ASSOCIATIVE_LIMIT") or 2
        min_act: float = flag_float("MEMO_ASSOCIATIVE_MIN_ACTIVATION") or 0.5
        hits = associate(
            seed_ids,
            store=memory.graph,
            codegraph_adj=_codegraph_adj(),
            hops=hops,
            limit=limit,
            exclude_ids=frozenset(seed_ids),
            min_activation=min_act,
        )
        if time.monotonic() > deadline:
            return []
        out: list[Any] = []
        for h in hits:
            rec = memory.get(h.id)
            if rec is not None:
                out.append(NudgeItem(id=h.id, title=getattr(rec, "title", h.id), via=h.via))
        return out
    except Exception:
        return []
