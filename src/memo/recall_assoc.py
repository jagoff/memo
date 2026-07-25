"""Recall-path adapter for associative recall: turn the top-K into a nudge.

Bridges the pure `associate()` engine to the recall hook — resolves titles,
applies the time guard, and returns objects shaped for `render_recall_context`'s
nudge slot (`.id`, `.title`). Degrades to `[]` on any error.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from datetime import UTC
from typing import Any

from memo.associative import associate
from memo.flags import flag_bool, flag_float, flag_int


@dataclass(frozen=True)
class NudgeItem:
    id: str
    title: str
    via: str
    # True when the hit shares a memory↔memory semantic_relations edge with a
    # seed at confidence >= dream_edge_verify.VERIFIED_CONFIDENCE (earned by
    # the nightly grounded co-use pass). Drives the label only, never ranking.
    verified: bool = False


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

        # When Negative Recall is on, failure_pattern anti-memories are surfaced
        # only in their own ⛔ AVOID block (and excluded from normal recall). The
        # graph associate() walk excludes seed_ids but NOT by type, so a
        # failure_pattern connected to a normal seed could resurface in the "🔗
        # Also connected" tail — a duplicate of the ⛔ block. Drop it here.
        from memo.negative_recall import FAILURE_PATTERN_TYPE

        _drop_failure_patterns = flag_bool("MEMO_NEGATIVE_RECALL_ENABLED")

        seed_ids = [r.id for r in relevant]
        hops: int = 2 if (_h := flag_int("MEMO_ASSOCIATIVE_HOPS")) is None else _h
        limit: int = 2 if (_l := flag_int("MEMO_ASSOCIATIVE_LIMIT")) is None else _l
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
            if _drop_failure_patterns and getattr(rec, "type", None) == FAILURE_PATTERN_TYPE:
                continue  # surfaced only in the ⛔ block — never in the assoc tail
            adj = h.activation * _recency_weight(getattr(rec, "updated", "") or "")
            scored.append((adj, NudgeItem(id=h.id, title=getattr(rec, "title", h.id), via=h.via)))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [item for _, item in scored[:limit]]
        ver = _verified_pair_ids(memory.graph, [i.id for i in top], seed_ids)
        if ver:
            top = [replace(i, verified=True) if i.id in ver else i for i in top]
        return top
    except Exception:
        return []


def _verified_pair_ids(graph: Any, hit_ids: list[str], seed_ids: list[str]) -> set[str]:
    """Hit ids that share a memory↔memory ``semantic_relations`` edge with any
    seed at confidence >= ``dream_edge_verify.VERIFIED_CONFIDENCE`` — the one
    shared threshold the nightly edge-verify pass promotes toward, so the
    recall label and the pass agree by construction.

    One batch query per nudge (hook-budget-friendly): the confidence filter
    alone keeps the scan tiny — extractor priors max out at 0.82, so only
    pass-promoted edges clear it. Ids are matched on their 8-char prefixes
    (grounding.log convention). Degrades to ``set()`` on any error.
    """
    try:
        from memo.dream_edge_verify import VERIFIED_CONFIDENCE

        conn = getattr(graph, "_conn", None)
        if conn is None or not hit_ids or not seed_ids:
            return set()
        rows = conn.execute(
            "SELECT source_id, target_id FROM semantic_relations "
            "WHERE source_kind = 'memory' AND target_kind = 'memory' "
            "AND confidence >= ?",
            (VERIFIED_CONFIDENCE,),
        ).fetchall()
        hits8 = {str(h)[:8]: h for h in hit_ids}
        seeds8 = {str(s)[:8] for s in seed_ids}
        out: set[str] = set()
        for row in rows:
            s8, t8 = str(row[0])[:8], str(row[1])[:8]
            if s8 in hits8 and t8 in seeds8:
                out.add(hits8[s8])
            if t8 in hits8 and s8 in seeds8:
                out.add(hits8[t8])
        return out
    except Exception:
        return set()


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

    The "· unverified" framing drops only when EVERY shown item is verified —
    i.e. carries an edge the nightly ``dream_edge_verify`` pass promoted past
    ``VERIFIED_CONFIDENCE`` from grounded co-use (conservative on a mixed
    nudge). Label text only; ranking is untouched.

    ``token_budget <= 0`` means no cap (always append).  Otherwise the line is
    skipped when ``len(context) + len(line) > token_budget * 4``.
    """
    if not nudge:
        return context
    parts = "; ".join(f"[{h.id[:8]}] {h.title} — via {h.via}" for h in nudge)
    label = (
        "via graph"
        if all(getattr(h, "verified", False) for h in nudge)
        else "via graph · unverified"
    )
    line = f"\n_🔗 Also connected ({label}): {parts}._"
    if token_budget > 0 and len(context) + len(line) > token_budget * 4:
        return context
    return context + line
