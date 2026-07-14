"""Belief-revision decision: which side of a contradiction (if any) supersedes.

Shared by `memo maintain` (cli_maintain) and the nightly Dream contradict pass
(cli_dream_passes) so the recency-clobber fix lives in exactly one place.

Pure and READ-ONLY over the store (no writes, no MLX). Runs only in the
maintenance path, never the 5s recall hook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memo.flags import flag_bool, flag_float, flag_int

# actions
ARCHIVE = "archive"  # supersede the dominated side
COMPETING = "competing"  # keep both: neither side dominates
HOLD_OPEN = "hold_open"  # keep both open: high-support loser held for triage (legacy C2)


@dataclass(frozen=True)
class SupersedeDecision:
    action: str  # ARCHIVE | COMPETING | HOLD_OPEN
    dominant_id: str
    dominated_id: str
    reason: str
    support_dominated: int


def _score(health: dict[str, dict[str, float]], id_: str) -> float:
    """confidence x roi_score for an id; 1.0 when absent (neutral)."""
    h = health.get(id_)
    if not h:
        return 1.0
    return float(h.get("confidence", 1.0)) * float(h.get("roi_score", 1.0))


def supersede_decision(mem: Any, *, older_id: str, newer_id: str) -> SupersedeDecision:
    """Resolve a contradiction between the recency-older and recency-newer sides.

    Flags OFF (default) → legacy recency: the newer side wins, the older is
    archived, unless the older's support_count >= MEMO_SUPERSEDE_SUPPORT_GATE
    (the existing C2 gate), in which case it is held open for triage.

    MEMO_BELIEF_COMPETING ON → dominance by trust (confidence x roi_score):
    within MEMO_SUPERSEDE_MARGIN the pair is 'competing' (both kept); otherwise
    the weaker side is archived (subject to the support gate on that side).
    """
    gate = flag_int("MEMO_SUPERSEDE_SUPPORT_GATE") or 0

    def _support(ids: list[str]) -> dict[str, int]:
        try:
            return {i: int(v) for i, v in mem.store.get_support_batch(ids).items()}
        except Exception:
            return {}

    if not flag_bool("MEMO_BELIEF_COMPETING"):
        support_older = _support([older_id]).get(older_id, 0) if gate > 0 else 0
        if gate > 0 and support_older >= gate:
            return SupersedeDecision(
                HOLD_OPEN,
                newer_id,
                older_id,
                f"support {support_older} >= gate {gate}",
                support_older,
            )
        return SupersedeDecision(ARCHIVE, newer_id, older_id, "recency: newer wins", support_older)

    # belief mode: dominance by trust score
    try:
        health = mem.store.get_health_batch([older_id, newer_id])
    except Exception:
        health = {}
    s_old, s_new = _score(health, older_id), _score(health, newer_id)
    # tie (or newer stronger) → newer dominates, preserving legacy lean on exact ties
    dominant = newer_id if s_new >= s_old else older_id
    dominated = older_id if dominant == newer_id else newer_id
    support = _support([older_id, newer_id])
    support_dominated = support.get(dominated, 0)

    margin = flag_float("MEMO_SUPERSEDE_MARGIN")
    margin = 0.15 if margin is None else margin
    if abs(s_new - s_old) <= margin:
        return SupersedeDecision(
            COMPETING,
            dominant,
            dominated,
            f"within margin ({s_old:.3f} vs {s_new:.3f} <= {margin:g})",
            support_dominated,
        )
    if gate > 0 and support_dominated >= gate:
        return SupersedeDecision(
            HOLD_OPEN,
            dominant,
            dominated,
            f"dominated support {support_dominated} >= gate {gate}",
            support_dominated,
        )
    return SupersedeDecision(
        ARCHIVE,
        dominant,
        dominated,
        f"trust dominance ({s_old:.3f} vs {s_new:.3f})",
        support_dominated,
    )


def nway_competing_pairs(pairs: list[tuple[int, str, str]]) -> set[int]:
    """Return pair_ids whose two memories belong to a connected component of
    3+ mutually-contradicting memories. Pure graph (union-find), no store I/O."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for _pid, a, b in pairs:
        union(a, b)

    size: dict[str, int] = {}
    members: dict[str, set[str]] = {}
    for node in list(parent):
        root = find(node)
        members.setdefault(root, set()).add(node)
    for root, nodes in members.items():
        size[root] = len(nodes)

    return {pid for pid, a, b in pairs if size.get(find(a), 0) >= 3}
