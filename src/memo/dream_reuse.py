"""F4 consolidate-reuse metric — read-only.

Measures whether the ``type=synthesis`` memories created by
``consolidate-episodes`` are actually grounded/reused in real recall.

A synthesis memory is "reused" when its id 8-char prefix appears in
grounding.log in a row where ``grounding_used(row)`` is True — the same
single production decision used by every other grounding consumer.

Never mutates the store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memo.memory import Memory


def consolidated_reuse(memory: Memory, *, limit: int = 1000) -> dict[str, Any]:
    """Read-only F4 metric: of the ``type=synthesis`` memories consolidate
    created, how many are actually reused (grounded) in real recall.

    Args:
        memory: A Memory facade instance (or any object with ``store.list_recent``
            and ``cfg.state_dir``).
        limit: Maximum number of synthesis memories to inspect (most-recently
            updated first, matching ``list_recent`` ordering).

    Returns:
        Dict with keys:
          - ``n_consolidated`` — total type=synthesis memories found
          - ``n_reused`` — how many appear in grounding.log as "used"
          - ``reuse_fraction`` — n_reused / n_consolidated (0.0 if none)
          - ``cross_session`` — count of *reused* memories where
            ``extra.synthesis_kind == "cross_session"``
    """
    from memo.dashboard_logs import read_grounding_log
    from memo.dashboard_metrics import grounding_used

    # Enumerate type=synthesis memories via the store's list_recent API.
    rows = memory.store.list_recent(limit=limit, type_="synthesis")
    n_consolidated = len(rows)

    if n_consolidated == 0:
        return {
            "n_consolidated": 0,
            "n_reused": 0,
            "reuse_fraction": 0.0,
            "cross_session": 0,
        }

    # Index by 8-char prefix — the same truncation grounding.log uses for recall_id.
    id_to_extra: dict[str, dict[str, Any]] = {}
    for r in rows:
        prefix = (r.get("id") or "")[:8]
        if prefix:
            id_to_extra[prefix] = r.get("extra") or {}

    # Read grounding log once; build the set of actually-reused 8-char prefixes.
    state_dir = memory.cfg.state_dir
    grounding_rows = read_grounding_log(state_dir)
    reused_prefixes: set[str] = {
        row["recall_id"] for row in grounding_rows if grounding_used(row) and row.get("recall_id")
    }

    n_reused = 0
    cross_session = 0
    for prefix, extra in id_to_extra.items():
        if prefix in reused_prefixes:
            n_reused += 1
            if extra.get("synthesis_kind") == "cross_session":
                cross_session += 1

    reuse_fraction = n_reused / n_consolidated

    return {
        "n_consolidated": n_consolidated,
        "n_reused": n_reused,
        "reuse_fraction": round(reuse_fraction, 4),
        "cross_session": cross_session,
    }
