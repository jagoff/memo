"""Dream pass — MinHash+LSH-blocked LLM entity canonicalization (K1).

`GraphStore.canonicalize_existing()` folds only EXACT fold_key matches —
"FastAPI"/"fast-api" merge, but "memo recall daemon"/"memo recall daemons"
never do. Confirming every fuzzy pair with the helper LLM is O(n²) MLX
calls — infeasible nightly. This pass blocks the pair space first with
MinHash+LSH (graph_minhash.py, pure Python) so the LLM is consulted only
on pairs that share an LSH bucket, capped per night. The receipt records
`pairs_naive` (what an unblocked sweep would cost) vs `pairs_blocked` vs
`llm_calls`, so the saving is measured on every run.

OFF by default — opt in with MEMO_DREAM_ENTITY_CANON_ENABLED. Never in the
5s recall hook. Mirrors the dream_communities.py pass shape.
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)

_SYS = (
    "You decide whether two knowledge-graph entity names refer to the SAME "
    'real-world thing. Reply ONLY with JSON {"same": true} or {"same": false}. '
    "Spelling/plural/separator variants of one thing are the same entity; "
    "distinct projects, people, or technologies are not."
)


def _llm_same_entity(mem: Any, name_a: str, name_b: str) -> bool | None:
    import json as _json

    from memo.memory.record import chat_with_timeout

    out = chat_with_timeout(
        mem._ensure_chat(),
        timeout=15,
        model=mem.cfg.helper_model,
        messages=[
            {"role": "system", "content": _SYS},
            {"role": "user", "content": f'A: "{name_a}"\nB: "{name_b}"'},
        ],
        options={"temperature": 0.0, "max_tokens": 16, "thinking": False},
    )
    if out is None:
        return None
    text = ((out.get("message") or {}).get("content") or "").strip()
    try:
        obj = _json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("same"), bool):
        return None
    return bool(obj["same"])


def run_entity_canon(
    cfg: Any,
    mem: Any,
    *,
    max_pairs: int = 30,
    dry_run: bool = False,
) -> dict[str, Any]:
    """One blocked-canonicalization pass. Never raises. OFF unless flagged."""
    from memo.flags import flag_bool
    from memo.graph_minhash import candidate_pairs

    res: dict[str, Any] = {
        "status": "noop",
        "entities": 0,
        "pairs_naive": 0,
        "pairs_blocked": 0,
        "llm_calls": 0,
        "merged": [],
    }
    if not flag_bool("MEMO_DREAM_ENTITY_CANON_ENABLED"):
        res["status"] = "disabled"
        return res
    try:
        # Exact fold_key folding FIRST, so the fuzzy stage never re-litigates
        # what deterministic canonicalization already merges for free.
        mem.graph.canonicalize_existing()
        rows = mem.graph.list_entities(min_mentions=1)
        by_name = {r["name"]: r for r in rows}
        res["entities"] = len(by_name)
        n = len(by_name)
        res["pairs_naive"] = n * (n - 1) // 2  # cost of an unblocked LLM sweep
        pairs = candidate_pairs(by_name.keys())
        res["pairs_blocked"] = len(pairs)
        for a, b, est in pairs[:max_pairs]:
            ra, rb = by_name.get(a), by_name.get(b)
            if ra is None or rb is None:  # endpoint already merged this run
                continue
            res["llm_calls"] += 1
            if _llm_same_entity(mem, a, b) is not True:
                continue
            # Canonical = more mentions; tie-break prefers the shorter name.
            keep, drop = (
                (ra, rb)
                if (ra["mention_count"], -len(a)) >= (rb["mention_count"], -len(b))
                else (rb, ra)
            )
            if dry_run:
                res["merged"].append(
                    {"keep": keep["name"], "drop": drop["name"], "est": est, "dry_run": True}
                )
                continue
            mem.graph.merge_entity_pair(keep["id"], drop["id"], drop["name"])
            by_name.pop(drop["name"], None)
            res["merged"].append({"keep": keep["name"], "drop": drop["name"], "est": est})
        res["status"] = "done"
    except Exception as exc:
        res["status"] = "error"
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res
