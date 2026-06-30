"""Dream v2 pass — synthesize one insight per articulation *bridge* (spec 3).

A bridge is an entity that is the sole connector between two otherwise-separate
regions of the knowledge graph. Each bridge is abstracted into one durable
``type=synthesis`` memory (``synthesis_kind=bridge``) naming the link
("<left_rep> and <right_rep> are connected via <bridge>"), with the bridge,
side representatives, and source memory ids as provenance. The link statement is
deterministic and graph-derived — never fabricated. Dedup by provenance hash.
OFF by default — opt in with ``MEMO_DREAM_BRIDGES_ENABLED``.

Sibling of ``dream_communities`` (one synthesis per cluster); here the unit is a
single articulation entity instead of a whole community.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from memo.dream_communities import provenance_hash
from memo.graph_bridges import find_bridges

_DATE_RE = re.compile(r"^\d{4}([-/]\d{1,2}){0,2}$")
# Generic co-mention tokens that are not meaningful link anchors.
_GENERIC_ANCHORS = frozenset(
    {
        "files", "file", "archivos", "archivo", "print", "log", "logs", "note",
        "notes", "data", "test", "tests", "todo", "string", "value", "none",
        "true", "false", "null", "error", "errors",
    }
)


def _is_junk_anchor(name: str) -> bool:
    """A bridge/representative that carries no real semantic link: a date, a bare
    number, or a generic co-mention token — these surfaced as noisy 'X via <date>'
    insights, so they are filtered before synthesis."""
    n = (name or "").strip().lower()
    if not n or n.isdigit() or _DATE_RE.match(n):
        return True
    return n in _GENERIC_ANCHORS


def _representative(side: list[str], adjacency: dict[str, dict[str, float]]) -> str:
    """Highest weighted-degree entity in a side; tie -> smallest name."""
    return min(
        side,
        key=lambda e: (-sum((adjacency.get(e) or {}).values()), e),
    )


def bridge_insights(
    mem: Any, *, min_side: int = 2, max_bridges: int = 5
) -> list[dict[str, Any]]:
    """Articulation bridges -> [{bridge, left, right, left_rep, right_rep, memory_ids}].

    Builds the weighted adjacency from the materialized entity edges, finds
    bridges, picks a representative per side, and maps the bridge + reps back to
    the memories that mention them. Deterministic.
    """
    adjacency: dict[str, dict[str, float]] = defaultdict(dict)
    for a, b, w in mem.graph.all_weighted_edges():
        adjacency[a][b] = w
        adjacency[b][a] = w

    bridges = find_bridges(dict(adjacency), min_side=min_side)
    # Largest combined span first; deterministic tie-break on the bridge name.
    bridges.sort(
        key=lambda br: (-(len(br["left"]) + len(br["right"])), br["bridge"])
    )

    out: list[dict[str, Any]] = []
    for br in bridges:
        if len(out) >= max_bridges:
            break
        left, right = list(br["left"]), list(br["right"])
        left_rep = _representative(left, adjacency)
        right_rep = _representative(right, adjacency)
        # Drop junk-anchor links (dates, numbers, generic tokens) and tautologies
        # where both sides share a representative — they read as near-noise.
        if (
            _is_junk_anchor(br["bridge"])
            or _is_junk_anchor(left_rep)
            or _is_junk_anchor(right_rep)
            or left_rep == right_rep
        ):
            continue
        seen: set[str] = set()
        mem_ids: list[str] = []
        for ent in (br["bridge"], left_rep, right_rep):
            for mid in mem.graph.entity_memories(ent):
                if mid not in seen:
                    seen.add(mid)
                    mem_ids.append(mid)
        out.append(
            {
                "bridge": br["bridge"],
                "left": left,
                "right": right,
                "left_rep": left_rep,
                "right_rep": right_rep,
                "memory_ids": mem_ids,
            }
        )
    return out


def _synthesize_bridge(mem: Any, bridge: dict[str, Any]) -> dict[str, str] | None:
    """Deterministic, graph-derived link statement — never fabricates."""
    left_rep, right_rep, mid = bridge["left_rep"], bridge["right_rep"], bridge["bridge"]
    title = f"{left_rep} ↔ {right_rep} via {mid}"[:120]
    body = f"{left_rep} and {right_rep} are connected via {mid}."
    return {"title": title, "body": body}


def decide_bridges(
    bridges: list[dict[str, Any]],
    *,
    synthesize_fn: Callable[[dict[str, Any]], dict[str, str] | None],
    exists_fn: Callable[[str], bool],
    dry_run: bool = False,
    max_bridges: int = 5,
) -> list[dict[str, Any]]:
    """Turn bridge insights into save-decisions, deduped by provenance hash."""
    decisions: list[dict[str, Any]] = []
    for br in bridges[:max_bridges]:
        provenance = [br["bridge"], br["left_rep"], br["right_rep"]]
        phash = provenance_hash(provenance)
        if exists_fn(phash):
            decisions.append({"status": "skip_exists", "provenance_hash": phash})
            continue
        if dry_run:
            decisions.append(
                {
                    "status": "would_save",
                    "provenance_hash": phash,
                    "bridge": br["bridge"],
                    "left_rep": br["left_rep"],
                    "right_rep": br["right_rep"],
                }
            )
            continue
        synth = synthesize_fn(br)
        if not synth:
            decisions.append({"status": "synth_failed", "provenance_hash": phash})
            continue
        decisions.append(
            {
                "status": "save",
                "provenance_hash": phash,
                "title": synth["title"],
                "body": synth["body"],
                "provenance": provenance,
                "memory_ids": br.get("memory_ids", [])[:20],
            }
        )
    return decisions


def run_synthesize_bridges(
    cfg: Any,
    mem: Any,
    *,
    min_side: int = 2,
    max_bridges: int = 5,
    dry_run: bool = False,
) -> dict[str, Any]:
    """One bridge-synthesis pass. Never raises. OFF unless the flag is set."""
    from memo.flags import flag_bool

    res: dict[str, Any] = {"status": "noop", "synthesized": []}
    if not flag_bool("MEMO_DREAM_BRIDGES_ENABLED"):
        res["status"] = "disabled"
        return res
    try:
        bridges = bridge_insights(mem, min_side=min_side, max_bridges=max_bridges)
        if not bridges:
            res["status"] = "skipped"
            return res

        def _exists(phash: str) -> bool:
            # Best-effort dedup: the phash is embedded verbatim in the saved body
            # ("[bridge <hash>]"). Search a small window and match the literal.
            try:
                hits = mem.search(f"bridge {phash}", limit=5, disable_reranker=True)
                return any(phash in (getattr(h, "body", "") or "") for h in hits)
            except Exception:
                return False

        decisions = decide_bridges(
            bridges,
            synthesize_fn=lambda br: _synthesize_bridge(mem, br),
            exists_fn=_exists,
            dry_run=dry_run,
            max_bridges=max_bridges,
        )
        for d in decisions:
            if d.get("status") == "save":
                try:
                    mem.save(
                        content=f"{d['body']}\n\n[bridge {d['provenance_hash']}]",
                        type="synthesis",
                        title=d["title"],
                        extra={
                            "synthesis_kind": "bridge",
                            "synthesis_sources": d["provenance"],
                            "synthesis_source_memories": d.get("memory_ids", []),
                        },
                    )
                    d["status"] = "saved"
                except Exception as exc:
                    d["status"] = "save_failed"
                    d["error"] = f"{type(exc).__name__}: {exc}"
        res["synthesized"] = decisions
        res["status"] = "done"
    except Exception as exc:
        res["status"] = "error"
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res
