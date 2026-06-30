"""Dream v2 pass — synthesize one insight per knowledge-graph community (spec 3).

Detects communities in the ENTITY graph (code symbols excluded so the clusters
are about knowledge, not code structure) and abstracts each significant cluster
into a durable ``type=synthesis`` memory (``synthesis_kind=community``), with the
community's entities + source memory ids as provenance. Dedup by provenance
hash. OFF by default — opt in with ``MEMO_DREAM_COMMUNITIES_ENABLED``.

Mirrors ``dream_consolidate`` (cross-session synthesis); the only difference is
the clustering source: graph communities instead of recurring sessions.
"""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Callable
from typing import Any

# A community whose representative is one of the top-N weighted-degree entities
# is hub-dominated (an incoherent grab-bag), not a theme — excluded from synthesis.
_HUB_TOP_N = 3

_SYS = (
    "You abstract a cluster of a user's memories into one durable insight. "
    "Reply ONLY with JSON {\"title\": str, \"insight\": str}. The insight names "
    "the cross-cutting theme the cluster shares; no preamble, no markdown."
)


def provenance_hash(keys: list[str]) -> str:
    """Stable 16-hex hash over a community's member keys (order-independent)."""
    return hashlib.sha256("|".join(sorted(keys)).encode("utf-8")).hexdigest()[:16]


def _community_key(cluster: dict[str, Any]) -> list[str]:
    """A drift-tolerant identity for a community: its representative plus the
    alphabetically-first entities. Hashing the full membership re-synthesizes a
    near-duplicate whenever one bridging memory grows/merges the component, so we
    anchor on the most stable core instead.
    """
    rep = str(cluster.get("representative") or "")
    return [rep, *sorted(cluster.get("entities") or [])[:8]]


def community_clusters(
    mem: Any, *, min_size: int, max_communities: int, max_size: int = 40
) -> list[dict[str, Any]]:
    """Entity-only graph communities -> [{entities, representative, memory_ids}].

    Forces the entity-only graph (codegraph off) for the detection so the
    clusters reflect knowledge, then maps each community's entities back to the
    memories that mention them. Communities larger than ``max_size`` are skipped:
    a hub entity (e.g. "memo") fuses most of the graph into one giant blob that
    is "everything", not a coherent theme to synthesize.
    """
    # Force the entity-only graph per-call (no process-global env mutation):
    # community synthesis is about knowledge, not code structure.
    comms = mem.navigator.detect_communities(min_size=min_size, use_codegraph=False)

    # Hub-led communities are incoherent grab-bags — a mega-hub (e.g. "memo")
    # pulls scattered nodes into one cluster under the size cap. Drop a community
    # whose representative is one of the graph's top weighted-degree hubs.
    from collections import defaultdict as _dd

    degree: dict[str, float] = _dd(float)
    with contextlib.suppress(Exception):
        for a, b, w in mem.graph.all_weighted_edges():
            degree[a] += w
            degree[b] += w
    hubs = {n for n, _ in sorted(degree.items(), key=lambda kv: (-kv[1], kv[0]))[:_HUB_TOP_N]}

    out: list[dict[str, Any]] = []
    eligible = [
        c for c in comms if c.size <= max_size and c.representative_entity not in hubs
    ]
    for c in sorted(eligible, key=lambda c: -c.size)[:max_communities]:
        seen: set[str] = set()
        mem_ids: list[str] = []
        for ent in c.entities:
            for mid in mem.graph.entity_memories(ent):
                if mid not in seen:
                    seen.add(mid)
                    mem_ids.append(mid)
        out.append(
            {
                "entities": list(c.entities),
                "representative": c.representative_entity,
                "memory_ids": mem_ids,
            }
        )
    return out


def decide_syntheses(
    clusters: list[dict[str, Any]],
    *,
    synthesize_fn: Callable[[dict[str, Any]], dict[str, str] | None],
    exists_fn: Callable[[str], bool],
    dry_run: bool = False,
    max_clusters: int = 5,
) -> list[dict[str, Any]]:
    """Turn clusters into save-decisions, deduped by provenance hash."""
    decisions: list[dict[str, Any]] = []
    for cl in clusters[:max_clusters]:
        phash = provenance_hash(_community_key(cl))
        if exists_fn(phash):
            decisions.append({"status": "skip_exists", "provenance_hash": phash})
            continue
        if dry_run:
            decisions.append(
                {
                    "status": "would_save",
                    "provenance_hash": phash,
                    "representative": cl.get("representative"),
                    "entities": cl["entities"],
                }
            )
            continue
        synth = synthesize_fn(cl)
        if not synth:
            decisions.append({"status": "synth_failed", "provenance_hash": phash})
            continue
        decisions.append(
            {
                "status": "save",
                "provenance_hash": phash,
                "title": synth["title"],
                "body": synth["body"],
                "provenance": cl["entities"],
                "memory_ids": cl["memory_ids"][:20],
            }
        )
    return decisions


def _llm_synthesize_community(mem: Any, cluster: dict[str, Any]) -> dict[str, str] | None:
    import json as _json

    from memo.memory.record import chat_with_timeout

    titles: list[str] = []
    for mid in cluster["memory_ids"][:12]:
        rec = mem.get(mid)
        if rec is not None and getattr(rec, "title", None):
            titles.append(str(rec.title)[:120])
    ents = ", ".join(cluster["entities"][:15])
    body = "\n".join(f"- {t}" for t in titles)
    prompt = (
        f"A knowledge-graph community around '{cluster.get('representative')}' "
        f"(entities: {ents}) spans these memories:\n{body}\n\n"
        "What recurring theme / decision / area do they share?"
    )
    out = chat_with_timeout(
        mem._ensure_chat(),
        timeout=30,
        model=mem.cfg.helper_model,
        messages=[{"role": "system", "content": _SYS}, {"role": "user", "content": prompt}],
        options={"temperature": 0.0, "max_tokens": 256, "thinking": False},
    )
    if out is None:
        return None
    text = ((out.get("message") or {}).get("content") or "").strip()
    try:
        obj = _json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    title, insight = obj.get("title"), obj.get("insight")
    if not title or not insight:
        return None
    return {"title": str(title)[:120], "body": str(insight)}


def run_synthesize_communities(
    cfg: Any,
    mem: Any,
    *,
    min_size: int = 4,
    max_communities: int = 5,
    dry_run: bool = False,
) -> dict[str, Any]:
    """One community-synthesis pass. Never raises. OFF unless the flag is set."""
    from memo.flags import flag_bool

    res: dict[str, Any] = {"status": "noop", "synthesized": []}
    if not flag_bool("MEMO_DREAM_COMMUNITIES_ENABLED"):
        res["status"] = "disabled"
        return res
    try:
        # Refresh the substrate so synthesis runs on a de-fragmented, weighted
        # graph (cheap + idempotent; best-effort).
        try:
            mem.graph.canonicalize_existing()
            mem.graph.rebuild_edges()
        except Exception as exc:  # pragma: no cover - defensive
            res["rebuild_error"] = f"{type(exc).__name__}: {exc}"
        clusters = community_clusters(mem, min_size=min_size, max_communities=max_communities)
        if not clusters:
            res["status"] = "skipped"
            return res

        def _exists(phash: str) -> bool:
            # Best-effort dedup: the phash is embedded verbatim in the saved body
            # ("[community <hash>]"). Search a small window (limit>1 so the target
            # is not lost behind one higher-ranked hit) and match the literal hash.
            try:
                hits = mem.search(f"community {phash}", limit=5, disable_reranker=True)
                return any(phash in (getattr(h, "body", "") or "") for h in hits)
            except Exception:
                return False

        decisions = decide_syntheses(
            clusters,
            synthesize_fn=lambda cl: _llm_synthesize_community(mem, cl),
            exists_fn=_exists,
            dry_run=dry_run,
            max_clusters=max_communities,
        )
        for d in decisions:
            if d.get("status") == "save":
                try:
                    mem.save(
                        content=f"{d['body']}\n\n[community {d['provenance_hash']}]",
                        type="synthesis",
                        title=d["title"],
                        extra={
                            "synthesis_kind": "community",
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
