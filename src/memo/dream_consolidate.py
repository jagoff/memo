"""`memo dream consolidate-episodes` — episodic→semantic consolidation (Phase 2).

The sleep-consolidation half nothing else does: per-turn mining
(``signal_gather``) and cross-*memory* synthesis (``synthesize_cross_cluster``)
both work *within* a session or *across notes*. This pass works across
*sessions* — it groups recent episodes by project and, where a project recurs
across ≥N distinct sessions, abstracts ONE durable cross-session memory
(``type=synthesis``, ``synthesis_kind=cross_session``) with the session ids as
provenance. Dedup by provenance hash; no destructive episodic decay (deferred).
OFF by default (``MEMO_DREAM_CONSOLIDATE_EPISODES_ENABLED``).

The clustering + consolidation decisions are pure functions (fully tested with
injected synthesize/exists callables); the orchestrator wires the real episode
store, LLM, and save, and is guarded so it never breaks the dream pipeline.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from typing import Any

from memo.dream_synthesis_tags import synthesis_tags

# --- pure core (testable) ----------------------------------------------------


def _project_key(cwd: str) -> str:
    """A stable project key from a working dir — its basename (the repo/folder)."""
    cwd = (cwd or "").rstrip("/")
    return os.path.basename(cwd) if cwd else ""


def provenance_hash(session_ids: list[str]) -> str:
    """Stable hash of a cluster's sessions — used to skip re-consolidation."""
    payload = "|".join(sorted(session_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cluster_by_project(
    episodes: list[dict[str, Any]], *, min_sessions: int
) -> list[dict[str, Any]]:
    """Group episodes by project; keep clusters spanning ≥ min_sessions sessions."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for ep in episodes:
        key = _project_key(str(ep.get("cwd") or ""))
        if key:
            groups.setdefault(key, []).append(ep)
    clusters: list[dict[str, Any]] = []
    for key, eps in groups.items():
        sids = sorted({str(e["session_id"]) for e in eps if e.get("session_id")})
        if len(sids) >= min_sessions:
            clusters.append({"project": key, "episodes": eps, "session_ids": sids})
    return clusters


def consolidate_clusters(
    clusters: list[dict[str, Any]],
    *,
    synthesize_fn: Callable[[dict[str, Any]], dict[str, str] | None],
    exists_fn: Callable[[str], bool],
    dry_run: bool,
    max_clusters: int = 5,
) -> list[dict[str, Any]]:
    """Consolidate each cluster into one cross-session memory (or skip).

    ``synthesize_fn(cluster) -> {title, body} | None`` (the LLM; None = nothing
    durable). ``exists_fn(provenance_hash) -> bool`` (dedup). Saving itself is
    the caller's job — this returns the decisions + payloads.
    """
    results: list[dict[str, Any]] = []
    for cl in clusters[:max_clusters]:
        phash = provenance_hash(cl["session_ids"])
        if exists_fn(phash):
            results.append({"project": cl["project"], "status": "exists"})
            continue
        out = synthesize_fn(cl)
        if not out or not out.get("title") or not out.get("body"):
            results.append({"project": cl["project"], "status": "skipped"})
            continue
        results.append(
            {
                "project": cl["project"],
                "status": "would_save" if dry_run else "save",
                "title": out["title"],
                "body": out["body"],
                "provenance": cl["session_ids"],
                "provenance_hash": phash,
            }
        )
    return results


# --- orchestrator (guarded; wires real store/LLM/save) -----------------------

_SYS = (
    "You consolidate recurring cross-session work into ONE durable note. "
    "Abstract ONLY what the session summaries state — never invent. "
    'Reply with a single line of JSON: {"title": "...", "insight": "..."} '
    'or {"title": null} if there is no durable cross-session insight.'
)


def _llm_synthesize(mem: Any, cluster: dict[str, Any]) -> dict[str, str] | None:
    import json as _json

    from memo.memory.record import chat_with_timeout

    summaries = "\n".join(
        f"- {str(e.get('summary') or '')[:300]}" for e in cluster["episodes"][:12]
    )
    prompt = (
        f"Project '{cluster['project']}' across {len(cluster['session_ids'])} sessions:\n"
        f"{summaries}\n\nWhat recurring decision/approach/goal do these share?"
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
    title = obj.get("title")
    insight = obj.get("insight")
    if not title or not insight:
        return None
    return {"title": str(title)[:120], "body": str(insight)}


def run_consolidate_episodes(
    cfg: Any,
    mem: Any,
    *,
    min_sessions: int = 2,
    window: int = 50,
    max_clusters: int = 5,
    dry_run: bool = False,
) -> dict[str, Any]:
    """One nightly cross-session consolidation pass. Never raises."""
    res: dict[str, Any] = {"status": "noop", "consolidated": []}
    try:
        from memo.resume._index import open_store

        store = open_store(cfg)
        if store is None or store.count() == 0:
            res["status"] = "skipped"  # episodic memory disabled or empty
            return res
        episodes = store.recent(limit=window)
        clusters = cluster_by_project(episodes, min_sessions=min_sessions)
        if not clusters:
            return res

        def _exists(_phash: str) -> bool:
            # dedup is best-effort: a search miss simply allows a (deduped-on-save) write
            try:
                hits = mem.search(f"cross-session {_phash}", limit=1, disable_reranker=True)
                return any(_phash in (getattr(h, "body", "") or "") for h in hits)
            except Exception:
                return False

        decisions = consolidate_clusters(
            clusters,
            synthesize_fn=lambda cl: _llm_synthesize(mem, cl),
            exists_fn=_exists,
            dry_run=dry_run,
            max_clusters=max_clusters,
        )
        for d in decisions:
            if d.get("status") == "save":
                try:
                    mem.save(
                        content=f"{d['body']}\n\n[cross-session {d['provenance_hash']}]",
                        type_="synthesis",
                        tags=synthesis_tags("cross_session"),
                        title=d["title"],
                        extra={
                            "synthesis_kind": "cross_session",
                            "synthesis_sources": d["provenance"],
                        },
                    )
                    d["status"] = "saved"
                except Exception as exc:
                    d["status"] = "save_failed"
                    d["error"] = f"{type(exc).__name__}: {exc}"
        res["consolidated"] = decisions
        res["status"] = "done"
    except Exception as exc:  # surfaced, never silent
        res["status"] = "error"
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res
