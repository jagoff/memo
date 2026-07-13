"""Dream pass — Phase 2 DISTILLATION (upward re-abstraction of mature clusters).

Distinct from the one-shot ``synthesize_cross_cluster`` (which fires on any
loose cluster) and from the graph/session/folder siblings: this pass is
PERIODIC, ALTITUDE-AWARE, and gated on MATURITY — a cluster only distills when
its members are old enough, confident enough, and corroborated enough to be
settled knowledge. It is ADDITIVE + LINKING: it saves ONE new
``type=synthesis`` memory (``synthesis_kind=distillation``) whose provenance
points back at its sources; it NEVER supersedes, archives, or deletes a source,
so removing the distilled memory is a pure no-op rollback.

Structure mirrors ``dream_communities`` / ``dream_folder_abstracts``: pure
maturity/decision functions (fully testable with injected cluster/synthesize/
exists callables) + a guarded ``run_distill`` orchestrator that wires the real
store batches, LLM, and save. OFF by default (``MEMO_DREAM_DISTILL_ENABLED``).
Nightly only; never in the 5s recall hook.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# --- pure core (testable) ----------------------------------------------------


def provenance_hash(ids: list[str]) -> str:
    """Stable 16-hex hash over a cluster's source ids (order-independent).
    Same shape as dream_communities.provenance_hash so dedup behaves alike."""
    return hashlib.sha256("|".join(sorted(ids)).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class MaturityStats:
    size: int
    mean_support: float
    mean_confidence: float
    min_age_days: float  # age of the YOUNGEST member — the whole cluster must be settled


def _age_days(created: str, now: _dt.datetime) -> float:
    """Days between an ISO ``created`` string and ``now``. Unknown/unparseable
    → 0.0 (fresh), so an undated member conservatively fails the age floor."""
    if not created:
        return 0.0
    try:
        dt = _dt.datetime.fromisoformat(str(created).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.UTC)
    return max(0.0, (now - dt.astimezone(_dt.UTC)).total_seconds() / 86400.0)


def cluster_maturity(members: list[dict[str, Any]], *, now: _dt.datetime) -> MaturityStats:
    """Aggregate a cluster's per-member {id, created, confidence, support_count}
    into MaturityStats. Empty cluster → all-zero stats (fails every floor)."""
    n = len(members)
    if n == 0:
        return MaturityStats(size=0, mean_support=0.0, mean_confidence=0.0, min_age_days=0.0)
    mean_support = sum(float(m.get("support_count", 0)) for m in members) / n
    mean_confidence = sum(float(m.get("confidence", 1.0)) for m in members) / n
    min_age = min(_age_days(str(m.get("created") or ""), now) for m in members)
    return MaturityStats(
        size=n,
        mean_support=mean_support,
        mean_confidence=mean_confidence,
        min_age_days=min_age,
    )


def is_mature(
    stats: MaturityStats,
    *,
    min_cluster: int,
    min_support: float,
    min_confidence: float,
    min_age_days: float,
) -> bool:
    """The maturity gate — every floor must clear."""
    return (
        stats.size >= min_cluster
        and stats.mean_support >= min_support
        and stats.mean_confidence >= min_confidence
        and stats.min_age_days >= min_age_days
    )


def corroboration_weighted_confidence(stats: MaturityStats) -> str:
    """Distilled memory's confidence string, lifted by corroboration. Maps onto
    the synthesize_cross_cluster confidence vocabulary (high/medium/low)."""
    if stats.mean_confidence >= 0.75 and stats.mean_support >= 3:
        return "high"
    if stats.mean_confidence >= 0.5:
        return "medium"
    return "low"


def assemble_clusters(
    items: list[dict[str, Any]],
    cluster_index_lists: list[list[int]],
    *,
    health: dict[str, dict[str, float]],
    support: dict[str, int],
    created_by_id: dict[str, str],
    min_cluster: int,
    now: _dt.datetime,
) -> list[dict[str, Any]]:
    """Turn (_pull_embeddings items, _greedy_cluster index lists) into candidate
    clusters carrying maturity. Drops clusters below ``min_cluster``. Missing
    per-id health/support default to confidence=1.0 / support=0 (the store's own
    missing-row convention). Largest cluster first."""
    out: list[dict[str, Any]] = []
    for idxs in cluster_index_lists:
        if len(idxs) < min_cluster:
            continue
        ids = [items[i]["id"] for i in idxs]
        titles = [str(items[i].get("title") or "") for i in idxs]
        members = [
            {
                "id": mid,
                "created": created_by_id.get(mid, ""),
                "confidence": float((health.get(mid) or {}).get("confidence", 1.0)),
                "support_count": int(support.get(mid, 0)),
            }
            for mid in ids
        ]
        out.append({"ids": ids, "titles": titles, "stats": cluster_maturity(members, now=now)})
    out.sort(key=lambda c: -c["stats"].size)
    return out


def decide_distillations(
    clusters: list[dict[str, Any]],
    *,
    synthesize_fn: Callable[[dict[str, Any]], dict[str, str] | None],
    exists_fn: Callable[[str], bool],
    is_mature_fn: Callable[[MaturityStats], bool],
    dry_run: bool,
    max_clusters: int,
) -> list[dict[str, Any]]:
    """Turn candidate clusters into save-decisions. Mirrors
    dream_communities.decide_syntheses, plus the maturity gate. Deduped by
    provenance hash. ``synthesize_fn(cluster) -> {title, body} | None`` is the
    LLM; ``exists_fn(phash) -> bool`` the dedup; saving is the caller's job."""
    decisions: list[dict[str, Any]] = []
    for cl in clusters[:max_clusters]:
        phash = provenance_hash(cl["ids"])
        if not is_mature_fn(cl["stats"]):
            decisions.append({"status": "immature", "provenance_hash": phash})
            continue
        if exists_fn(phash):
            decisions.append({"status": "skip_exists", "provenance_hash": phash})
            continue
        if dry_run:
            decisions.append(
                {"status": "would_save", "provenance_hash": phash, "size": cl["stats"].size}
            )
            continue
        synth = synthesize_fn(cl)
        if not synth or not synth.get("title") or not synth.get("body"):
            decisions.append({"status": "synth_failed", "provenance_hash": phash})
            continue
        decisions.append(
            {
                "status": "save",
                "provenance_hash": phash,
                "title": synth["title"],
                "body": synth["body"],
                "provenance": cl["ids"],
                "confidence": corroboration_weighted_confidence(cl["stats"]),
            }
        )
    return decisions


# --- orchestrator (guarded; wires real store/LLM/save) -----------------------

_SYS = (
    "You distill a cluster of a user's SETTLED memories into ONE higher-altitude "
    "principle — the durable rule/decision/fact they collectively establish that "
    "no single one states alone. Abstract ONLY what the members state; never "
    'invent. Reply ONLY with JSON {"title": str, "insight": str}: a short title '
    "and a 2-4 sentence principle. No preamble, no markdown."
)


def _llm_distill(mem: Any, cluster: dict[str, Any]) -> dict[str, str] | None:
    import json as _json

    from memo.memory.record import chat_with_timeout

    titles = "\n".join(f"- {t[:140]}" for t in cluster["titles"][:15] if t)
    prompt = (
        f"A mature cluster of {cluster['stats'].size} corroborated memories "
        f"(mean support {cluster['stats'].mean_support:.1f}):\n{titles}\n\n"
        "What higher-altitude principle do they collectively establish?"
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


def run_distill(
    cfg: Any,
    mem: Any,
    *,
    min_cluster: int = 3,
    min_support: float = 2.0,
    min_age_days: float = 14.0,
    max_clusters: int = 5,
    threshold: float = 0.78,
    min_confidence: float = 0.5,
    dry_run: bool = False,
) -> dict[str, Any]:
    """One nightly distillation pass. Never raises. OFF unless the flag is set.
    ADDITIVE + LINKING only — never supersedes/archives/deletes a source."""
    from memo.flags import flag_bool

    res: dict[str, Any] = {"status": "noop", "distilled": []}
    if not flag_bool("MEMO_DREAM_DISTILL_ENABLED"):
        res["status"] = "disabled"
        return res
    try:
        items = mem._pull_embeddings(exclude_types={"reference", "synthesis"})
        if not items:
            res["status"] = "skipped"
            return res
        clusters_idx = mem._greedy_cluster(items, threshold)
        ids = [it["id"] for it in items]
        health = mem.store.get_health_batch(ids)
        support = mem.store.get_support_batch(ids)
        # get_batch returns a list[dict] (NOT keyed by id) — re-key on "id".
        created_by_id = {r["id"]: str(r.get("created") or "") for r in mem.store.get_batch(ids)}
        now = _dt.datetime.now(_dt.UTC)

        candidates = assemble_clusters(
            items,
            clusters_idx,
            health=health,
            support=support,
            created_by_id=created_by_id,
            min_cluster=min_cluster,
            now=now,
        )
        if not candidates:
            res["status"] = "skipped"
            return res

        def _exists(phash: str) -> bool:
            try:
                hits = mem.search(f"distill {phash}", limit=5, disable_reranker=True)
                return any(phash in (getattr(h, "body", "") or "") for h in hits)
            except Exception:
                return False

        def _mature(stats: MaturityStats) -> bool:
            return is_mature(
                stats,
                min_cluster=min_cluster,
                min_support=min_support,
                min_confidence=min_confidence,
                min_age_days=min_age_days,
            )

        decisions = decide_distillations(
            candidates,
            synthesize_fn=lambda cl: _llm_distill(mem, cl),
            exists_fn=_exists,
            is_mature_fn=_mature,
            dry_run=dry_run,
            max_clusters=max_clusters,
        )
        for d in decisions:
            if d.get("status") == "save":
                try:
                    mem.save(
                        content=f"{d['body']}\n\n[distill {d['provenance_hash']}]",
                        type_="synthesis",
                        title=d["title"],
                        extra={
                            "synthesis_kind": "distillation",
                            "synthesis_sources": d["provenance"],
                            "synthesis_confidence": d["confidence"],
                        },
                    )
                    d["status"] = "saved"
                except Exception as exc:
                    d["status"] = "save_failed"
                    d["error"] = f"{type(exc).__name__}: {exc}"
        res["distilled"] = decisions
        res["status"] = "done"
    except Exception as exc:  # surfaced, never silent
        res["status"] = "error"
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res
