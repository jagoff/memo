"""`memo dream` maintenance passes — the per-phase implementations.

Extracted verbatim from `cli_dream.py` to keep that module focused on the
Click command wiring. Each `_run_*` helper performs one maintenance pass over
the corpus and returns a compact summary; `_build_orientation` is the
read-only pre-mutation inventory. All are imported (and re-exported) by
`cli_dream` so `from memo.cli_dream import _run_eviction` keeps working.
"""

from __future__ import annotations

import logging as _logging
from typing import TYPE_CHECKING, Any

from memo.transcript_miner import mine_transcripts

if TYPE_CHECKING:
    from memo.memory.facade import Memory

_log = _logging.getLogger(__name__)


def _build_orientation(mem: Memory) -> dict:
    """Read-only corpus inventory — runs before any mutation."""
    conn = mem.store._conn
    result: dict = {
        "total": 0,
        "by_type": {},
        "low_roi": 0,
        "stale_candidates": 0,
        "open_contradictions": 0,
        "unindexed_entities": 0,
    }
    try:
        row = conn.execute("SELECT COUNT(*) AS n FROM meta WHERE type != 'reference'").fetchone()
        result["total"] = int(row["n"]) if row else 0
    except Exception:  # noqa: S110
        pass

    try:
        rows = conn.execute(
            "SELECT type, COUNT(*) AS n FROM meta WHERE type != 'reference' GROUP BY type"
        ).fetchall()
        result["by_type"] = {r["type"]: int(r["n"]) for r in rows}
    except Exception:  # noqa: S110
        pass

    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM meta m "
            "LEFT JOIN memory_health h ON h.id = m.id "
            "WHERE COALESCE(h.roi_score, 1.0) < 0.3 AND m.type != 'reference'"
        ).fetchone()
        result["low_roi"] = int(row["n"]) if row else 0
    except Exception:  # noqa: S110
        pass

    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM meta m "
            "LEFT JOIN access a ON a.id = m.id "
            "WHERE m.updated < datetime('now', '-365 days') "
            "AND COALESCE(a.access_count, 0) = 0 "
            "AND m.type != 'reference'"
        ).fetchone()
        result["stale_candidates"] = int(row["n"]) if row else 0
    except Exception:  # noqa: S110
        pass

    try:
        pairs = mem.contradict_store.list_open()
        result["open_contradictions"] = len(pairs)
    except Exception:  # noqa: S110
        pass

    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM meta m "
            "WHERE m.type != 'reference' "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM entity_memoria em "
            "  JOIN entities e ON e.id = em.entity_id "
            "  WHERE em.memoria_id = m.id"
            ")"
        ).fetchone()
        result["unindexed_entities"] = int(row["n"]) if row else 0
    except Exception:  # noqa: S110
        pass

    return result


def _run_signal_gather(since_days: int, file_limit: int = 20) -> dict:
    """Run transcript mining and return a compact summary.

    Never raises — exceptions are captured in the returned dict.
    """
    try:
        res = mine_transcripts(since_days=since_days, file_limit=file_limit)
        return {
            "files_processed": res.get("files_processed", 0),
            "memorias_saved": len(res.get("saved") or []),
            "skipped_dup": res.get("skipped_dup", 0),
        }
    except Exception as exc:
        return {"files_processed": 0, "memorias_saved": 0, "skipped_dup": 0, "error": str(exc)}


def _run_prune_floor(
    mem: Memory,
    roi_floor: float,
    min_age_days: int,
    dry_run: bool,
) -> list[dict]:
    """Archive memories below roi_floor with zero access and age >= min_age_days.

    Returns list of {id, roi_score, days_old} candidates (even in dry-run).
    """
    candidates = mem.store.prune_floor_candidates(roi_floor=roi_floor, min_age_days=min_age_days)
    if not dry_run:
        for c in candidates:
            try:
                mem.lifecycle.archive_memoria(c["id"])
            except Exception as exc:
                _log.warning("prune_floor: archive failed for %s: %s", c["id"], exc)
    return candidates


def _run_eviction(mem: Memory, max_count: int, dry_run: bool) -> list[dict]:
    """Archive LFU candidates until corpus size <= max_count.

    Returns list of {id, access_count} archived (or would-archive in dry-run).
    """
    conn = mem.store._conn
    try:
        total_row = conn.execute(
            "SELECT COUNT(*) AS n FROM meta WHERE type != 'reference'"
        ).fetchone()
        total = int(total_row["n"]) if total_row else 0
    except Exception:
        return []

    excess = total - max_count
    if excess <= 0:
        return []

    candidates = mem.store.eviction_candidates(
        policy="lfu",
        limit=excess,
        exclude_types={"reference", "synthesis"},
    )
    if not dry_run:
        for c in candidates:
            try:
                mem.lifecycle.archive_memoria(c["id"])
            except Exception as exc:
                _log.warning("eviction: archive failed for %s: %s", c["id"], exc)
    return [{"id": c["id"], "access_count": c.get("access_count", 0)} for c in candidates]


def _run_compress(mem: Memory, threshold: int, dry_run: bool) -> list[dict]:
    """Compress verbose memories (body > threshold chars) to 2-3 sentences.

    Returns list of {id, original_len, compressed_len}.
    """
    conn = mem.store._conn
    try:
        rows = conn.execute(
            "SELECT m.id, m.path FROM meta m "
            "JOIN fts ON fts.id = m.id "
            "WHERE m.type NOT IN ('reference','synthesis') "
            "AND length(fts.body) > ?",
            (threshold,),
        ).fetchall()
    except Exception:
        return []

    if not rows:
        return []

    from memo.memory.record import chat_with_timeout

    chat = mem._ensure_chat()
    results = []
    for row in rows:
        mid = row["id"]
        try:
            rec = mem.get(mid)
            if not rec or not rec.body:
                continue
            body_len = len(rec.body)
            if body_len <= threshold:
                continue
            user_prompt = (
                "Compress the following memory note to 2-3 concise sentences "
                "preserving all key facts, decisions, and context. "
                "Output ONLY the compressed text, no preamble.\n\n" + rec.body[:4000]
            )
            chat_out = chat_with_timeout(
                chat,
                timeout=30,
                model=mem.cfg.helper_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a concise technical writer. Compress memory notes.",
                    },
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.0, "max_tokens": 256, "thinking": False},
            )
            if chat_out is None:
                continue
            compressed = ((chat_out.get("message") or {}).get("content") or "").strip()
            if not compressed or len(compressed) >= body_len:
                continue
            if not dry_run:
                mem.update(mid, content=compressed)
            results.append({"id": mid, "original_len": body_len, "compressed_len": len(compressed)})
        except Exception as exc:
            _log.warning("compress: failed for %s: %s", mid, exc)
    return results


def _run_prewarm_queries(cfg: Any, mem: Memory, n: int) -> dict:
    """Pre-embed the n most recent unique queries from recall.log.

    Warms the LRU embed cache so the next recall-hook invocation hits cached
    embeddings instead of recomputing them. Never raises.
    """
    try:
        from memo.dashboard_logs import read_recall_log

        entries = read_recall_log(cfg.state_dir, limit=n * 3)
        seen: list[str] = []
        for e in entries:
            q = (e.get("prompt") or "").strip()
            if q and q not in seen:
                seen.append(q)
            if len(seen) >= n:
                break

        warmed = 0
        for q in seen:
            try:
                mem.embedder.embed_query(q)
                warmed += 1
            except Exception:  # noqa: S110
                pass
        return {"queries_warmed": warmed, "queries_available": len(seen)}
    except Exception as exc:
        return {"queries_warmed": 0, "queries_available": 0, "error": str(exc)}


def _run_presynthesis(cfg: Any, mem: Memory, top_n: int, dry_run: bool) -> list[dict]:
    """Pre-synthesize clusters for the top recurring queries.

    Reads recall.log, picks the top_n most frequent queries, runs a focused
    synthesis pass on the memories each query surfaces. Returns a list of
    synthesis results per query.
    """
    try:
        from collections import Counter

        from memo.dashboard_logs import read_recall_log

        entries = read_recall_log(cfg.state_dir, limit=200)
        counts: Counter = Counter()
        for e in entries:
            q = (e.get("prompt") or "").strip()
            if q:
                counts[q] += 1

        top_queries = [q for q, _ in counts.most_common(top_n)]
        if not top_queries:
            return []

        all_results = []
        for query in top_queries:
            try:
                hits = mem.search(query, limit=20, disable_reranker=True)
                if len(hits) < 3:
                    continue
                # Synthesize across the hit cluster
                source_ids = [h.id for h in hits]
                result = mem.synthesize_cross_cluster(
                    dry_run=dry_run, min_cluster_size=3, max_clusters=1
                )
                if result:
                    all_results.append(
                        {
                            "query": query[:80],
                            "hits": len(source_ids),
                            "synthesized": len(result),
                        }
                    )
            except Exception as exc:
                _log.warning("presynthesis: failed for query %r: %s", query[:50], exc)
        return all_results
    except Exception as exc:
        return [{"error": str(exc)}]
