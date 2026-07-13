"""HyPE read-path max-fold — merge question-space candidates into vec hits.

The nightly HyPE pass (`dream_hype.py`) indexes LLM-generated hypothetical
questions per durable memory in `HypeStore`. At read time, the query is
matched against that QUESTION space too, and the two candidate sets fold:

- memory already in the doc hits → `score = max(doc_score, question_score)`
  (annotated `hype=True` when the question side won);
- memory the doc vector alone did NOT bring → appended as a new candidate
  with the question score (the real gain: candidate *generation*, not just
  re-ranking).

Pure function — no flags, no I/O beyond the injected `store.knn` and
`fetch_meta`. The gating (`MEMO_HYPE_ENABLED`) lives at the call site in
`memory/search_ops.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memo.store.hype_store import HypeStore


def hype_fold(
    doc_hits: list[dict[str, Any]],
    query_embedding: list[float],
    store: HypeStore,
    fetch_meta: Callable[[str], dict[str, Any] | None],
    *,
    pool: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Max-fold question-space kNN results into `doc_hits`.

    `doc_hits` are the dicts `VecStore.search` returns (carry `id` and
    `score`). `store.knn(query_embedding, k=pool)` yields the best question
    per memory as `{memory_id, question, score}` — `pool` caps the RAW
    question pool before the per-memory collapse, so fewer rows than `pool`
    is normal. New memories are materialized via `fetch_meta(memory_id)`
    (a meta row dict, or None when deleted/unknown → skipped).

    Inputs are never mutated; folded/added hits are fresh dicts. Result is
    sorted by score desc and cut to `limit`. Empty knn → `doc_hits` as-is.
    """
    question_hits = store.knn(query_embedding, k=pool)
    if not question_hits:
        return doc_hits

    q_score_by_id: dict[str, float] = {}
    for qh in question_hits:
        mid = str(qh["memory_id"])
        score = float(qh["score"])
        if score > q_score_by_id.get(mid, -1.0):
            q_score_by_id[mid] = score

    doc_ids = {str(h["id"]) for h in doc_hits}
    out: list[dict[str, Any]] = []
    for hit in doc_hits:
        q_score = q_score_by_id.get(str(hit["id"]))
        if q_score is not None and q_score > float(hit.get("score") or 0.0):
            out.append({**hit, "score": q_score, "hype": True})
        else:
            out.append(hit)

    for mid, q_score in q_score_by_id.items():
        if mid in doc_ids:
            continue
        meta = fetch_meta(mid)
        if meta is None:
            continue
        out.append({**meta, "score": q_score, "hype": True})

    out.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
    return out[:limit]


__all__ = ["hype_fold"]
