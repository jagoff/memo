"""Chat pipeline orchestrator: retrieval → quality stages → synthesis, as events."""

from __future__ import annotations

import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from memo.chat.config import ChatConfig
from memo.chat.dedup import collapse_near_duplicates, score_of
from memo.chat.expand import allows_multi_query, classify_query, expand_query
from memo.chat.feedback import (
    SourceVoteStore,
    boost_positive_sources,
    boost_semantic,
    filter_negative_sources,
    question_key,
)
from memo.chat.fulldoc import dominant_doc_group, resolve_fulldoc
from memo.chat.fusion import normalize_scores, rrf_fuse
from memo.chat.rewrite import rewrite_query
from memo.chat.synthesis import filter_by_relevance, synthesize_stream
from memo.retrieval_boost import boost_for

_SNIPPET_CHARS = 700


def _record_to_source(r: Any) -> dict[str, Any]:
    body = str(getattr(r, "body", "") or "")
    return {
        "source": "memory",
        "id": str(getattr(r, "id", "")),
        "title": str(getattr(r, "title", "")),
        "type": str(getattr(r, "type", "")),
        "score": float(getattr(r, "score", None) or 0.0),
        "snippet": body[:_SNIPPET_CHARS],
        "path": str(getattr(r, "path", "") or ""),
    }


def _hit_to_source(h: Any) -> dict[str, Any]:
    return {
        "source": "vault",
        "id": str(getattr(h, "id", "")),
        "title": str(getattr(h, "path", "") or ""),
        "type": "repo",
        "score": float(getattr(h, "score", None) or 0.0),
        "snippet": str(getattr(h, "text", "") or "")[:_SNIPPET_CHARS],
        "path": str(getattr(h, "path", "") or ""),
        "repo_name": str(getattr(h, "repo_name", "") or ""),
        "locator": str(getattr(h, "locator", "") or ""),
    }


def _apply_title_boost(sources: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    out = []
    for s in sources:
        s = dict(s)
        factor = boost_for(
            query=query,
            filename=str(s.get("path") or ""),
            title=str(s.get("title") or ""),
        )
        if factor > 1.0 and isinstance(s.get("normalized_score"), (int, float)):
            s["normalized_score"] = round(float(s["normalized_score"]) * factor, 6)
            s["filename_boost"] = factor
        out.append(s)
    out.sort(key=score_of, reverse=True)
    return out


def chat_stream(
    memory: Any,
    question: str,
    *,
    history: list[dict[str, str]] | None = None,
    k: int | None = None,
) -> Iterator[dict[str, Any]]:
    cfg = ChatConfig.load(memory.cfg.state_dir)
    base_k = k or cfg.base_k
    t0 = time.monotonic()

    memo_query = rewrite_query(question, history)
    yield {"type": "stage", "stage": "retrieval", "query": memo_query}

    with ThreadPoolExecutor(max_workers=2) as pool:
        mem_future = pool.submit(memory.search, memo_query, limit=base_k, mode="hybrid")
        vault_future = pool.submit(memory.repo_search, memo_query, limit=base_k)
        mem_sources = [_record_to_source(r) for r in (mem_future.result() or [])]
        vault_sources = [_hit_to_source(h) for h in (vault_future.result() or [])]

    rankings = [mem_sources, vault_sources]
    if cfg.multi_query and allows_multi_query(classify_query(memo_query)):
        variants = expand_query(
            memory._ensure_chat(), memory.cfg.llm_model, memo_query, n=cfg.multi_query_n
        )
        for variant in variants:
            hits = memory.search(variant, limit=base_k, mode="hybrid") or []
            rankings.append([_record_to_source(r) for r in hits])

    fused = normalize_scores(rrf_fuse(rankings, limit=base_k))
    fused = _apply_title_boost(fused, memo_query)
    dominant = dominant_doc_group(fused) if cfg.fulldoc else None

    sources = collapse_near_duplicates(fused)
    store = SourceVoteStore(cfg.feedback_dir)
    latest = store.latest_by_pair()
    qkey = question_key(memo_query)
    sources = filter_negative_sources(sources, latest, qkey)
    sources = boost_positive_sources(sources, latest, qkey, factor=cfg.vote_boost)
    try:
        query_vec = memory.embedder.embed_query(memo_query)
    except Exception:
        query_vec = []
    if query_vec:
        sources = boost_semantic(
            sources,
            query_vec,
            store.load(),
            threshold=cfg.semantic_threshold,
            factor=cfg.vote_boost,
        )

    yield {"type": "context", "sources": sources[:base_k]}

    def _done(answer: str, synthesis_source: str) -> dict[str, Any]:
        return {
            "type": "done",
            "answer": answer,
            "sources": sources[:base_k],
            "synthesis_source": synthesis_source,
            "total_ms": int((time.monotonic() - t0) * 1000),
        }

    if dominant:
        doc = resolve_fulldoc(memory, dominant)
        if doc:
            yield {"type": "token", "text": doc["text"]}
            yield _done(doc["text"], f"memo.fulldoc.{doc['fulldoc_source']}")
            return

    head = filter_by_relevance(sources, floor=cfg.relevance_floor)[: cfg.synth_head]
    parts: list[str] = []
    try:
        for token in synthesize_stream(
            memory._ensure_chat(),
            memory.cfg.llm_model,
            question,
            head,
            max_tokens=cfg.answer_max_tokens,
        ):
            parts.append(token)
            yield {"type": "token", "text": token}
    except Exception:
        yield {"type": "error", "message": "synthesis failed", "answer_partial": "".join(parts)}
        return
    yield _done("".join(parts), "memo.chat")
