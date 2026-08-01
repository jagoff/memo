"""Chat pipeline orchestrator: retrieval → quality stages → synthesis, as events."""

from __future__ import annotations

import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from memo.chat import whatsapp_live
from memo.chat.config import ChatConfig
from memo.chat.contacts_alias import build_index as build_contacts_index
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


def _whatsapp_live_source(cfg: ChatConfig, memo_query: str) -> list[dict[str, Any]] | None:
    """Resolve a single exclusive WA-live source for a recency-intent query.

    The semantic index only stores WhatsApp transcripts as ingested, day-
    granular chunks, so it structurally can't answer "el último mensaje con
    X" — this reads the live bridge DB directly for a message-granular,
    always-current answer instead. Returns ``None`` (caller falls back to
    normal semantic retrieval) when the query has no recency intent, no
    chat/messages resolve, or anything on this path raises.
    """
    if not whatsapp_live.recency_conversation_intent(memo_query):
        return None
    try:
        db = whatsapp_live.bridge_db_path()
        contacts_index = build_contacts_index(cfg.contacts_dir) if cfg.contacts_dir else {}
        chats = whatsapp_live.resolve_chats(memo_query, db, contacts_index)
        if not chats:
            return None
        jid, label = chats[0]
        singular = whatsapp_live.singular_last_intent(memo_query)
        today = whatsapp_live.today_only_intent(memo_query)
        limit = 1 if singular else (200 if today else 10)
        msgs = whatsapp_live.last_messages(db, jid, limit=limit, today_only=today and not singular)
        if not msgs:
            return None
        last_date = str(msgs[-1].get("ts", ""))[:10]
        return [
            {
                "id": f"wa-live:{label.lower()}",
                "source": "memory",
                "type": "whatsapp_live",
                "title": f"WhatsApp · {label} — {last_date}",
                "snippet": whatsapp_live.format_transcript(label, msgs),
                "score": 0.99,
                "normalized_score": 0.99,
            }
        ]
    except Exception:
        return None


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
    retrieval_t0 = time.monotonic()
    yield {"type": "stage", "name": "memo_retrieval", "phase": "start"}

    dominant = None
    sources: list[dict[str, Any]] | None = None
    if cfg.whatsapp_live:
        sources = _whatsapp_live_source(cfg, memo_query)

    if sources is None:
        try:
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
        except Exception:
            yield {"type": "error", "message": "retrieval failed", "answer_partial": ""}
            return

    yield {
        "type": "stage",
        "name": "memo_retrieval",
        "phase": "done",
        "ms": int((time.monotonic() - retrieval_t0) * 1000),
    }
    yield {"type": "context", "sources": sources[:base_k]}

    def _done(answer: str, synthesis_source: str) -> dict[str, Any]:
        return {
            "type": "done",
            "answer": answer,
            "sources": sources[:base_k],
            "synthesis_source": synthesis_source,
            "total_ms": int((time.monotonic() - t0) * 1000),
        }

    streaming_t0 = time.monotonic()
    yield {"type": "stage", "name": "streaming", "phase": "start"}

    if dominant:
        doc = resolve_fulldoc(memory, dominant)
        if doc:
            yield {"type": "token", "text": doc["text"]}
            yield {
                "type": "stage",
                "name": "streaming",
                "phase": "done",
                "ms": int((time.monotonic() - streaming_t0) * 1000),
            }
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
    yield {
        "type": "stage",
        "name": "streaming",
        "phase": "done",
        "ms": int((time.monotonic() - streaming_t0) * 1000),
    }
    yield _done("".join(parts), "memo.chat")
