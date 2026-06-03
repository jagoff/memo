"""Question-answering operations for `Memory` — ask / chat_ask and helpers.

`_AskOpsMixin` holds the synthesis methods (`ask`, `ask_stream`, `chat_ask`,
`chat_ask_stream`) and their private context-building / verbatim / citation
helpers, moved verbatim from the former `memory.py` god-file.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

from memo.llm import MLXChat
from memo.memory._base import _MemoryBase
from memo.memory.record import (
    _ASK_SYSTEM_PROMPT,
    MemoryRecord,
    _is_conversation_query,
    _is_group_chat,
    _is_recency_query,
    _is_whatsapp_hit,
    _log,
    _norm_dedup_path,
    _recency_key,
    _vault_dedup_keys,
)


class _AskOpsMixin(_MemoryBase):
    # -- chat ask -----------------------------------------------------------

    def chat_ask(
        self,
        question: str,
        *,
        k: int = 7,
        type_: str | None = None,
        history: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Chat-shaped RAG envelope owned by Memo.

        Synapse may provide federation context and Memflow-backed history, but
        retrieval, citations, and synthesis stay inside Memo.
        """
        started = time.perf_counter()
        clean_history = self._normalize_chat_history(history or [])
        clean_context = context or {}
        retrieval_question = self._chat_retrieval_question(
            question,
            history=clean_history,
            context=clean_context,
        )
        rag = self.ask(retrieval_question, k=k, type_=type_)
        total_ms = int((time.perf_counter() - started) * 1000)
        answer = str(rag.get("answer") or "").strip()
        sources = [item for item in (rag.get("sources") or []) if isinstance(item, dict)]
        synthesis_error = ""
        if not question.strip():
            status = "unavailable"
            synthesis_error = "empty question"
        elif answer.startswith("(error consultando el modelo:"):
            status = "error"
            synthesis_error = answer
        elif not answer:
            status = "error"
            synthesis_error = "empty answer"
        elif not sources:
            status = "unavailable"
            synthesis_error = answer
        else:
            status = "ok"
        context_keys = sorted(str(key) for key in clean_context)
        return {
            "schema": "memo.chat_ask.v2",
            "question": question,
            "answer": answer,
            "sources": sources,
            "citations": self._chat_citations(sources),
            "retrieval_trace": [
                {
                    "stage": "memo.chat_ask",
                    "ms": total_ms,
                    "source_count": len(sources),
                    "history_turns": len(clean_history),
                    "context_keys": context_keys,
                    "retrieval_query_chars": len(retrieval_question),
                }
            ],
            "synthesis_status": status,
            "synthesis_source": f"memo.ask:{self.cfg.llm_model}" if sources else "memo.ask",
            "synthesis_error": synthesis_error,
            "total_ms": total_ms,
            "history_turns_used": len(clean_history),
            "context_keys": context_keys,
        }

    def chat_ask_stream(
        self,
        question: str,
        *,
        k: int = 7,
        type_: str | None = None,
        history: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Streaming chat-shaped RAG envelope.

        Wraps `ask_stream`, re-shaping events into the `memo.chat_ask.v2`
        schema and emitting:

          - {"event":"context", "schema":"memo.chat_ask.v2",
             "sources":[...], "citations":[...]}              once
          - {"event":"token",   "delta":"..."}                 N times
          - {"event":"done",    ...full envelope...}           once
          - on synthesis error: final `done` has
             synthesis_status="error", synthesis_error=<exc>,
             answer=<partial accumulator>
        """
        started = time.perf_counter()
        clean_history = self._normalize_chat_history(history or [])
        clean_context = context or {}
        retrieval_question = self._chat_retrieval_question(
            question,
            history=clean_history,
            context=clean_context,
        )
        context_keys = sorted(str(key) for key in clean_context)

        sources: list[dict[str, Any]] = []
        accum_parts: list[str] = []
        synthesis_error = ""
        had_error = False

        if not question.strip():
            total_ms = int((time.perf_counter() - started) * 1000)
            yield {
                "event": "done",
                "schema": "memo.chat_ask.v2",
                "question": question,
                "answer": "",
                "sources": [],
                "citations": [],
                "retrieval_trace": [{
                    "stage": "memo.chat_ask_stream",
                    "ms": total_ms,
                    "source_count": 0,
                    "history_turns": len(clean_history),
                    "context_keys": context_keys,
                    "retrieval_query_chars": len(retrieval_question),
                }],
                "synthesis_status": "unavailable",
                "synthesis_source": "memo.ask",
                "synthesis_error": "empty question",
                "total_ms": total_ms,
                "history_turns_used": len(clean_history),
                "context_keys": context_keys,
            }
            return

        for ev in self.ask_stream(retrieval_question, k=k, type_=type_):
            kind = ev.get("event")
            if kind == "sources":
                sources = list(ev.get("sources") or [])
                yield {
                    "event": "context",
                    "schema": "memo.chat_ask.v2",
                    "sources": sources,
                    "citations": self._chat_citations(sources),
                }
            elif kind == "token":
                delta = str(ev.get("delta") or "")
                if delta:
                    accum_parts.append(delta)
                    yield {"event": "token", "delta": delta}
            elif kind == "error":
                had_error = True
                synthesis_error = str(ev.get("message") or "synthesis error")
                partial = str(ev.get("answer_partial") or "")
                if partial and not accum_parts:
                    accum_parts.append(partial)
                break
            elif kind == "done":
                # ask_stream's done carries the final accumulated answer and
                # the source list; prefer it as the authoritative state.
                done_answer = str(ev.get("answer") or "")
                done_sources = ev.get("sources")
                if done_answer and not accum_parts:
                    accum_parts.append(done_answer)
                if isinstance(done_sources, list) and not sources:
                    sources = done_sources

        total_ms = int((time.perf_counter() - started) * 1000)
        answer = "".join(accum_parts).strip()
        if had_error:
            status = "error"
        elif answer.startswith("(error consultando el modelo:"):
            status = "error"
            synthesis_error = answer
        elif not answer:
            status = "error"
            synthesis_error = "empty answer"
        elif not sources:
            status = "unavailable"
            synthesis_error = answer
        else:
            status = "ok"

        yield {
            "event": "done",
            "schema": "memo.chat_ask.v2",
            "question": question,
            "answer": answer,
            "sources": sources,
            "citations": self._chat_citations(sources),
            "retrieval_trace": [{
                "stage": "memo.chat_ask_stream",
                "ms": total_ms,
                "source_count": len(sources),
                "history_turns": len(clean_history),
                "context_keys": context_keys,
                "retrieval_query_chars": len(retrieval_question),
            }],
            "synthesis_status": status,
            "synthesis_source": (
                f"memo.ask_stream:{self.cfg.llm_model}" if sources else "memo.ask_stream"
            ),
            "synthesis_error": synthesis_error,
            "total_ms": total_ms,
            "history_turns_used": len(clean_history),
            "context_keys": context_keys,
        }

    @staticmethod
    def _normalize_chat_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for item in history[-8:]:
            role = str(item.get("role") or "").strip().lower()
            text = str(item.get("text") or item.get("content") or "").strip()
            if role not in {"user", "assistant"} or not text:
                continue
            normalized.append({"role": role, "text": text[:1200]})
        return normalized

    @staticmethod
    def _chat_retrieval_question(
        question: str,
        *,
        history: list[dict[str, str]],
        context: dict[str, Any],
    ) -> str:
        parts = [question.strip()]
        if history:
            turns = "\n".join(
                f"{turn['role']}: {turn['text']}" for turn in history[-6:]
            )
            parts.append(f"Conversation history:\n{turns}")
        if context:
            compact = json.dumps(context, ensure_ascii=False, sort_keys=True, default=str)
            parts.append(f"Federation context:\n{compact[:2400]}")
        return "\n\n".join(part for part in parts if part)

    @staticmethod
    def _chat_citations(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        for index, source in enumerate(sources, start=1):
            source_kind = str(source.get("source") or "memo")
            source_id = str(
                source.get("id_short")
                or source.get("locator")
                or source.get("id")
                or index
            )
            metadata: dict[str, Any] = {}
            if source.get("id"):
                metadata["id"] = source.get("id")
            if source.get("path"):
                metadata["path"] = source.get("path")
            if source.get("repo_name"):
                metadata["repo_name"] = source.get("repo_name")
            citations.append(
                {
                    "n": index,
                    "id": source_id,
                    "source": "memo" if source_kind == "memory" else source_kind,
                    "title": str(source.get("title") or source_id),
                    "metadata": metadata,
                }
            )
        return citations

    # -- ask ----------------------------------------------------------------

    def _build_ask_context(
        self, question: str, *, k: int, type_: str | None,
        snippet_chars: int, include_repos: bool, disable_reranker: bool = True,
    ) -> tuple[str, list[dict[str, Any]], str, list[MemoryRecord]]:
        """Retrieval half of ask()/ask_stream().

        Returns (normalized_question, sources, user_msg, hits). When no
        hits found, returns (question, [], "", []) — caller must
        short-circuit. `hits` is the raw `MemoryRecord` list (with `body`
        populated) so callers can run cheap heuristics like verbatim
        short-circuit without re-running search.

        Args:
            disable_reranker: If True (default for chat), skip cross-encoder
                reranking. RRF is sufficient for synthesis and reranker adds
                ~150ms latency.
        """
        if not question or not question.strip():
            return question, [], "", []
        _MAX_QUESTION_CHARS = 4000
        if len(question) > _MAX_QUESTION_CHARS:
            _log.warning(
                "ask: question truncated from %d to %d chars",
                len(question), _MAX_QUESTION_CHARS,
            )
            question = question[:_MAX_QUESTION_CHARS]
        # Recency intent ("lo último que dijo X", "last message") and the wider
        # conversation intent ("mostrame el chat con X", "qué me escribió X")
        # both widen the pool so the dated transcript isn't crowded out by a
        # same-named contact/profile card. Only recency biases retrieval toward
        # *recent* material (freshness decay) — a conversation ask shouldn't
        # down-weight by age.
        recency_intent = _is_recency_query(question)
        convo_intent = recency_intent or _is_conversation_query(question)
        search_limit = max(k, 12) if convo_intent else k
        # Lazy-load bodies: defer disk I/O until after reranking
        hits: list[MemoryRecord] = self.search(
            question, limit=search_limit, type_=type_, mode="hybrid", load_bodies=False,
            disable_reranker=disable_reranker, read_through=True, recency=recency_intent,
        )
        repo_hits = []
        if include_repos and self.store.list_repo_sources(limit=1):
            with contextlib.suppress(Exception):
                repo_hits = self.repo_search(question, limit=k, mode="hybrid")
        if not hits and not repo_hits:
            return question, [], "", []

        # Load bodies only for the final hits that will be used.
        # MemoryRecord is frozen, so rebuild rather than mutate in place.
        hits = [
            h if h.body else replace(h, body=self._read_body(h.path))
            for h in hits
        ]

        # Recency/conversation re-ranking: float WhatsApp transcripts above a
        # same-named contact/profile card, preferring a 1:1 chat over a same-
        # named group. For a recency ask, the dated content the hit carries is
        # the tiebreaker (newest first); for a conversation-only ask the date
        # is left out so semantic relevance is preserved among ties (the sort is
        # stable). Gated on `has_wa` for conversation intent so a non-WhatsApp
        # conversational query isn't destructively re-sorted. Bodies are loaded
        # above, so _recency_key can read in-transcript dates.
        has_wa = any(_is_whatsapp_hit(h) for h in hits)
        if recency_intent or (convo_intent and has_wa):
            ql = question.lower()
            # Prefer a 1:1 chat over a same-named group ONLY as a same-date
            # tiebreaker (below), never over a genuinely newer conversation.
            prefer_direct = "group" not in ql and "grupo" not in ql

            def _recency_sort_key(h: MemoryRecord) -> tuple[int, str, int]:
                wa = _is_whatsapp_hit(h)
                direct = 1 if (prefer_direct and wa and not _is_group_chat(h)) else 0
                # 1) float ALL transcripts above non-transcripts (a same-named
                #    contact card sinks regardless of its fresh `updated` stamp);
                # 2) for a recency ask, newest dated content wins — so a group
                #    active today beats an older 1:1; 3) prefer a 1:1 only to
                #    break a date tie. Conversation-only asks leave date="" so
                #    the 1:1 preference leads.
                date = _recency_key(h) if recency_intent else ""
                return (1 if wa else 0, date, direct)

            hits = sorted(hits, key=_recency_sort_key, reverse=True)[:k]

        snippet_lines: list[str] = []
        sources: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for h in hits:
            id_short = h.id[:8]
            snippet = (h.body or "")[:snippet_chars]
            if len(h.body or "") > snippet_chars:
                snippet = snippet.rstrip() + "…"
            tags = ", ".join(h.tags) or "—"
            snippet_lines.append(
                f"[{id_short}] title: {h.title}  |  type: {h.type}  |  tags: {tags}\n"
                f"{snippet}\n"
            )
            sources.append({
                "source": "memory",
                "id": h.id,
                "id_short": id_short,
                "title": h.title,
                "type": h.type,
                "score": h.score,
                "snippet": snippet,
            })
            seen_paths.update(_vault_dedup_keys(h))
        seen_repo_keys: set[tuple[str, str]] = set()
        for h in repo_hits:
            norm = _norm_dedup_path(h.path)
            base = norm.rsplit("/", 1)[-1] if norm else ""
            # Skip if same file already surfaced as a vault memoria.
            if norm and (norm in seen_paths or base in seen_paths):
                continue
            # Dedup intra-repo: keep only the first (highest-score) chunk per file.
            repo_key = (h.repo_name, norm)
            if repo_key in seen_repo_keys:
                continue
            seen_repo_keys.add(repo_key)
            label = h.locator
            snippet = (h.text or "")[:snippet_chars]
            if len(h.text or "") > snippet_chars:
                snippet = snippet.rstrip() + "…"
            snippet_lines.append(
                f"[{label}] source: repo  |  path: {h.path}  |  "
                f"lines: {h.line_start}-{h.line_end}  |  match: {h.match_type}\n"
                f"{snippet}\n"
            )
            sources.append({
                "source": "repo",
                "id": h.id,
                "id_short": label,
                "title": h.path,
                "type": "repo",
                "score": h.score,
                "snippet": snippet,
                "repo_name": h.repo_name,
                "path": h.path,
                "line_start": h.line_start,
                "line_end": h.line_end,
                "locator": label,
            })

        user_msg = (
            f"Pregunta del user:\n{question}\n\n"
            f"Contexto relevante ({len(hits)} memorias, {len(repo_hits)} snippets de repo):\n\n"
            + "\n---\n".join(snippet_lines)
        )
        return question, sources, user_msg, hits

    def _verbatim_short_circuit(
        self, question: str, hits: list[MemoryRecord],
    ) -> str | None:
        """If the query is a literal phrase lookup (short, no `?`) and the
        text appears inside the top hit's body, return that body verbatim
        instead of calling the LLM.

        The LLM tends to "helpfully" condense — for `letra`, `comando`,
        `snippet`, `CBU`, `URL` style lookups the user wants the WHOLE
        note dumped, not a 2-sentence summary. Returning early dodges
        token spend and avoids the model second-guessing the user.
        """
        if not hits:
            return None
        q = (question or "").strip()
        if not q:
            return None
        # Question form → defer to synthesis.
        if "?" in q or "¿" in q:
            return None
        tokens = [t for t in q.split() if t]
        if len(tokens) > 12:
            return None
        top = hits[0]
        body = (top.body or "").strip()
        if not body or len(body) <= len(q):
            return None
        if q.lower() not in body.lower():
            return None
        return f"{body}\n\n[{top.id[:8]}]"

    def ask(
        self, question: str, *, k: int = 5, type_: str | None = None,
        snippet_chars: int = 2000, include_repos: bool = True,
    ) -> dict[str, Any]:
        """Synthesised Q&A over the memory archive (RAG).

        Pipeline: hybrid search top-`k` → format snippets with `[id]`
        citations → MLXChat 7B (`MEMO_LLM_MODEL`) generates a prose
        answer with inline citations. Returns:

            {
                "question": str,
                "answer": str,            # may say "no encuentro ..."
                "sources": [               # the snippets the LLM saw
                    {id, title, type, score, snippet}, ...
                ],
            }

        Latency: ~3-8s on a cold 7B load + ~1-2s decode for short
        answers. For token-by-token output use `ask_stream`. Use
        `search` if you only need IDs to scan manually.
        """
        if not question or not question.strip():
            return {"question": question, "answer": "", "sources": []}
        norm_question, sources, user_msg, hits = self._build_ask_context(
            question, k=k, type_=type_,
            snippet_chars=snippet_chars, include_repos=include_repos,
        )
        if not sources:
            from memo.flags import flag_str
            return {
                "question": norm_question,
                "answer": flag_str("MEMO_ASK_FALLBACK_MSG"),
                "sources": [],
            }

        # Verbatim short-circuit: literal phrase lookups bypass the LLM
        # and return the matched note body directly. Avoids the model
        # over-summarising when the user clearly wants the raw content.
        verbatim = self._verbatim_short_circuit(question, hits)
        if verbatim is not None:
            return {
                "question": norm_question,
                "answer": verbatim,
                "sources": sources,
            }

        # Lazy-construct the chat client (same instance used by auto_derive).
        if self._chat is None:
            self._chat = MLXChat()
        try:
            out = self._chat.chat(
                model=self.cfg.llm_model,
                messages=[
                    {"role": "system", "content": _ASK_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                # Higher max_tokens than auto_derive — answers can run a
                # paragraph or two.
                options={"temperature": 0.0, "max_tokens": 768},
            )
            answer = ((out.get("message") or {}).get("content") or "").strip()
        except Exception as exc:
            answer = f"(error consultando el modelo: {type(exc).__name__})"

        return {
            "question": norm_question,
            "answer": answer,
            "sources": sources,
        }

    def ask_stream(
        self, question: str, *, k: int = 5, type_: str | None = None,
        snippet_chars: int = 2000, include_repos: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """Streaming variant of `ask()` — yields token-level events.

        Event protocol (NDJSON-compatible dicts):
          - {"event": "sources", "sources": [...]}            once, after retrieval
          - {"event": "token",   "delta": "<chunk>"}          one per LLM yield
          - {"event": "done",    "answer": "<acc>", "sources": [...]}
          - {"event": "error",   "message": "...", "answer_partial": "..."}

        Empty-question / no-hits paths short-circuit with a single `done`
        carrying the same refusal text as `ask()`.
        """
        if not question or not question.strip():
            yield {"event": "done", "answer": "", "sources": []}
            return
        _, sources, user_msg, hits = self._build_ask_context(
            question, k=k, type_=type_,
            snippet_chars=snippet_chars, include_repos=include_repos,
        )
        if not sources:
            yield {
                "event": "done",
                "answer": "no encuentro la respuesta en las memorias guardadas",
                "sources": [],
            }
            return

        yield {"event": "sources", "sources": sources}

        # Verbatim short-circuit (literal-phrase queries): emit body as a
        # single token-style event so consumers that show progressive
        # output still get something to render, then a terminal `done`.
        verbatim = self._verbatim_short_circuit(question, hits)
        if verbatim is not None:
            yield {"event": "token", "delta": verbatim}
            yield {"event": "done", "answer": verbatim, "sources": sources}
            return

        if self._chat is None:
            self._chat = MLXChat()
        accum_parts: list[str] = []
        try:
            for delta in self._chat.chat_stream(
                model=self.cfg.llm_model,
                messages=[
                    {"role": "system", "content": _ASK_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                options={"temperature": 0.0, "max_tokens": 768},
            ):
                accum_parts.append(delta)
                yield {"event": "token", "delta": delta}
        except Exception as exc:
            yield {
                "event": "error",
                "message": f"{type(exc).__name__}: {exc}",
                "answer_partial": "".join(accum_parts).strip(),
            }
            return

        yield {
            "event": "done",
            "answer": "".join(accum_parts).strip(),
            "sources": sources,
        }

