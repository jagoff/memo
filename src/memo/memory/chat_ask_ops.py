"""Conversational QA for `Memory` — chat_ask / chat_ask_stream + helpers.

`_ChatAskOpsMixin` holds the multi-turn chat surface (`chat_ask`,
`chat_ask_stream`) and its private history-normalisation, retrieval-question,
and citation helpers. Split out of `ask_ops.py` to keep both files under the
repo's 800-line limit; `Memory` inherits this mixin alongside `_AskOpsMixin`,
so every cross-helper `self.*` call resolves unchanged via MRO.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

from memo.memory._base import _MemoryBase


class _ChatAskOpsMixin(_MemoryBase):
    def chat_ask(
        self,
        question: str,
        *,
        k: int = 7,
        type_: str | None = None,
        history: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
        snippet_chars: int | None = None,
    ) -> dict[str, Any]:
        """Chat-shaped RAG envelope owned by Memo.

        Synapse may provide federation context and Memflow-backed history, but
        retrieval, citations, and synthesis stay inside Memo.
        """
        started = time.perf_counter()
        clean_history = self._normalize_chat_history(history or [])
        clean_context = context or {}
        resolved_snippet_chars = 800 if snippet_chars is None else snippet_chars
        if resolved_snippet_chars == 800:
            from memo.flags import flag_int

            resolved_snippet_chars = flag_int("MEMO_ASK_SNIPPET_CHARS") or 800
        retrieval_question = self._chat_retrieval_question(
            question,
            history=clean_history,
            context=clean_context,
        )
        # Session-scoped RAG context caching: a session_id in the chat context
        # lets repeated asks in the same thread reuse retrieval (see
        # _build_ask_context / RagContextCache).
        session_id = clean_context.get("session_id") if isinstance(clean_context, dict) else None
        rag = self.ask(
            retrieval_question,
            k=k,
            type_=type_,
            snippet_chars=resolved_snippet_chars,
            intent_text=question,
            session_id=session_id if isinstance(session_id, str) else None,
        )
        total_ms = int((time.perf_counter() - started) * 1000)
        answer = str(rag.get("answer") or "").strip()
        sources = [item for item in (rag.get("sources") or []) if isinstance(item, dict)]
        synthesis_error = ""
        if not question.strip():
            status = "unavailable"
            synthesis_error = "empty question"
        elif answer.startswith("(error querying the model:"):
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
        snippet_chars: int | None = None,
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
        resolved_snippet_chars = 800 if snippet_chars is None else snippet_chars
        if resolved_snippet_chars == 800:
            from memo.flags import flag_int

            resolved_snippet_chars = flag_int("MEMO_ASK_SNIPPET_CHARS") or 800
        retrieval_question = self._chat_retrieval_question(
            question,
            history=clean_history,
            context=clean_context,
        )
        context_keys = sorted(str(key) for key in clean_context)
        # Session-scoped RAG context caching: a session_id in the chat context
        # lets repeated streaming asks in the same thread reuse retrieval
        # (mirrors chat_ask / _build_ask_context).
        session_id = clean_context.get("session_id") if isinstance(clean_context, dict) else None

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
                "retrieval_trace": [
                    {
                        "stage": "memo.chat_ask_stream",
                        "ms": total_ms,
                        "source_count": 0,
                        "history_turns": len(clean_history),
                        "context_keys": context_keys,
                        "retrieval_query_chars": len(retrieval_question),
                    }
                ],
                "synthesis_status": "unavailable",
                "synthesis_source": "memo.ask",
                "synthesis_error": "empty question",
                "total_ms": total_ms,
                "history_turns_used": len(clean_history),
                "context_keys": context_keys,
            }
            return

        for ev in self.ask_stream(
            retrieval_question,
            k=k,
            type_=type_,
            snippet_chars=resolved_snippet_chars,
            intent_text=question,
            session_id=session_id if isinstance(session_id, str) else None,
        ):
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
        elif answer.startswith("(error querying the model:"):
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
            "retrieval_trace": [
                {
                    "stage": "memo.chat_ask_stream",
                    "ms": total_ms,
                    "source_count": len(sources),
                    "history_turns": len(clean_history),
                    "context_keys": context_keys,
                    "retrieval_query_chars": len(retrieval_question),
                }
            ],
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
            turns = "\n".join(f"{turn['role']}: {turn['text']}" for turn in history[-6:])
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
                source.get("id_short") or source.get("locator") or source.get("id") or index
            )
            metadata: dict[str, Any] = {}
            if source.get("id"):
                metadata["id"] = source.get("id")
            if source.get("path"):
                metadata["path"] = source.get("path")
            if source.get("repo_name"):
                metadata["repo_name"] = source.get("repo_name")
            # Provenance trail — surfaced so the consumer can trace which agent
            # session / route produced the memory that fed this answer.
            for prov_key in ("synapse_trace_id", "synapse_agent_id"):
                val = source.get(prov_key)
                if val:
                    metadata[prov_key] = val
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
