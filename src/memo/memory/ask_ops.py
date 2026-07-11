"""Question-answering operations for `Memory` — ask / chat_ask and helpers.

`_AskOpsMixin` holds the synthesis methods (`ask`, `ask_stream`, `chat_ask`,
`chat_ask_stream`) and their private context-building / verbatim / citation
helpers, moved verbatim from the former `memory.py` god-file.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

from memo.grounding_judge import score_grounding
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
    record_from_row,
)
from memo.prompt_overrides import resolve_prompt

# Max provenance memories pulled into the ask context per _build_ask_context
# call when MEMO_ASK_EXPAND_SYNTHESIS is on (bounds disk reads + tokens).
_EXPAND_SOURCES_MAX = 4


def _context_budget_chars(*, snippet_chars: int, k: int) -> int:
    return max(snippet_chars * max(k, 1) + 1200, 2000)


def _format_expanded_section(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = ["Expanded source memories:"]
    for row in rows:
        context_note = str(row.get("context_note") or "")
        context_text = f"  |  context: {context_note}" if context_note else ""
        lines.append(
            f"[{row['id_short']}] title: {row['title']} | type: {row['type']} | "
            f"quality: {row['quality_bucket']}{context_text}\n{row['snippet']}"
        )
    return "\n\n".join(lines)


def _format_repo_section(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = ["Repository snippets:"]
    for row in rows:
        lines.append(
            f"[{row['id_short']}] source: repo | path: {row['path']} | "
            f"lines: {row['line_start']}-{row['line_end']} | match: {row['match_type']}\n"
            f"{row['snippet']}"
        )
    return "\n\n".join(lines)


def _trimmed_item_note(count: int, singular: str, plural: str | None = None) -> str:
    if count <= 0:
        return ""
    noun = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {noun} trimmed by budget"


def _shortened_snippet(snippet: str, *, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(snippet) <= max_chars:
        return snippet
    if max_chars <= 3:
        return "." * max_chars
    return snippet[: max_chars - 3].rstrip() + "..."


def _render_context_pack_sections(
    pack: Any,
    *,
    current: list[dict[str, Any]],
    supporting: list[dict[str, Any]],
    stale: list[dict[str, Any]],
    expanded: list[dict[str, Any]],
    repo: list[dict[str, Any]],
    extra_omissions: list[str],
) -> str:
    from memo.context_pack import ContextPack

    omission_parts = [str(pack.omissions).strip(), *[part for part in extra_omissions if part]]
    omissions = "; ".join(part for part in omission_parts if part)
    prompt_pack = ContextPack(
        question=pack.question,
        summary=pack.summary,
        current_facts=current,
        supporting_context=supporting,
        stale_or_conflicting=stale,
        omissions=omissions,
    )
    sections = [
        prompt_pack.to_prompt(),
        _format_expanded_section(expanded),
        _format_repo_section(repo),
    ]
    return "\n\n".join(section for section in sections if section.strip())


def _fit_context_pack_prompt(
    pack: Any,
    *,
    expanded_rows: list[dict[str, Any]],
    repo_rows: list[dict[str, Any]],
    budget_chars: int,
    expanded_sensitive_omitted: int = 0,
) -> tuple[
    str,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    current = [dict(row) for row in pack.current_facts]
    supporting = [dict(row) for row in pack.supporting_context]
    stale = [dict(row) for row in pack.stale_or_conflicting]
    expanded = [dict(row) for row in expanded_rows]
    repo = [dict(row) for row in repo_rows]
    trimmed_counts = {"supporting": 0, "stale": 0, "expanded": 0, "repo": 0}

    def _omission_notes() -> list[str]:
        notes: list[str] = []
        if expanded_sensitive_omitted:
            noun = (
                "expanded source memory"
                if expanded_sensitive_omitted == 1
                else "expanded source memories"
            )
            notes.append(f"{expanded_sensitive_omitted} sensitive {noun} omitted from compacted context")
        notes.extend(
            note
            for note in (
                _trimmed_item_note(trimmed_counts["supporting"], "supporting memory"),
                _trimmed_item_note(trimmed_counts["stale"], "stale/conflicting memory"),
                _trimmed_item_note(trimmed_counts["expanded"], "expanded source memory"),
                _trimmed_item_note(trimmed_counts["repo"], "repo snippet"),
            )
            if note
        )
        return notes

    prompt = _render_context_pack_sections(
        pack,
        current=current,
        supporting=supporting,
        stale=stale,
        expanded=expanded,
        repo=repo,
        extra_omissions=_omission_notes(),
    )
    if budget_chars <= 0:
        return prompt, current, supporting, stale, expanded, repo

    while len(prompt) > budget_chars:
        previous_prompt: str | None = None
        if supporting:
            supporting.pop()
            trimmed_counts["supporting"] += 1
        elif stale:
            stale.pop()
            trimmed_counts["stale"] += 1
        elif expanded:
            expanded.pop()
            trimmed_counts["expanded"] += 1
        elif repo:
            repo.pop()
            trimmed_counts["repo"] += 1
        elif len(current) > 1:
            current.pop()
        elif current:
            previous_prompt = prompt
            overflow = len(prompt) - budget_chars
            trimmed = dict(current[-1])
            trimmed["snippet"] = _shortened_snippet(
                str(trimmed.get("snippet") or ""),
                max_chars=max(len(str(trimmed.get("snippet") or "")) - overflow, 0),
            )
            current[-1] = trimmed
        else:
            break
        prompt = _render_context_pack_sections(
            pack,
            current=current,
            supporting=supporting,
            stale=stale,
            expanded=expanded,
            repo=repo,
            extra_omissions=_omission_notes(),
        )
        if current and previous_prompt is not None and prompt == previous_prompt:
            current.pop()
            prompt = _render_context_pack_sections(
                pack,
                current=current,
                supporting=supporting,
                stale=stale,
                expanded=expanded,
                repo=repo,
                extra_omissions=_omission_notes(),
            )

    if len(prompt) > budget_chars:
        prompt = prompt[:budget_chars].rstrip()
    return prompt, current, supporting, stale, expanded, repo


def _filter_verbatim_hits(
    hits: list[MemoryRecord],
    sources: list[dict[str, Any]],
    *,
    use_context_pack: bool,
) -> list[MemoryRecord]:
    if not use_context_pack:
        return hits
    allowed_ids = {
        str(source.get("id") or "")
        for source in sources
        if source.get("source") == "memory"
    }
    if not allowed_ids:
        return []
    return [hit for hit in hits if hit.id in allowed_ids]


class _AskOpsMixin(_MemoryBase):
    # -- chat ask -----------------------------------------------------------

    _MULTI_ROUND_SYS = (
        "You check whether retrieved snippets suffice to answer a question. "
        'Reply with ONE line of JSON: {"sufficient": true} or '
        '{"sufficient": false, "queries": ["...", "..."]} — 1-3 refined search '
        "queries, each under 12 words. No other text."
    )

    def _multi_round_augment(
        self,
        question: str,
        hits: list[MemoryRecord],
        *,
        k: int,
        type_: str | None,
    ) -> list[MemoryRecord]:
        """One bounded extra retrieval round (MEMO_ASK_MULTI_ROUND, default off).

        LLM inspects round-1 snippets; if insufficient it emits 1-3 refined
        queries, each searched once; union capped at `k` NEW hits. ask/chat
        path only — never the 5s recall hook. Never raises (degrades to
        round-1 hits on any failure/timeout)."""
        import json as _json

        from memo.memory.record import chat_with_timeout

        try:
            snippets = "\n".join(
                f"[{h.id[:8]}] {h.title}: {(h.body or '')[:200]}" for h in hits[:8]
            )
            out = chat_with_timeout(
                self._ensure_chat(),
                timeout=10.0,
                model=self.cfg.llm_model,
                messages=[
                    {"role": "system", "content": self._MULTI_ROUND_SYS},
                    {"role": "user", "content": f"Question:\n{question}\n\nSnippets:\n{snippets}"},
                ],
                options={"temperature": 0.0, "max_tokens": 120},
            )
            raw = ((out or {}).get("message") or {}).get("content") or ""
            verdict = _json.loads(raw.strip().splitlines()[-1])
            if verdict.get("sufficient", True):
                return hits
            queries = [str(q) for q in (verdict.get("queries") or [])[:3] if str(q).strip()]
            seen = {h.id for h in hits}
            added: list[MemoryRecord] = []
            for q in queries:
                for h in self.search(
                    q,
                    limit=k,
                    type_=type_,
                    mode="hybrid",
                    disable_reranker=True,
                    quality_rerank=True,
                ):
                    if h.id in seen or len(added) >= k:
                        continue
                    seen.add(h.id)
                    added.append(h if h.body else replace(h, body=self._read_body(h.path)))
            if added:
                _log.info("ask multi-round: +%d hits via %d refined queries", len(added), len(queries))
            return [*hits, *added]
        except Exception as exc:
            _log.debug("ask multi-round skipped: %s", exc)
            return hits

    def _build_ask_context(
        self,
        question: str,
        *,
        k: int,
        type_: str | None,
        snippet_chars: int,
        include_repos: bool,
        disable_reranker: bool = True,
        intent_text: str | None = None,
        session_id: str | None = None,
        use_context_pack: bool = False,
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
        # Session-scoped RAG context cache. Opt-in: only engages when the
        # caller supplies a session_id (chat threads pass one), so the default
        # path is byte-for-byte unchanged. Invalidated by corpus_version +
        # TTL inside the cache. See rag_cache.RagContextCache.
        cache_key = None
        if session_id:
            cache_key = "|".join(
                str(p)
                for p in (
                    session_id,
                    k,
                    type_,
                    snippet_chars,
                    include_repos,
                    disable_reranker,
                    use_context_pack,
                    intent_text or "",
                    question.strip(),
                )
            )
            cache = self._get_rag_cache()
            hit = cache.get(cache_key, corpus_version=self._corpus_version(), now=time.time())
            if hit is not None:
                norm_q, cached_sources, user_msg, cached_hits = hit
                # Return copies of the list containers so a caller mutating its
                # result can't corrupt the cached entry (elements are treated
                # read-only by ask()/ask_stream()).
                return norm_q, list(cached_sources), user_msg, list(cached_hits)
        _MAX_QUESTION_CHARS = 4000
        if len(question) > _MAX_QUESTION_CHARS:
            _log.warning(
                "ask: question truncated from %d to %d chars",
                len(question),
                _MAX_QUESTION_CHARS,
            )
            question = question[:_MAX_QUESTION_CHARS]
        # Recency intent ("lo último que dijo X", "last message") and the wider
        # conversation intent ("mostrame el chat con X", "qué me escribió X")
        # both widen the pool so the dated transcript isn't crowded out by a
        # same-named contact/profile card. Only recency biases retrieval toward
        # *recent* material (freshness decay) — a conversation ask shouldn't
        # down-weight by age.
        # Detect intent on the original user question too, not only the
        # (possibly rewritten / context-wrapped) retrieval text. Synapse sends
        # memo a cleaned `retrieval_question` that can drop the recency token,
        # which would silently disable the recency path; `intent_text` carries
        # the raw question so the signal survives.
        intent_src = f"{question} {intent_text or ''}"
        recency_intent = _is_recency_query(intent_src)
        convo_intent = recency_intent or _is_conversation_query(intent_src)
        # Recency asks re-sort the pool by in-transcript date (below), but the
        # sort can only float candidates that made it into the pool. The newest
        # message of a chat is often semantically bland ("cómo te fue hoy?") and
        # scores low, so a tight pool drops it and the recency sort surfaces a
        # stale-but-relevant chunk instead. Widen the pool for recency so the
        # freshest dated chunk reliably enters before the re-sort.
        if recency_intent:
            search_limit = max(k, 60)
        elif convo_intent:
            search_limit = max(k, 12)
        else:
            search_limit = k
        # Lazy-load bodies: defer disk I/O until after reranking
        hits: list[MemoryRecord] = self.search(
            question,
            limit=search_limit,
            type_=type_,
            mode="hybrid",
            load_bodies=False,
            disable_reranker=disable_reranker,
            read_through=True,
            recency=recency_intent,
            quality_rerank=True,
        )
        repo_hits = []
        if include_repos and self.store.list_repo_sources(limit=1):
            with contextlib.suppress(Exception):
                repo_hits = self.repo_search(question, limit=k, mode="hybrid")
        if not hits and not repo_hits:
            return question, [], "", []

        # Load bodies only for the final hits that will be used.
        # MemoryRecord is frozen, so rebuild rather than mutate in place.
        hits = [h if h.body else replace(h, body=self._read_body(h.path)) for h in hits]

        from memo.flags import flag_bool

        if flag_bool("MEMO_ASK_MULTI_ROUND") and hits:
            hits = self._multi_round_augment(question, hits, k=k, type_=type_)

        # Recency augmentation: the newest chunk of a long transcript is often
        # semantically bland ("cómo te fue hoy?") and never makes the candidate
        # pool on cosine, so the recency re-sort below can't surface it. For a
        # recency ask, pull the latest chunks of every transcript already
        # retrieved (same parent note) straight from metadata and fold them in,
        # so the genuine last message can win the sort.
        if recency_intent:
            seen_ids = {h.id for h in hits}
            parents = {
                p for h in hits if _is_whatsapp_hit(h) and (p := (h.extra or {}).get("parent_path"))
            }
            for parent in parents:
                for row in self.store.chunks_by_parent(parent, limit=3):
                    if row["id"] in seen_ids:
                        continue
                    seen_ids.add(row["id"])
                    hits.append(record_from_row(row, body=self._read_body(row["path"])))

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
        primary_memory_sources: dict[str, dict[str, Any]] = {}
        seen_paths: set[str] = set()
        for h in hits:
            id_short = h.id[:8]
            snippet = (h.body or "")[:snippet_chars]
            if len(h.body or "") > snippet_chars:
                snippet = snippet.rstrip() + "…"
            tags = ", ".join(h.tags) or "—"
            graph_info = "  |  context: related-via-graph" if h.extra.get("graph_expanded") else ""
            related_facts = [
                f"{f.get('subject')} {f.get('predicate')} {f.get('object')}"
                for f in (h.extra or {}).get("related_fact_edges", [])
                if isinstance(f, dict)
                and f.get("subject")
                and f.get("predicate")
                and f.get("object")
            ][:3]
            facts_info = f"  |  facts: {'; '.join(related_facts)}" if related_facts else ""
            snippet_lines.append(
                f"[{id_short}] title: {h.title}  |  type: {h.type}  |  tags: {tags}{graph_info}{facts_info}\n{snippet}\n"
            )
            extra = h.extra or {}
            source = {
                "source": "memory",
                "id": h.id,
                "id_short": id_short,
                "title": h.title,
                "type": h.type,
                "score": h.score,
                "snippet": snippet,
                "graph_expanded": bool(extra.get("graph_expanded")),
                "related_fact_edges": extra.get("related_fact_edges") or [],
                "synapse_trace_id": extra.get("synapse_trace_id") or "",
                "synapse_agent_id": extra.get("synapse_agent_id") or "",
            }
            sources.append(source)
            primary_memory_sources[h.id] = source
            seen_paths.update(_vault_dedup_keys(h))
        # Lazy synthesis_sources expansion (MEMO_ASK_EXPAND_SYNTHESIS, default
        # off): a synthesis hit is an ABSTRACT — for holistic asks the concrete
        # answer often lives in its source memories. Pull a bounded number of
        # provenance memories into the context (store fetches only, NO LLM
        # call; ask path only — never the 5s recall hook). Community-kind
        # syntheses keep entity NAMES in synthesis_sources; non-resolvable
        # strings skip via self.get() returning None.
        from memo.flags import flag_bool

        expanded_memory_rows: list[dict[str, Any]] = []
        expanded_memory_sources: dict[str, dict[str, Any]] = {}
        expanded_sensitive_omitted = 0
        if flag_bool("MEMO_ASK_EXPAND_SYNTHESIS"):
            from memo.context_pack import build_context_row

            _seen_ids = {h.id for h in hits}
            _expanded = 0
            for h in hits:
                if h.type != "synthesis" or _expanded >= _EXPAND_SOURCES_MAX:
                    continue
                _hx = h.extra or {}
                _src_ids = (
                    _hx.get("synthesis_source_memories") or _hx.get("synthesis_sources") or []
                )
                if not isinstance(_src_ids, list):
                    continue
                for sid in _src_ids:
                    if _expanded >= _EXPAND_SOURCES_MAX:
                        break
                    if not isinstance(sid, str) or sid in _seen_ids:
                        continue
                    src = self.get(sid)
                    if src is None or src.id in _seen_ids:
                        continue
                    row = build_context_row(src, snippet_chars=snippet_chars)
                    if row is None:
                        _seen_ids.add(src.id)
                        _seen_ids.add(sid)
                        expanded_sensitive_omitted += 1
                        continue
                    _seen_ids.add(src.id)
                    _seen_ids.add(sid)
                    _expanded += 1
                    _snip = row["snippet"]
                    expanded_line = (
                        f"[{src.id[:8]}] title: {src.title}  |  type: {src.type}"
                        f"  |  context: source-of [{h.id[:8]}]\n{_snip}\n"
                    )
                    if use_context_pack:
                        row["context_note"] = f"source-of [{h.id[:8]}]"
                        expanded_memory_rows.append(row)
                        expanded_memory_sources[src.id] = {
                            "source": "memory",
                            "id": src.id,
                            "id_short": src.id[:8],
                            "title": src.title,
                            "type": src.type,
                            "score": None,
                            "snippet": row["snippet"],
                            "expanded_from": h.id,
                            "quality_bucket": row["quality_bucket"],
                            "quality_reasons": list(row["quality_reasons"]),
                        }
                    else:
                        snippet_lines.append(expanded_line)
                        sources.append(
                            {
                                "source": "memory",
                                "id": src.id,
                                "id_short": src.id[:8],
                                "title": src.title,
                                "type": src.type,
                                "score": None,
                                "snippet": _snip,
                                "expanded_from": h.id,
                            }
                        )
        seen_repo_keys: set[tuple[str, str]] = set()
        repo_rows: list[dict[str, Any]] = []
        repo_sources: list[dict[str, Any]] = []
        for h in repo_hits:
            norm = _norm_dedup_path(h.path)
            base = norm.rsplit("/", 1)[-1] if norm else ""
            # Skip if same file already surfaced as a vault memory.
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
            repo_row = {
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
                "match_type": h.match_type,
            }
            repo_rows.append(repo_row)
            repo_sources.append(repo_row)
            if not use_context_pack:
                repo_line = (
                    f"[{label}] source: repo  |  path: {h.path}  |  "
                    f"lines: {h.line_start}-{h.line_end}  |  match: {h.match_type}\n"
                    f"{snippet}\n"
                )
                snippet_lines.append(repo_line)
                sources.append(dict(repo_row))

        if use_context_pack:
            from memo.context_pack import build_context_pack

            pack = build_context_pack(
                question,
                hits,
                snippet_chars=snippet_chars,
                budget_chars=0,
            )
            context_header = (
                f"Relevant context pack ({len(hits)} memories, {len(repo_rows)} repo snippets):\n\n"
            )
            budget_chars = max(
                _context_budget_chars(snippet_chars=snippet_chars, k=k) - len(context_header),
                0,
            )
            context_prompt, current_rows, supporting_rows, stale_rows, kept_expanded_rows, kept_repo_rows = (
                _fit_context_pack_prompt(
                    pack,
                    expanded_rows=expanded_memory_rows,
                    repo_rows=repo_rows,
                    budget_chars=budget_chars,
                    expanded_sensitive_omitted=expanded_sensitive_omitted,
                )
            )
            user_msg = (
                f"User question:\n{question}\n\n"
                f"{context_header}"
                f"{context_prompt}"
            )
            sources = []
            for row in [*current_rows, *supporting_rows, *stale_rows]:
                source_data = primary_memory_sources.get(str(row["id"]) or "")
                if not source_data:
                    continue
                source = dict(source_data)
                source["snippet"] = row["snippet"]
                source["quality_bucket"] = row["quality_bucket"]
                source["quality_reasons"] = list(row["quality_reasons"])
                sources.append(source)
            for row in kept_expanded_rows:
                source_data = expanded_memory_sources.get(str(row["id"]) or "")
                if not source_data:
                    continue
                source = dict(source_data)
                source["snippet"] = row["snippet"]
                sources.append(source)
            kept_repo_ids = {str(row["id"]) for row in kept_repo_rows}
            sources.extend(dict(source) for source in repo_sources if str(source["id"]) in kept_repo_ids)
        else:
            user_msg = (
                f"User question:\n{question}\n\n"
                f"Relevant context ({len(hits)} memories, {len(repo_rows)} repo snippets):\n\n"
                + "\n---\n".join(snippet_lines)
            )
        if cache_key is not None and sources:
            # Only cache non-empty retrievals — an empty result is cheap to
            # recompute and may reflect a transient cold embedder.
            self._get_rag_cache().put(
                cache_key,
                (question, sources, user_msg, hits),
                corpus_version=self._corpus_version(),
                now=time.time(),
            )
        return question, sources, user_msg, hits

    def _get_rag_cache(self) -> Any:
        """Lazily build the process-local RAG context cache (TTL via
        MEMO_RAG_CACHE_TTL_S, default 300s)."""
        cache = getattr(self, "_rag_cache", None)
        if cache is None:
            from memo.flags import flag_int
            from memo.rag_cache import RagContextCache

            cache = RagContextCache(ttl_s=float(flag_int("MEMO_RAG_CACHE_TTL_S") or 300))
            self._rag_cache = cache
        return cache

    def _corpus_version(self) -> str:
        """Cheap corpus fingerprint: row count + latest update timestamp.
        Any save/update/delete moves it, invalidating cached retrievals.
        Includes repo source state so repo index/embed/delete busts the cache.
        Memoized per-instance; auto-invalidated by the write generation
        counter (bumped on save/update/delete)."""
        gen = getattr(self, "_write_gen", 0)
        cached: tuple[int, str] | None = getattr(self, "_corpus_version_cached", None)
        if cached is not None and cached[0] == gen:
            return cached[1]
        try:
            meta = self.store._conn.execute(
                "SELECT COUNT(*), COALESCE(MAX(updated), '') FROM meta"
            ).fetchone()
            ver = f"m{meta[0]}:{meta[1]}"
        except Exception:
            ver = "m0:"
        try:
            repo = self.store._conn.execute(
                "SELECT COUNT(*), COALESCE(MAX(indexed_at), '') FROM repo_sources"
            ).fetchone()
            ver += f":r{repo[0]}:{repo[1]}"
        except Exception:
            ver += ":r0:"
        self._corpus_version_cached = (gen, ver)
        return ver

    def _verbatim_short_circuit(
        self,
        question: str,
        hits: list[MemoryRecord],
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
        self,
        question: str,
        *,
        k: int = 5,
        type_: str | None = None,
        snippet_chars: int = 800,
        include_repos: bool = True,
        intent_text: str | None = None,
        session_id: str | None = None,
        use_context_pack: bool | None = None,
    ) -> dict[str, Any]:
        """Synthesised Q&A over the memory archive (RAG).

        Pipeline: hybrid search top-`k` → format snippets with `[id]`
        citations → MLXChat 7B (`MEMO_LLM_MODEL`) generates a prose
        answer with inline citations. Returns:

            {
                "question": str,
                "answer": str,            # may say "I couldn't find ..."
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
        from memo.flags import flag_bool, flag_int

        if snippet_chars == 800:
            snippet_chars = flag_int("MEMO_ASK_SNIPPET_CHARS") or 800
        if use_context_pack is None:
            use_context_pack = flag_bool("MEMO_CONTEXT_PACK")
        norm_question, sources, user_msg, hits = self._build_ask_context(
            question,
            k=k,
            type_=type_,
            snippet_chars=snippet_chars,
            include_repos=include_repos,
            intent_text=intent_text,
            session_id=session_id,
            use_context_pack=use_context_pack,
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
        verbatim_hits = _filter_verbatim_hits(hits, sources, use_context_pack=use_context_pack)
        verbatim = self._verbatim_short_circuit(question, verbatim_hits)
        if verbatim is not None:
            return {
                "question": norm_question,
                "answer": verbatim,
                "sources": sources,
            }

        # Lazy-construct the chat client (same instance used by auto_derive).
        chat = self._ensure_chat()
        try:
            out = chat.chat(
                model=self.cfg.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": resolve_prompt("ask", _ASK_SYSTEM_PROMPT, self.cfg.state_dir),
                    },
                    {"role": "user", "content": user_msg},
                ],
                # Higher max_tokens than auto_derive — answers can run a
                # paragraph or two.
                options={"temperature": 0.0, "max_tokens": 768},
            )
            answer = ((out.get("message") or {}).get("content") or "").strip()
        except Exception as exc:
            answer = f"(error querying the model: {type(exc).__name__})"

        from memo.flags import flag_float

        _ask_min = flag_float("MEMO_GROUNDING_ASK_MIN") or 0.0
        if _ask_min > 0.0 and answer:
            _src_text = "\n\n".join(str(s.get("snippet") or "") for s in sources)
            _entail = score_grounding(chat, self.cfg.llm_model, source=_src_text, claim=answer)
            if _entail is not None and _entail < _ask_min:
                from memo.flags import flag_str

                answer = flag_str("MEMO_ASK_FALLBACK_MSG")

        return {
            "question": norm_question,
            "answer": answer,
            "sources": sources,
        }

    def ask_stream(
        self,
        question: str,
        *,
        k: int = 5,
        type_: str | None = None,
        snippet_chars: int = 800,
        include_repos: bool = True,
        intent_text: str | None = None,
        session_id: str | None = None,
        use_context_pack: bool | None = None,
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
        from memo.flags import flag_bool, flag_int

        if snippet_chars == 800:
            snippet_chars = flag_int("MEMO_ASK_SNIPPET_CHARS") or 800
        if use_context_pack is None:
            use_context_pack = flag_bool("MEMO_CONTEXT_PACK")
        _, sources, user_msg, hits = self._build_ask_context(
            question,
            k=k,
            type_=type_,
            snippet_chars=snippet_chars,
            include_repos=include_repos,
            intent_text=intent_text,
            session_id=session_id,
            use_context_pack=use_context_pack,
        )
        if not sources:
            from memo.flags import flag_str

            yield {
                "event": "done",
                "answer": flag_str("MEMO_ASK_FALLBACK_MSG"),
                "sources": [],
            }
            return

        yield {"event": "sources", "sources": sources}

        # Verbatim short-circuit (literal-phrase queries): emit body as a
        # single token-style event so consumers that show progressive
        # output still get something to render, then a terminal `done`.
        verbatim_hits = _filter_verbatim_hits(hits, sources, use_context_pack=use_context_pack)
        verbatim = self._verbatim_short_circuit(question, verbatim_hits)
        if verbatim is not None:
            yield {"event": "token", "delta": verbatim}
            yield {"event": "done", "answer": verbatim, "sources": sources}
            return

        chat = self._ensure_chat()
        accum_parts: list[str] = []
        try:
            for delta in chat.chat_stream(
                model=self.cfg.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": resolve_prompt("ask", _ASK_SYSTEM_PROMPT, self.cfg.state_dir),
                    },
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
