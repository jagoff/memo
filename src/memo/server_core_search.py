from __future__ import annotations

from typing import Annotated, Any

from fastmcp import Context
from pydantic import Field

from memo.memory import Memory
from memo.server_annotations import READ_ONLY, annotated_tool
from memo.server_common import log_consult, now_ms, run_synth


def _read_notification(memory: Memory) -> str:
    """Compose memo's cross-agent presence line with the pending idle notice.

    Peeks (never consumes) the idle-capture notification, and — gated by
    ``MEMO_PRESENCE_NOTIFY`` — prepends today's activity summary so MCP-only
    agents (no statusline: Codex, Devin, opencode, Cursor) see memo working on
    every tool response. Both parts are best-effort; presence is decoration and
    must never break a read.
    """
    notif_path = memory.cfg.state_dir / "pending_idle_notification.txt"
    try:
        idle = notif_path.read_text(encoding="utf-8").strip()
    except Exception:
        idle = ""
    presence_line = ""
    try:
        from memo.flags import flag_bool

        if flag_bool("MEMO_PRESENCE_NOTIFY"):
            from memo import presence

            presence_line = presence.summary_line(presence.read_today(memory.cfg.state_dir))
    except Exception:
        presence_line = ""
    return "\n".join(p for p in (presence_line, idle) if p)


def _file_search_notes(
    *,
    file: str,
    date_from: str | None,
    date_to: str | None,
    explain: bool,
) -> list[str]:
    """Describe options that file-scoped search cannot apply."""
    notes: list[str] = []
    if file and (date_from or date_to):
        notes.append(
            "date filters (when/date_from/date_to) are not applied when 'file' is set; "
            "results are not filtered by date"
        )
    if file and explain:
        notes.append("explain is not available when 'file' is set; 'explain' fields are empty")
    return notes


def register(server: Any, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_get_embedder_profile() -> dict[str, Any]:
        """Return the active embedding model profile.

        Read-only. Use this to inspect the model id, vector dimensions,
        normalization, and provider that memo uses for semantic search.
        Useful when verifying compatibility with stored vectors or external
        retrieval components.
        """
        cfg = memory.cfg
        return {
            "schema": "memo.embedder_profile.v1",
            "model_id": memory.store.embedder_model,
            "dims": int(cfg.embedder_dims),
            "normalization": "l2",
            "max_seq_len": None,
            "quantization": None,
            "provider": "memo",
        }

    @annotated_tool(server, **READ_ONLY)
    def memo_unified_briefing(
        cwd: Annotated[
            str | None,
            Field(
                description="Working directory; its basename scopes operational items "
                "(focus, handoffs, attention, conflicts) to that project. "
                "None includes every project."
            ),
        ] = None,
        source: Annotated[
            str,
            Field(
                description="Calling layer for consult attribution "
                "(e.g. 'claude-code', 'codex'); empty falls back to client info."
            ),
        ] = "",
    ) -> dict[str, Any]:
        """Load a compact startup briefing from Memo's durable and operational state.

        Read-only. Call before deciding or answering so prior durable facts can
        ground the task. Pass `cwd` to bias project context and `source` to
        attribute consult logs to the calling client.
        """
        from memo.briefing import (
            compact_text,
            memo_native_briefing_lines,
            operational_briefing_lines,
        )
        from memo.flags import flag_int

        t0 = now_ms()
        loops_n = max(1, flag_int("MEMO_BRIEFING_LOOPS_N") or 5)
        loops_days = max(1, flag_int("MEMO_BRIEFING_LOOPS_DAYS") or 7)
        raw_lines: list[str] = memo_native_briefing_lines(
            memory, loops_n=loops_n, loops_days=loops_days
        )
        raw_lines.extend(operational_briefing_lines(memory, cwd))
        markdown = compact_text("\n".join(raw_lines), max_chars=900)
        lines = markdown.splitlines() if markdown else []
        log_consult(
            memory,
            tool="unified_briefing",
            query=cwd or "briefing",
            hits=[],
            t0_ms=t0,
            source=source,
        )

        return {
            "available": bool(lines),
            "markdown": markdown,
            "lines": lines,
            "notification": _read_notification(memory),
        }

    @annotated_tool(server, **READ_ONLY)
    def memo_search(
        query: Annotated[
            str,
            Field(description="Search text; empty or whitespace-only returns no hits."),
        ],
        limit: Annotated[
            int,
            Field(description="Maximum hits to return (clamped to 1-500)."),
        ] = 10,
        type: Annotated[
            str | None,
            Field(
                description="Restrict retrieval to one memory type "
                "(e.g. 'decision', 'fact'); None searches every type."
            ),
        ] = None,
        body_chars: Annotated[
            int,
            Field(
                description="Character cap per hit body; longer bodies are truncated "
                "with body_truncated=true. Negative disables truncation."
            ),
        ] = 280,
        mode: Annotated[
            str,
            Field(
                description="Retrieval mode: 'hybrid' (RRF fusion of vector + BM25, "
                "default), 'vec' (semantic only), 'bm25' (keyword FTS5), 'exact' "
                "(strict-AND keyword with tag/title boost), or 'fuzzy' (typo-tolerant "
                "keyword). Unrecognized values behave as 'hybrid'."
            ),
        ] = "hybrid",
        explain: Annotated[
            bool,
            Field(
                description="When true, add per-hit 'explain' fields and a 'trace' key "
                "showing pipeline scoring. Not available with 'file' (explain stays "
                "empty)."
            ),
        ] = False,
        source: Annotated[
            str,
            Field(
                description="Calling layer for consult attribution "
                "(e.g. 'claude-code', 'codex'); empty falls back to client info."
            ),
        ] = "",
        file: Annotated[
            str,
            Field(
                description="Keep only hits whose capture-stamped files_read/"
                "files_modified arrays contain this path fragment (case-insensitive "
                "substring). Date filters and explain are not applied in file mode."
            ),
        ] = "",
        date_from: Annotated[
            str | None,
            Field(
                description="Inclusive ISO date/datetime lower bound "
                "(e.g. '2026-07-01'); ignored when 'file' is set."
            ),
        ] = None,
        date_to: Annotated[
            str | None,
            Field(
                description="Inclusive ISO date/datetime upper bound; a bare date "
                "covers that whole day. Ignored when 'file' is set."
            ),
        ] = None,
        when: Annotated[
            str | None,
            Field(
                description="Natural-language date phrase (EN/ES: 'yesterday', "
                "'last week', 'hace 3 dias', ...) parsed into date_from/date_to when "
                "neither is set; unrecognized phrases apply no date filter."
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Search durable memories by text, vector similarity, or hybrid mode.

        Read-only. Returns raw ranked hits — source records with ids, dates,
        tags, and excerpts. Use memo_context instead for a budgeted
        prompt-ready pack, memo_search_trace for scoring diagnostics, and
        memo_rerank to reorder hits you already have.
        """
        if when and not (date_from or date_to):
            from memo.nl_dates import parse_date_range

            date_from, date_to = parse_date_range(when)
        t0 = now_ms()
        out: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        explanations: dict[str, dict[str, Any]] = {}
        # File-scoped search (search_by_file) takes no date filters and computes
        # no explain trace. Say so explicitly instead of silently dropping the
        # parameters — the caller must not present unfiltered hits as filtered.
        notes = _file_search_notes(
            file=file,
            date_from=date_from,
            date_to=date_to,
            explain=explain,
        )
        if explain and file:
            records = memory.search_by_file(
                query,
                file=file,
                limit=limit,
                mode=mode,
                type_=type,
                quality_rerank=True,
            )
        elif explain:
            from memo.search_explain import build_search_explanations

            envelope = memory.search_with_trace(
                query,
                limit=limit,
                type_=type,
                mode=mode,
                date_from=date_from,
                date_to=date_to,
                quality_rerank=True,
            )
            records = envelope["hits"]
            trace = envelope.get("trace") or []
            explanations = build_search_explanations(records, trace)
        else:
            records = (
                memory.search_by_file(
                    query,
                    file=file,
                    limit=limit,
                    mode=mode,
                    type_=type,
                    quality_rerank=True,
                )
                if file
                else memory.search(
                    query,
                    limit=limit,
                    type_=type,
                    mode=mode,
                    date_from=date_from,
                    date_to=date_to,
                    quality_rerank=True,
                )
            )
        for r in records:
            d = r.to_dict()
            body = d.get("body") or ""
            if body_chars >= 0 and len(body) > body_chars:
                d["body"] = body[:body_chars].rstrip() + "…"
                d["body_truncated"] = True
            if explain:
                d["explain"] = explanations.get(str(d.get("id") or ""), {})
            out.append(d)
        log_consult(memory, tool="search", query=query, hits=out, t0_ms=t0, source=source)

        # Cross-agent presence: reflect this recall so MCP-only agents (which
        # never run the Claude recall-hook) read honest counts. Decoration only.
        if out:
            from memo import presence

            presence.bump(memory.cfg.state_dir, recalls=len(out))

        # Read pending idle notification (best-effort, races with writer)
        notification = _read_notification(memory)

        return {
            "hits": out,
            "notification": notification,
            **({"note": " ".join(notes)} if notes else {}),
            **({"trace": trace} if explain else {}),
        }

    @annotated_tool(server, **READ_ONLY)
    def memo_context(
        question: Annotated[
            str,
            Field(
                description="Natural-language question used to retrieve and pack relevant memories."
            ),
        ],
        k: Annotated[
            int,
            Field(
                description="Number of query hits retrieved for the pack "
                "(hybrid mode; clamped to 1-500)."
            ),
        ] = 7,
        type: Annotated[
            str | None,
            Field(
                description="Restrict retrieval to one memory type "
                "(e.g. 'decision', 'fact'); None searches every type."
            ),
        ] = None,
        snippet_chars: Annotated[
            int,
            Field(description="Character cap per memory snippet inside the pack."),
        ] = 700,
        budget_chars: Annotated[
            int,
            Field(
                description="Total character budget for the returned readonly prompt "
                "wrapper; rows beyond the budget are trimmed and counted in "
                "'omissions'."
            ),
        ] = 6000,
        include_profile: Annotated[
            bool,
            Field(description="Include the static profile section (identity/preference lines)."),
        ] = True,
        include_dynamic: Annotated[
            bool,
            Field(
                description="Include the dynamic section: up to 5 memories updated "
                "in the last 7 days."
            ),
        ] = True,
        source: Annotated[
            str,
            Field(
                description="Calling layer for consult attribution "
                "(e.g. 'claude-code', 'codex'); empty falls back to client info."
            ),
        ] = "",
    ) -> dict[str, Any]:
        """Build prompt-ready memory context without calling the answer LLM.

        Read-only. Unlike memo_search's raw hit list, this returns a budgeted,
        prompt-ready context pack for direct injection: static profile, dynamic
        recent context, query hits, omissions, and a readonly prompt wrapper.
        """
        from memo.context_surface import build_context_surface, consult_hits_from_context
        from memo.flags import flag_bool

        if not flag_bool("MEMO_CONTEXT_SURFACE"):
            return {
                "status": "disabled",
                "reason": "MEMO_CONTEXT_SURFACE=0 disables memo_context.",
                "question": question,
            }
        t0 = now_ms()
        out = build_context_surface(
            memory,
            question,
            k=k,
            type_=type,
            snippet_chars=snippet_chars,
            budget_chars=budget_chars,
            include_profile=include_profile,
            include_dynamic=include_dynamic,
        )
        log_consult(
            memory,
            tool="context",
            query=question,
            hits=consult_hits_from_context(out),
            t0_ms=t0,
            source=source,
        )
        out["notification"] = _read_notification(memory)
        return out

    @annotated_tool(server, **READ_ONLY)
    def memo_search_trace(
        query: Annotated[
            str,
            Field(description="Search text; empty or whitespace-only returns no hits."),
        ],
        limit: Annotated[
            int,
            Field(description="Maximum hits to return (clamped to 1-500)."),
        ] = 10,
        type: Annotated[
            str | None,
            Field(
                description="Restrict retrieval to one memory type "
                "(e.g. 'decision', 'fact'); None searches every type."
            ),
        ] = None,
        body_chars: Annotated[
            int,
            Field(
                description="Character cap per hit body; longer bodies are truncated "
                "with body_truncated=true. Negative disables truncation."
            ),
        ] = 280,
        mode: Annotated[
            str,
            Field(
                description="Retrieval mode: 'hybrid' (RRF fusion of vector + BM25, "
                "default), 'vec' (semantic only), 'bm25' (keyword FTS5), 'exact' "
                "(strict-AND keyword with tag/title boost), or 'fuzzy' (typo-tolerant "
                "keyword). Unrecognized values behave as 'hybrid'."
            ),
        ] = "hybrid",
        source: Annotated[
            str,
            Field(
                description="Calling layer for consult attribution "
                "(e.g. 'claude-code', 'codex'); empty falls back to client info."
            ),
        ] = "",
    ) -> dict[str, Any]:
        """Search memories and include retrieval trace diagnostics.

        Read-only. Debug variant of memo_search: use it when investigating
        ranking, filters, or recall misses rather than for normal lookup.
        Returns the same style of hits as memo_search plus trace metadata that
        explains how candidates were selected and scored at each stage.
        """
        t0 = now_ms()
        envelope = memory.search_with_trace(
            query,
            limit=limit,
            type_=type,
            mode=mode,
            quality_rerank=True,
        )
        hits: list[dict[str, Any]] = []
        for r in envelope["hits"]:
            d = r.to_dict()
            body = d.get("body") or ""
            if body_chars >= 0 and len(body) > body_chars:
                d["body"] = body[:body_chars].rstrip() + "…"
                d["body_truncated"] = True
            hits.append(d)
        log_consult(memory, tool="search_trace", query=query, hits=hits, t0_ms=t0, source=source)
        return {"hits": hits, "trace": envelope["trace"]}

    @annotated_tool(server, **READ_ONLY)
    def memo_rerank(
        query: Annotated[
            str,
            Field(
                description="Query text the cross-encoder scores each hit against; "
                "empty returns the hits unchanged."
            ),
        ],
        hits: Annotated[
            list[dict[str, Any]],
            Field(
                description="Candidate hit dicts (e.g. from memo_search); each is "
                "scored on its title plus snippet/body and returned with a "
                "'rerank_score' field added. Non-dict entries are dropped."
            ),
        ],
        top_n: Annotated[
            int | None,
            Field(
                description="Keep only the N best hits after reranking; "
                "None or a non-positive value returns all."
            ),
        ] = None,
        body_chars: Annotated[
            int,
            Field(description="Character cap of each hit's snippet/body fed to the reranker."),
        ] = 1200,
    ) -> list[dict[str, Any]]:
        """Rerank candidate memory hits for a query.

        Read-only. Performs no retrieval of its own (unlike memo_search) — it
        only reorders caller-supplied hit dictionaries, e.g. after memo_search
        or another retrieval source, when you need the most relevant subset
        ordered for answer synthesis. When reranking is disabled in this
        install, hits pass through in input order without scores.
        """
        return memory.rerank_hits(query, hits, top_n=top_n, body_chars=body_chars)

    @annotated_tool(server, **READ_ONLY)
    def memo_embed_query(
        text: Annotated[
            str,
            Field(
                description="Query text to embed on the query side "
                "(instruction-prefixed for asymmetric retrieval); empty or "
                "whitespace-only raises an error."
            ),
        ],
    ) -> dict[str, Any]:
        """Embed one query string with memo's query embedding path.

        Read-only. Use for diagnostics or integrations that need the exact
        vector memo would use for retrieval queries. Rejects empty text and
        returns the vector, dimension, and model id.
        """
        if not text or not text.strip():
            raise ValueError("memo_embed_query: empty text")
        vec = memory.embedder.embed_query(text)
        return {"vector": vec, "dim": len(vec), "model": memory.store.embedder_model}

    @annotated_tool(server, **READ_ONLY)
    def memo_embed_batch(
        texts: Annotated[
            list[str],
            Field(
                description="Document strings to embed on the document side (raw, no "
                "query instruction prefix); an empty list returns no vectors."
            ),
        ],
    ) -> dict[str, Any]:
        """Embed one or more document strings with memo's document embedder.

        Read-only. Use for diagnostics or external indexing when you need
        document vectors from the same model memo uses internally. Pass a list
        of strings; an empty list returns no vectors without error.
        """
        if not texts:
            return {"vectors": [], "dim": 0, "model": memory.store.embedder_model}
        vecs = memory.embedder.embed(texts)
        dim = len(vecs[0]) if vecs else 0
        return {"vectors": vecs, "dim": dim, "model": memory.store.embedder_model}

    @annotated_tool(server, **READ_ONLY)
    async def memo_ask(
        question: Annotated[
            str,
            Field(description="Natural-language question to answer from durable memories."),
        ],
        k: Annotated[
            int,
            Field(description="Number of memories to retrieve as grounding (top-k)."),
        ] = 5,
        type: Annotated[
            str | None,
            Field(
                description="Restrict retrieval to one memory type "
                "(e.g. 'decision', 'fact'); None searches every type."
            ),
        ] = None,
        snippet_chars: Annotated[
            int | None,
            Field(description="Character cap per cited snippet; None uses the default."),
        ] = None,
        include_repos: Annotated[
            bool,
            Field(description="Also search indexed repository knowledge, not just memories."),
        ] = True,
        session_id: Annotated[
            str | None,
            Field(description="Tracked memo session id to associate the answer with."),
        ] = None,
        source: Annotated[
            str,
            Field(
                description="Calling layer for consult attribution "
                "(e.g. 'claude-code', 'codex'); empty falls back to client info."
            ),
        ] = "",
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Answer a question using memo retrieval and citations.

        Read-only. Use when you want a synthesized answer grounded in durable
        memories instead of raw hit lists. `k`, `type`, and `snippet_chars` tune
        retrieval; `source` attributes consult logging. With client sampling
        enabled, synthesis runs on the calling model (see `synthesizer` field).
        """
        # Enforce the same size/shape bounds the HTTP /chat route applies, so the
        # MCP surface can't be handed an unbounded question (e.g. echoed from a
        # prompt-injected tool result) that runs synchronously against the
        # in-process embedder/LLM with no cap.
        from memo.server_chat import validate_chat_payload

        question, k, type, _hist, _ctx = validate_chat_payload(
            {"question": question, "k": k, "type": type}
        )
        t0 = now_ms()
        res, synthesizer = await run_synth(
            memory,
            ctx,
            lambda: memory.ask(
                question,
                k=k,
                type_=type,
                snippet_chars=snippet_chars,
                include_repos=include_repos,
                session_id=session_id,
            ),
        )
        out = res if isinstance(res, dict) else {"answer": str(res)}
        cites = out.get("citations") or out.get("sources") or []
        hit_dicts = [c for c in cites if isinstance(c, dict)]
        log_consult(memory, tool="ask", query=question, hits=hit_dicts, t0_ms=t0, source=source)
        out["synthesizer"] = synthesizer

        # Read pending idle notification (best-effort, races with writer)
        out["notification"] = _read_notification(memory)

        return out

    @annotated_tool(server, **READ_ONLY)
    async def memo_chat_ask(
        question: Annotated[
            str,
            Field(description="Natural-language question for this conversational turn."),
        ],
        k: Annotated[
            int,
            Field(description="Number of memories to retrieve as grounding (top-k)."),
        ] = 7,
        type: Annotated[
            str | None,
            Field(
                description="Restrict retrieval to one memory type "
                "(e.g. 'decision', 'fact'); None searches every type."
            ),
        ] = None,
        history: Annotated[
            list[dict[str, Any]] | None,
            Field(
                description="Prior chat turns as {'role', 'content'} dicts; "
                "bounded (128 items / 512KB) and used to shape retrieval."
            ),
        ] = None,
        context: Annotated[
            dict[str, Any] | None,
            Field(
                description="Extra structured context for synthesis "
                "(bounded to 256KB); merged with session_id when given."
            ),
        ] = None,
        snippet_chars: Annotated[
            int | None,
            Field(description="Character cap per cited snippet; None uses the default."),
        ] = None,
        session_id: Annotated[
            str | None,
            Field(description="Tracked memo session id linking the answer to a session."),
        ] = None,
        source: Annotated[
            str,
            Field(
                description="Calling layer for consult attribution "
                "(e.g. 'claude-code', 'codex'); empty falls back to client info."
            ),
        ] = "",
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Answer a conversational question with optional history and context.

        Read-only. Use instead of memo_ask when prior turns or explicit
        `context` should shape retrieval and synthesis. `history` is a list of
        chat messages; `session_id` links the answer to a tracked memo session.
        With client sampling enabled, synthesis runs on the calling model (see
        `synthesizer` field).
        """
        # Enforce the same size/shape bounds the HTTP /chat route applies (32KB
        # question, 128-item/512KB history, 256KB context, depth cap), so the MCP
        # surface can't be handed an unbounded history/context payload running
        # synchronously against the in-process embedder/LLM with no cap.
        from memo.server_chat import validate_chat_payload

        question, k, type, history, context = validate_chat_payload(
            {"question": question, "k": k, "type": type, "history": history, "context": context}
        )
        t0 = now_ms()
        merged_context = dict(context or {})
        if session_id and "session_id" not in merged_context:
            merged_context["session_id"] = session_id
        res, synthesizer = await run_synth(
            memory,
            ctx,
            lambda: memory.chat_ask(
                question,
                k=k,
                type_=type,
                history=history,
                context=merged_context or None,
                snippet_chars=snippet_chars,
            ),
        )
        out = res if isinstance(res, dict) else {"answer": str(res)}
        out["synthesizer"] = synthesizer
        cites = out.get("citations") or out.get("sources") or []
        hit_dicts = [c for c in cites if isinstance(c, dict)]
        log_consult(
            memory, tool="chat_ask", query=question, hits=hit_dicts, t0_ms=t0, source=source
        )

        # Read pending idle notification (best-effort, races with writer)
        out["notification"] = _read_notification(memory)

        return out
