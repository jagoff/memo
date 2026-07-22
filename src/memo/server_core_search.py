from __future__ import annotations

from typing import Any

from fastmcp import Context

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
    return "\n".join(p for p in (_presence_line(memory), idle) if p)


def _presence_line(memory: Memory) -> str:
    """Today's cross-agent activity summary; '' when off or nothing happened.

    Every call is internally guarded — ``read_today`` swallows IO errors and
    ``summary_line`` is pure — so no broad except is needed here.
    """
    from memo.flags import flag_bool

    if not flag_bool("MEMO_PRESENCE_NOTIFY"):
        return ""
    from memo import presence

    return presence.summary_line(presence.read_today(memory.cfg.state_dir))


def _bump_recall_presence(memory: Memory, hits: list[Any]) -> None:
    """Reflect an MCP recall in today's presence counters (decoration only)."""
    if hits:
        from memo import presence

        presence.bump(memory.cfg.state_dir, recalls=len(hits))


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
        try:
            from consciousness_contracts import EmbedderProfile

            profile = EmbedderProfile(
                model_id=memory.store.embedder_model,
                dims=int(cfg.embedder_dims),
                normalization="l2",
                provider="memo",
            )
            return profile.to_dict()
        except ImportError:
            return {
                "schema": "consciousness.embedder_profile.v1",
                "model_id": memory.store.embedder_model,
                "dims": int(cfg.embedder_dims),
                "normalization": "l2",
                "max_seq_len": None,
                "quantization": None,
                "provider": "memo",
            }

    @annotated_tool(server, **READ_ONLY)
    def memo_unified_briefing(cwd: str | None = None, source: str = "") -> dict[str, Any]:
        """Load a compact startup briefing from memo and optional Synapse state.

        Read-only. Call before deciding or answering so prior durable facts can
        ground the task. Pass `cwd` to bias project context and `source` to
        attribute consult logs to the calling client.
        """
        from memo.briefing import (
            compact_text,
            memo_native_briefing_lines,
            synapse_briefing_lines,
        )
        from memo.flags import flag_int

        t0 = now_ms()
        # memo's OWN durable corpus FIRST — so an MCP-only agent gets grounded
        # even on a single Mac where synapse is unreachable (previously this
        # tool returned nothing but synapse's borrow → empty briefing offline).
        loops_n = max(1, flag_int("MEMO_BRIEFING_LOOPS_N") or 5)
        loops_days = max(1, flag_int("MEMO_BRIEFING_LOOPS_DAYS") or 7)
        raw_lines: list[str] = memo_native_briefing_lines(
            memory, loops_n=loops_n, loops_days=loops_days
        )
        # Synapse unified state is additive/optional.
        raw_lines.extend(synapse_briefing_lines(cwd))
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
        query: str,
        limit: int = 10,
        type: str | None = None,
        body_chars: int = 280,
        mode: str = "hybrid",
        explain: bool = False,
        source: str = "",
        file: str = "",
        date_from: str | None = None,
        date_to: str | None = None,
        when: str | None = None,
    ) -> dict[str, Any]:
        """Search durable memories by text, vector similarity, or hybrid mode.

        Read-only. Use for direct retrieval when you need source records, ids,
        dates, tags, or excerpts. `mode` accepts hybrid, vec, or bm25;
        `body_chars` controls snippet length, and `source` attributes consult
        logging.
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
        _bump_recall_presence(memory, out)

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
        question: str,
        k: int = 7,
        type: str | None = None,
        snippet_chars: int = 700,
        budget_chars: int = 6000,
        include_profile: bool = True,
        include_dynamic: bool = True,
        source: str = "",
    ) -> dict[str, Any]:
        """Build prompt-ready memory context without calling the answer LLM.

        Read-only. Returns static profile, dynamic recent context, query hits,
        omissions, and a readonly prompt wrapper for direct model injection.
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
        query: str,
        limit: int = 10,
        type: str | None = None,
        body_chars: int = 280,
        mode: str = "hybrid",
        source: str = "",
    ) -> dict[str, Any]:
        """Search memories and include retrieval trace diagnostics.

        Read-only. Use when debugging ranking, filters, or recall misses rather
        than for normal lookup. Returns the same style of hits as memo_search
        plus trace metadata that explains how candidates were selected.
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
        query: str,
        hits: list[dict[str, Any]],
        top_n: int | None = None,
        body_chars: int = 1200,
    ) -> list[dict[str, Any]]:
        """Rerank candidate memory hits for a query.

        Read-only. Use after memo_search or another retrieval source when you
        already have candidate hit dictionaries and need the most relevant
        subset ordered for answer synthesis. `top_n` limits the returned list.
        """
        return memory.rerank_hits(query, hits, top_n=top_n, body_chars=body_chars)

    @annotated_tool(server, **READ_ONLY)
    def memo_embed_query(text: str) -> dict[str, Any]:
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
    def memo_embed_batch(texts: list[str]) -> dict[str, Any]:
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
        question: str,
        k: int = 5,
        type: str | None = None,
        snippet_chars: int = 800,
        include_repos: bool = True,
        session_id: str | None = None,
        source: str = "",
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Answer a question using memo retrieval and citations.

        Read-only. Use when you want a synthesized answer grounded in durable
        memories instead of raw hit lists. `k`, `type`, and `snippet_chars` tune
        retrieval; `source` attributes consult logging. With client sampling
        enabled, synthesis runs on the calling model (see `synthesizer` field).
        """
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
        question: str,
        k: int = 7,
        type: str | None = None,
        history: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
        snippet_chars: int = 800,
        session_id: str | None = None,
        source: str = "",
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Answer a conversational question with optional history and context.

        Read-only. Use instead of memo_ask when prior turns or explicit
        `context` should shape retrieval and synthesis. `history` is a list of
        chat messages; `session_id` links the answer to a tracked memo session.
        With client sampling enabled, synthesis runs on the calling model (see
        `synthesizer` field).
        """
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
