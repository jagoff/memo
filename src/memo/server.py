"""MCP server — `memo-mcp` entry point.

Exposes the `Memory` API as MCP tools so any MCP-aware client (Claude
Code, Devin, Claude Desktop, etc) can save / search / list / get /
update / delete entries from natural language.

Tools are deliberately small and verb-shaped so the LLM picks them up
without prompt engineering. Each tool returns plain JSON-serialisable
dicts (no SimpleNamespaces, no dataclasses) — the MCP transport
serialises everything anyway, but a dict surfaces field names in the
tool result message which helps the LLM decide what to do next.

Run via the installed entry point:

    memo-mcp

Or programmatically:

    from memo.server import build_server
    server = build_server()
    server.run()
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, cast

from fastmcp import FastMCP

from memo import server_analytics as _srv_analytics
from memo import server_asof as _srv_asof
from memo import server_backup as _srv_backup
from memo import server_cache as _srv_cache
from memo import server_collaborative as _srv_collaborative
from memo import server_consolidate as _srv_consolidate
from memo import server_contextual as _srv_contextual
from memo import server_contradict as _srv_contradict
from memo import server_encrypt as _srv_encrypt
from memo import server_entities as _srv_entities
from memo import server_feedback as _srv_feedback
from memo import server_graph as _srv_graph
from memo import server_health as _srv_health
from memo import server_import_export as _srv_import_export
from memo import server_links as _srv_links
from memo import server_multimodal as _srv_multimodal
from memo import server_query as _srv_query
from memo import server_reflect as _srv_reflect
from memo import server_repo as _srv_repo
from memo import server_share as _srv_share
from memo import server_sync as _srv_sync
from memo import server_synthesis as _srv_synthesis
from memo import server_temporal as _srv_temporal
from memo import server_version as _srv_version
from memo._trace import TRACE_HEADER, trace_scope
from memo.config import Config
from memo.memory import AmbiguousIdError, Memory


def _make_trace_middleware() -> Any:
    """Bridge the synapse trace header into the shared trace contextvar.

    Synapse forwards its ``synapse://trace/<id>`` two ways: the
    ``SYNAPSE_TRACE_ID`` env var (subprocess path) and the
    ``x-synapse-trace-id`` HTTP header (warm-daemon path). The env var is read
    in `write_ops`; this middleware covers the header path so a warm-daemon
    write stamps the same trace id and the ledger trails stitch into one span.

    Returns ``None`` when FastMCP middleware isn't available so `build_server`
    can skip wiring it without failing.
    """
    try:
        from fastmcp.server.dependencies import get_http_headers
        from fastmcp.server.middleware import Middleware
    except ImportError:
        return None

    class _TraceMiddleware(Middleware):  # type: ignore[misc]
        async def on_call_tool(self, context: Any, call_next: Any) -> Any:
            trace_id = ""
            try:
                headers = get_http_headers() or {}
                trace_id = (headers.get(TRACE_HEADER) or "").strip()
            except Exception:
                trace_id = ""
            if not trace_id:
                return await call_next(context)
            with trace_scope(trace_id):
                return await call_next(context)

    return _TraceMiddleware()


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)


def _log_consult(
    memory: Memory,
    *,
    tool: str,
    query: str,
    hits: list[dict[str, Any]],
    t0_ms: int,
    source: str = "",
) -> None:
    """Record an MCP consult into the shared recall ring buffer so memo's
    usefulness is observable for EVERY consumer, not just the Claude Code
    recall-hook (see `memo usefulness`). Best-effort — telemetry must never
    break a tool call."""
    try:
        from memo.dashboard import append_recall_log

        append_recall_log(
            memory.cfg.state_dir,
            prompt=query or "",
            hits=hits or [],
            via=f"mcp:{tool}",
            source=(source or "").strip().lower() or None,
            latency_ms=_now_ms() - t0_ms,
        )
    except Exception:
        pass


def build_server(memory: Memory | None = None) -> FastMCP:
    """Build the MCP server. Accepts an explicit `Memory` for tests.

    The default constructs from `Config.from_env()` — production runs
    pick up `MEMO_*` env vars set by the calling shell or by Claude
    Code's `claude mcp add` invocation.
    """
    if memory is None:
        memory = Memory(Config.from_env())

    server = FastMCP("memo")

    # Stitch the synapse trace header into the shared trace contextvar so
    # warm-daemon writes carry the same trace id as the subprocess path.
    _trace_mw = _make_trace_middleware()
    if _trace_mw is not None:
        server.add_middleware(_trace_mw)

    # Domain tool modules register their @server.tool() closures here.
    _srv_repo.register(server, memory)
    _srv_entities.register(server, memory)
    _srv_temporal.register(server, memory)
    _srv_contradict.register(server, memory)
    _srv_consolidate.register(server, memory)
    _srv_synthesis.register(server, memory)
    _srv_reflect.register(server, memory)
    _srv_graph.register(server, memory)
    _srv_health.register(server, memory)
    _srv_contextual.register(server, memory)
    _srv_links.register(server, memory)
    _srv_version.register(server, memory)
    _srv_query.register(server, memory)
    _srv_backup.register(server, memory)
    _srv_sync.register(server, memory)
    _srv_cache.register(server, memory)
    _srv_encrypt.register(server, memory)
    _srv_share.register(server, memory)
    _srv_analytics.register(server, memory)
    _srv_import_export.register(server, memory)
    _srv_feedback.register(server, memory)
    _srv_multimodal.register(server, memory)
    _srv_collaborative.register(server, memory)
    _srv_asof.register(server, memory)
    # ToolSpec-registry tools (new pattern — see mcp_tools.py)
    from memo.mcp_tools import register_all as _register_mcp_tools

    _register_mcp_tools(server, memory)

    @server.tool()
    def memory_save(
        content: str,
        title: str | None = None,
        type: str = "note",
        tags: list[str] | None = None,
        auto_derive: bool = False,
        extra: dict[str, Any] | None = None,
        respect_synapse_freeze: bool | None = None,
    ) -> dict[str, Any]:
        """Persist a memory to the vault + index it.

        Args:
            content: Markdown body. Required, non-empty. The first
                non-empty line is used as title if `title` is omitted.
            title: Optional short title. Defaults to first line of
                content. Truncated to 80 chars.
            type: One of `decision`, `fact`, `bug`, `feedback`,
                `preference`, `note`, `manual`. Default `note`.
            tags: Optional list. Lower-cased + de-duplicated.
            auto_derive: When True and title/type/tags are missing,
                Qwen2.5-3B helper LLM derives them from the content.
                Adds ~1-2s latency on cold call. Use when the calling
                agent doesn't have enough context to derive metadata.
            extra: Optional metadata bag persisted into the frontmatter
                `extra` block and `meta.extra_json`. Synapse callers set
                provenance keys here (`synapse_trace_id`,
                `synapse_route_reason`, `synapse_write_policy_schema`,
                `synapse_write_target`, `synapse_agent_id`,
                `synapse_agent_signature`); those are also mirrored into
                `history.events.delta_json` so `memory_provenance(id)`
                can replay them.
            respect_synapse_freeze: When True, query synapse's
                RealityConflict ledger before commit and refuse the
                save if a blocking `freeze_write` covers this
                memoria's topic. Returns
                `{"status": "refused", "conflict": {...}}` instead of
                the persisted record. Defaults to the env knob
                `MEMO_RESPECT_SYNAPSE_FREEZE=1`. Only fires when
                `extra.synapse_trace_id` is set.

        Returns the persisted record (id, path, title, ...). When a
        synapse freeze blocks the write, returns
        `{"status": "refused", "conflict": {...}, "message": "..."}`.
        """
        from memo.memory import WriteRefused

        try:
            rec = memory.save(
                content=content,
                title=title,
                type_=type,
                tags=tags,
                auto_derive=auto_derive,
                extra=extra,
                respect_synapse_freeze=respect_synapse_freeze,
            )
        except WriteRefused as exc:
            return {
                "status": "refused",
                "conflict": exc.conflict,
                "message": str(exc),
            }
        return rec.to_dict()

    @server.tool()
    def memory_provenance(id: str) -> dict[str, Any] | None:
        """Return the full provenance trail for one memoria.

        Combines the current synapse_*/agent_* keys (subset of
        `meta.extra_json`) with the per-op history (each save/update
        carries its own provenance snapshot). Returns `None` if the id
        is unknown.

        Shape:

            {
              "id": "<full id>",
              "current": {synapse_trace_id, synapse_route_reason, ...},
              "events": [{"ts", "op", "title", "type", "provenance"}, ...]
            }
        """
        return memory.provenance(id)

    @server.tool()
    def memory_get_embedder_profile() -> dict[str, Any]:
        """Return the authoritative embedder profile for the trinity.

        Memo owns the embedding model + dimensions for the entire
        memo+memflow+synapse stack (M4). Other backends MUST read this
        profile at startup and refuse to operate when dims or model_id
        differ from their own caches — silent dimension mismatch is the
        single most common cause of "search returns nothing" failures.

        Returns ``consciousness_contracts.EmbedderProfile.to_dict()``
        when the contracts package is installed, else a memo-native
        fallback shape with the same field names. Field summary:

            {
              "schema": "consciousness.embedder_profile.v1",
              "model_id": "mlx-community/Qwen3-Embedding-...",
              "dims": 1024,
              "normalization": "l2",
              "max_seq_len": null,
              "quantization": null,
              "provider": "memo"
            }
        """
        cfg = memory.cfg
        try:
            from consciousness_contracts import EmbedderProfile

            profile = EmbedderProfile(
                model_id=cfg.embedder_model,
                dims=int(cfg.embedder_dims),
                normalization="l2",
                provider="memo",
            )
            return profile.to_dict()
        except ImportError:
            return {
                "schema": "consciousness.embedder_profile.v1",
                "model_id": cfg.embedder_model,
                "dims": int(cfg.embedder_dims),
                "normalization": "l2",
                "max_seq_len": None,
                "quantization": None,
                "provider": "memo",
            }

    @server.tool()
    def memory_unified_briefing(cwd: str | None = None, source: str = "") -> dict[str, Any]:
        """Return a synapse-aware briefing for the current focus.

        Pulls `synapse packet` (present_state + reality_conflicts) and
        composes it as markdown lines suitable for a SessionStart hook
        or any agent that wants a one-shot view of the consciousness
        stack. Falls back to an empty payload when synapse is
        unreachable — callers should layer the local `memo briefing`
        sections on top if they want a complete panel.

        Shape:

            {
              "available": bool,        # true iff synapse returned data
              "markdown": "<lines...>",  # empty when not available
              "lines": [str, ...],
            }
        """
        from memo.briefing import synapse_briefing_lines

        t0 = _now_ms()
        lines = synapse_briefing_lines(cwd)
        _log_consult(
            memory,
            tool="unified_briefing",
            query=cwd or "briefing",
            hits=[],
            t0_ms=t0,
            source=source,
        )
        return {
            "available": bool(lines),
            "markdown": "\n".join(lines),
            "lines": lines,
        }

    @server.tool()
    def memory_search(
        query: str,
        limit: int = 10,
        type: str | None = None,
        body_chars: int = 280,
        mode: str = "hybrid",
        source: str = "",
    ) -> list[dict[str, Any]]:
        """Top-k search — hybrid (semantic + keyword) by default.

        Returns records ordered by descending fused score. Each result
        has a `score` field.

        Args:
            query: Free-text query. Required, non-empty.
            limit: Max results. Defaults to 10.
            type: Optional filter by record type (e.g. only `decision`).
            body_chars: Truncate the `body` field to this many chars. The
                default keeps results compact for the LLM context — call
                `memory_get(id)` for the full body. Pass a very large
                number to disable truncation.
            mode: `hybrid` (default — RRF fusion of vec + bm25),
                `vec` (semantic only), or `bm25` (keyword only). Use
                `bm25` when looking up exact tag/path/code-snippet
                matches; the small embedder is unreliable on those.
        """
        t0 = _now_ms()
        out: list[dict[str, Any]] = []
        for r in memory.search(query, limit=limit, type_=type, mode=mode):
            d = r.to_dict()
            body = d.get("body") or ""
            if body_chars >= 0 and len(body) > body_chars:
                d["body"] = body[:body_chars].rstrip() + "…"
                d["body_truncated"] = True
            out.append(d)
        _log_consult(memory, tool="search", query=query, hits=out, t0_ms=t0, source=source)
        return out

    @server.tool()
    def memory_rerank(
        query: str,
        hits: list[dict[str, Any]],
        top_n: int | None = None,
        body_chars: int = 1200,
    ) -> list[dict[str, Any]]:
        """Rerank externally-supplied hits with memo's warm cross-encoder.

        Scores each ``{title, snippet|body, ...}`` hit against `query` using
        this server's already-loaded Qwen3-Reranker and returns the list
        reordered with a `rerank_score` per hit (original fields preserved).
        The warm-daemon equivalent of the `memo rerank` CLI — lets an external
        caller (Synapse `memo_ce`) avoid cold-loading the reranker per request.
        """
        return memory.rerank_hits(query, hits, top_n=top_n, body_chars=body_chars)

    @server.tool()
    def memory_embed_query(text: str) -> dict[str, Any]:
        """Embed a single query string with memo's MLX embedder.

        Returns `{vector, dim, model}`. Uses the **asymmetric query
        prefix** — appropriate for the query side of a cosine search.
        For symmetric (document) embedding use `memory_embed_batch`.

        Synapse calls this to unify the vector space across retrieval +
        rerank + HyDE (previously Synapse used its own Ollama embedder
        in a different space).
        """
        if not text or not text.strip():
            raise ValueError("memory_embed_query: empty text")
        vec = memory.embedder.embed_query(text)
        return {"vector": vec, "dim": len(vec), "model": memory.cfg.embedder_model}

    @server.tool()
    def memory_embed_batch(texts: list[str]) -> dict[str, Any]:
        """Batched **symmetric** embed for a list of texts.

        Returns `{vectors, dim, model}` where `vectors[i]` matches
        `texts[i]`. Use for document/snippet/hypothesis embedding (no
        query prefix). One call amortizes MLX inference; preferred over
        N individual `memory_embed_query` calls when callers have a
        batch in hand.
        """
        if not texts:
            return {"vectors": [], "dim": 0, "model": memory.cfg.embedder_model}
        vecs = memory.embedder.embed(texts)
        dim = len(vecs[0]) if vecs else 0
        return {"vectors": vecs, "dim": dim, "model": memory.cfg.embedder_model}

    @server.tool()
    def memory_ask(
        question: str,
        k: int = 5,
        type: str | None = None,
        snippet_chars: int = 800,
        include_repos: bool = True,
        source: str = "",
    ) -> dict[str, Any]:
        """RAG over the memory archive.

        Pipeline: hybrid search top-`k` → MLXChat 7B with citation prompt
        → returns `{question, answer, sources}`. The answer cites the
        memorias it used inline as `[id-prefix]`. If no relevant memorias
        exist, the answer literally says so (no hallucination).

        Args:
            question: Natural-language question.
            k: How many memorias to feed the LLM as context. Default 5.
            type: Optional filter to one record type before retrieval.
            snippet_chars: Cap each memoria's body in the prompt
                (default 800). Lower = cheaper + smaller context window
                used; higher = more grounding for long bodies.
        """
        t0 = _now_ms()
        res = memory.ask(
            question,
            k=k,
            type_=type,
            snippet_chars=snippet_chars,
            include_repos=include_repos,
        )
        out = res if isinstance(res, dict) else {"answer": str(res)}
        cites = out.get("citations") or out.get("sources") or []
        hit_dicts = [c for c in cites if isinstance(c, dict)]
        _log_consult(memory, tool="ask", query=question, hits=hit_dicts, t0_ms=t0, source=source)
        return out

    @server.tool()
    def memory_chat_ask(
        question: str,
        k: int = 7,
        type: str | None = None,
        history: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
        source: str = "",
    ) -> dict[str, Any]:
        """Chat-shaped RAG over the memory archive.

        This is the structured chat delivery contract for agents that
        need status, citations, and retrieval metadata in addition to the
        prose answer. It returns the `memo.chat_ask.v2` envelope:
        `{schema, question, answer, sources, citations, retrieval_trace,
        synthesis_status, synthesis_source, synthesis_error, total_ms,
        history_turns_used, context_keys}`.

        Args:
            question: Natural-language question.
            k: How many memorias to feed the LLM as context. Default 7.
            type: Optional filter to one record type before retrieval.
            history: Optional chat history as `{role, text}` or
                `{role, content}` turns. Only user/assistant turns are used.
            context: Optional caller context included in retrieval question.
            source: Calling layer name for memo's usefulness telemetry.
        """
        t0 = _now_ms()
        res = memory.chat_ask(
            question,
            k=k,
            type_=type,
            history=history,
            context=context,
        )
        out = res if isinstance(res, dict) else {"answer": str(res)}
        cites = out.get("citations") or out.get("sources") or []
        hit_dicts = [c for c in cites if isinstance(c, dict)]
        _log_consult(
            memory, tool="chat_ask", query=question, hits=hit_dicts, t0_ms=t0, source=source
        )
        return out

    @server.tool()
    def memory_list(
        limit: int = 20,
        type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Recent memories ordered by `updated` desc. No vector lookup
        — useful when you want to inspect "what did I save lately"
        without a search query."""
        return [r.to_dict() for r in memory.list(limit=limit, type_=type)]

    @server.tool()
    def memory_get(id: str) -> dict[str, Any] | None:
        """Fetch one memory record by id (full UUID hex or unique prefix
        ≥4 chars). Returns None when not found. On ambiguous prefix
        returns `{"error": "ambiguous", "matches": [...]}` instead of
        raising — keeps the MCP transport happy and lets the LLM read
        the candidates."""
        try:
            rec = memory.get(id)
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        if not rec:
            return None
        # Feedback loop: a full fetch is the strongest "this was useful" signal
        # we get. Feeds learned type/entity preferences back into recall ranking.
        with contextlib.suppress(Exception):
            memory.contextual.record_click(rec.id)
        return rec.to_dict()

    @server.tool()
    def memory_update(
        id: str,
        title: str | None = None,
        type: str | None = None,
        tags: list[str] | None = None,
        content: str | None = None,
    ) -> dict[str, Any] | None:
        """Patch fields on an existing memory. Re-embeds only when the
        body actually changed (saves a forward pass for retag/rename).

        Returns the updated record, or None if `id` is unknown. Tags
        replace the existing list (use `memory_get` first if you want to
        merge). On ambiguous prefix returns `{"error": "ambiguous",
        "matches": [...]}` so the LLM can surface candidates.
        """
        try:
            rec = memory.update(
                id,
                title=title,
                type_=type,
                tags=tags,
                content=content,
            )
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        return rec.to_dict() if rec else None

    @server.tool()
    def memory_reindex(force: bool = False) -> dict[str, int]:
        """Re-scan the memory dir, re-embed entries whose on-disk body
        diverged from the indexed `body_hash`. Picks up edits the user
        made to memory `.md` files in Obsidian.

        With `force=True`, re-embeds every indexed entry regardless of
        body_hash. Use after an embedder model swap or a change to the
        embed-input composition.

        Returns `{"checked", "reindexed", "added", "skipped"}`.
        """
        return memory.reindex(force=force)

    @server.tool()
    def memory_delete(id: str) -> dict[str, Any]:
        """Delete one memory by id (full or unique prefix). Removes both
        the vec entry and the backing `.md` file. Returns
        `{"deleted": true|false}` on success, or
        `{"error": "ambiguous", "matches": [...]}` if the prefix matches
        multiple records."""
        try:
            return {"deleted": memory.delete(id)}
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}

    @server.tool()
    def memory_forget(id: str, reason: str | None = None) -> dict[str, Any]:
        """Soft-forget one memory by id (full or unique prefix).

        Unlike `memory_delete`, this keeps the `.md` file and the index entry
        but excludes the memoria from search / recall / list by default. It is
        reversible with `memory_unforget`. Use this when a fact is obsolete or
        no longer wanted in context but you don't want to destroy it.

        Args:
            id: Memoria id (full UUID hex or unique prefix).
            reason: Optional free-text note on why it was forgotten.

        Returns `{"forgotten": true, "id": ...}` on success,
        `{"forgotten": false}` if the id is unknown, or
        `{"error": "ambiguous", "matches": [...]}` for an ambiguous prefix.
        """
        try:
            rec = memory.forget(id, reason=reason)
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        if rec is None:
            return {"forgotten": False}
        return {"forgotten": True, "id": rec.id}

    @server.tool()
    def memory_unforget(id: str) -> dict[str, Any]:
        """Reverse a `memory_forget`: make the memoria searchable again.

        Clears the `is_forgotten` flag (and any `forget_after` TTL) so the
        memoria reappears in search / recall / list.

        Args:
            id: Memoria id (full UUID hex or unique prefix).

        Returns `{"unforgotten": true, "id": ...}` on success,
        `{"unforgotten": false}` if the id is unknown, or
        `{"error": "ambiguous", "matches": [...]}` for an ambiguous prefix.
        """
        try:
            rec = memory.unforget(id)
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        if rec is None:
            return {"unforgotten": False}
        return {"unforgotten": True, "id": rec.id}

    @server.tool()
    def memory_consolidate(
        threshold: float = 0.85,
        max_clusters: int = 20,
        type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find clusters of near-duplicate memorias and propose actions.

        Read-only. Returns a list of clusters, each with `members`,
        `summary`, `relationship` (`duplicate`|`evolution`|`facets`|
        `unrelated`), and `rationale`. The user/agent reviews and
        decides whether to merge/delete via `memory_update` /
        `memory_delete`.

        Args:
            threshold: Cosine similarity floor (default 0.85).
            max_clusters: Cap LLM calls (default 20 largest clusters).
            type: Optional filter to one record type.
        """
        return memory.consolidate(
            threshold=threshold,
            max_clusters=max_clusters,
            type_=type,
        )

    @server.tool()
    def memory_lint() -> dict[str, list[dict[str, Any]]]:
        """Surface memorias with quality issues — `legacy_extra`,
        `few_tags`, `body_skinny`, `untitled`. Read-only; for the LLM
        to suggest a cleanup pass to the user."""
        return memory.lint()

    @server.tool()
    def memory_record_diff(id: str, limit: int = 50) -> dict[str, Any]:
        """Full edit history for one memoria with field-level diffs.

        Returns a chronological timeline of every save / update / delete
        event for `id`, with `delta` dicts showing `{field: [old, new]}`
        on each update. Use this to answer "when did this change?" or to
        review how a decision evolved.

        Args:
            id: Full or prefix id of the memoria (≥4 chars).
            limit: Max events to include. Default 50.
        """
        resolved_id = id
        if len(resolved_id) < 32:
            try:
                resolved_id = memory.resolve_id(resolved_id) or resolved_id
            except AmbiguousIdError as exc:
                return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        r = memory.get(resolved_id)
        events = memory.history.list_recent(limit=limit, record_id=resolved_id)
        events = list(reversed(events))  # chronological
        return {
            "id": resolved_id,
            "title": r.title if r else None,
            "type": r.type if r else None,
            "events": events,
            "total_events": len(events),
        }

    @server.tool()
    def memory_history(
        limit: int = 20,
        op: str | None = None,
        id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Recent save/update/delete events from the audit log.

        Args:
            limit: Max events. Defaults to 20.
            op: Optional filter: `save`, `update`, or `delete`.
            id: Optional filter to events for one record (full id or
                unique prefix ≥4 chars).
        """
        record_id = id
        if record_id and len(record_id) < 32:
            try:
                resolved = memory.resolve_id(record_id)
            except AmbiguousIdError as exc:
                return [{"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}]
            if resolved is None:
                return []
            record_id = resolved
        return memory.history.list_recent(limit=limit, op=op, record_id=record_id)

    @server.tool()
    def memory_session_list(
        limit: int = 10,
        project: str | None = None,
    ) -> list[dict[str, Any]]:
        """Recent Claude Code session snapshots ordered by `updated` desc.

        Each snapshot captures cwd / branch / last user prompt at the
        moment the Stop hook fired. Used by the `/memo resume` flow:
        list → user picks → load the snapshot's `last_user_msg` /
        `summary` as context to continue.

        Args:
            limit: Max sessions. Defaults to 10. Hard-capped by the
                LRU policy at 50.
            project: Optional basename filter (e.g. `"memo"`).

        Returns: list of `{session_id, project, branch, head_commit,
        last_user_msg, summary, transcript_path, created, updated,
        turn_count, modified_files, ...}`.
        """
        from memo.session import list_sessions

        return list_sessions(memory.cfg.state_dir, limit=limit, project=project)

    @server.tool()
    def memory_session_get(session_id: str) -> dict[str, Any] | None:
        """Fetch one session snapshot by id (full or unique prefix ≥4
        chars). Returns None when no match. Companion to
        `memory_session_list` for the `/memo resume` picker."""
        from memo.session import get_session

        return get_session(memory.cfg.state_dir, session_id)

    @server.tool()
    def memory_stats() -> dict[str, Any]:
        """Summary stats — total records, recent counts. No body load."""
        history_errors = 0
        with contextlib.suppress(Exception):
            history_errors = int(getattr(memory.history, "error_count", 0))
        stats: dict[str, Any] = {
            "total": memory.store.count(),
            "data_dir": str(memory.cfg.data_dir),
            "vault_path": (str(memory.cfg.vault_path) if memory.cfg.vault_path else None),
            "db_path": str(memory.cfg.db_path),
            "embedder_model": memory.cfg.embedder_model,
            "history_errors": history_errors,
        }
        # Recall health (surfaced/used proxy) so the "is memo useful?" signal
        # is visible over MCP too. Best-effort; never breaks stats.
        with contextlib.suppress(Exception):
            from memo.dashboard import recall_health

            stats["recall_health"] = recall_health(memory.cfg.state_dir)
        return stats

    # NB: lifecycle / suggest / federation are intentionally NOT exposed as
    # @server.tool() — "brain-like" verbs stay off memo's MCP surface (memo is
    # the store, not the cognition layer; see CLAUDE.md + the architecture-
    # boundary test). The previously-dead helper stubs for them were removed.

    # -- time-machine (as-of) ---------------------------------------------------

    @server.resource("memo://recent")
    def _resource_recent() -> str:
        """Most-recent memorias (top 20 by `updated` desc), formatted as
        a markdown index with `memo://memory/<id>` links. Refreshes on
        every read."""
        recs = memory.list(limit=20)
        if not recs:
            return "# memo · recent\n\n_(no memorias yet)_\n"
        out = ["# memo · recent", ""]
        for r in recs:
            tags = ", ".join(r.tags) if r.tags else ""
            out.append(
                f"- **[{r.id[:8]}]** [{r.title}](memo://memory/{r.id}) "
                f"_{r.type}_{(' · ' + tags) if tags else ''}",
            )
        return "\n".join(out) + "\n"

    @server.resource("memo://memory/{id}")
    def _resource_memory(id: str) -> str:
        """Single memoria by id (full prefix or any unique ≥4 char
        prefix). Returned as markdown with the frontmatter inlined as a
        header so the client renders cleanly."""
        try:
            rec = memory.get(id)
        except AmbiguousIdError as exc:
            return f"# Ambiguous id `{id}`\n\nMatches:\n\n" + "\n".join(
                f"- `{m}`" for m in exc.matches
            )
        if rec is None:
            return f"# Not found\n\nNo memoria for id `{id}`.\n"
        tags = ", ".join(rec.tags) if rec.tags else "—"
        return (
            f"# {rec.title}\n\n"
            f"- **id:** `{rec.id}`\n"
            f"- **type:** {rec.type}\n"
            f"- **tags:** {tags}\n"
            f"- **created:** {rec.created}\n"
            f"- **updated:** {rec.updated}\n\n"
            f"---\n\n{rec.body or ''}"
        )

    @server.custom_route("/chat/stream", methods=["POST"])
    async def chat_stream_route(request):  # type: ignore[no-untyped-def]
        """Real token streaming for chat synthesis (SSE).

        MCP `tools/call` can only return a single result, so the warm-daemon
        chat path otherwise has to buffer the whole answer before the client
        sees anything. This route exposes `Memory.chat_ask_stream` directly:
        emits one SSE `data:` line per event (context → token* → done), so the
        first token reaches the caller right after prefill instead of after the
        full decode. Output is identical to the non-streaming tool; only the
        delivery is incremental. Consumed by Synapse's MemoBackend.
        """
        import json as _json

        from starlette.responses import StreamingResponse

        try:
            body = await request.json()
        except Exception:
            from starlette.responses import JSONResponse

            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        question = str(body.get("question") or "")
        if not question.strip():
            from starlette.responses import JSONResponse

            return JSONResponse({"error": "empty question"}, status_code=400)
        k = int(body.get("k") or 7)
        type_ = body.get("type") or None
        history = body.get("history") or None
        context = body.get("context") or None

        def _gen():
            try:
                for ev in memory.chat_ask_stream(
                    question,
                    k=k,
                    type_=type_,
                    history=history,
                    context=context,
                ):
                    yield f"data: {_json.dumps(ev, ensure_ascii=False)}\n\n"
            except Exception as exc:  # never hang the client mid-stream
                err = {"event": "error", "message": f"{type(exc).__name__}: {exc}"}
                yield f"data: {_json.dumps(err, ensure_ascii=False)}\n\n"

        # Starlette drives the sync generator in a worker thread; the MLX
        # forward is GIL/Metal-bound and already serialized, so one stream at
        # a time is fine for the single-user daemon.
        return StreamingResponse(_gen(), media_type="text/event-stream")

    return server


def main() -> None:
    """Entry point for `memo-mcp` console script.

    Default transport is stdio (one client per process — Claude Code, Codex,
    etc.). Set ``MEMO_MCP_TRANSPORT=http`` (or ``streamable-http``/``sse``) to
    run as a long-lived HTTP daemon instead, so an external service can call
    the warm tools (e.g. ``memory_chat_ask``) without paying the ~per-process
    MLX cold-load. ``MEMO_MCP_HOST``/``MEMO_MCP_PORT`` pick the bind (default
    127.0.0.1:18768). One ``Memory`` instance is built here and reused across
    every request, so the embedder / reranker / synthesis LLM stay resident.
    """
    server = build_server()
    transport = os.environ.get("MEMO_MCP_TRANSPORT", "stdio").strip().lower()
    if transport in ("http", "streamable-http", "sse"):
        # Enable prompt cache + query embedding cache for long-lived daemon
        os.environ.setdefault("MEMO_PROMPT_CACHE", "1")
        os.environ.setdefault("MEMO_QUERY_CACHE_SIZE", "500")
        host = os.environ.get("MEMO_MCP_HOST", "127.0.0.1").strip() or "127.0.0.1"
        try:
            port = int(os.environ.get("MEMO_MCP_PORT", "18768").strip() or "18768")
        except ValueError:
            port = 18768  # malformed env → default rather than crash the daemon
        # transport is validated against the allowed set just above.
        server.run(transport=cast(Any, transport), host=host, port=port)
    else:
        server.run()


if __name__ == "__main__":
    main()
