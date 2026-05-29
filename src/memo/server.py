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

import os
from typing import Any

from fastmcp import FastMCP

from memo.config import Config
from memo.memory import AmbiguousIdError, Memory


def build_server(memory: Memory | None = None) -> FastMCP:
    """Build the MCP server. Accepts an explicit `Memory` for tests.

    The default constructs from `Config.from_env()` — production runs
    pick up `MEMO_*` env vars set by the calling shell or by Claude
    Code's `claude mcp add` invocation.
    """
    if memory is None:
        memory = Memory(Config.from_env())

    server = FastMCP("memo")

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
                content=content, title=title, type_=type, tags=tags,
                auto_derive=auto_derive, extra=extra,
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
    def memory_unified_briefing(cwd: str | None = None) -> dict[str, Any]:
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

        lines = synapse_briefing_lines(cwd)
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
        out: list[dict[str, Any]] = []
        for r in memory.search(query, limit=limit, type_=type, mode=mode):
            d = r.to_dict()
            body = d.get("body") or ""
            if body_chars >= 0 and len(body) > body_chars:
                d["body"] = body[:body_chars].rstrip() + "…"
                d["body_truncated"] = True
            out.append(d)
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
        return memory.ask(
            question,
            k=k,
            type_=type,
            snippet_chars=snippet_chars,
            include_repos=include_repos,
        )

    @server.tool()
    def memory_chat_ask(
        question: str,
        k: int = 7,
        type: str | None = None,
        history: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
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
        """
        return memory.chat_ask(
            question,
            k=k,
            type_=type,
            history=history,
            context=context,
        )

    @server.tool()
    def memory_list(
        limit: int = 20, type: str | None = None,
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
        return rec.to_dict() if rec else None

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
                id, title=title, type_=type, tags=tags, content=content,
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
    def memory_repo_index(
        url: str,
        name: str | None = None,
        ref: str | None = None,
        force: bool = False,
        with_embeddings: bool = True,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        max_file_bytes: int | None = None,
    ) -> dict[str, Any]:
        """Clone/fetch a Git repo and index all included text files.

        Auth is delegated to local `git`: public repos work directly;
        private repos work when SSH agent / credential helpers / tokens
        already let `git clone <url>` succeed on this machine.
        """
        return memory.repo_index(
            url,
            name=name,
            ref=ref,
            force=force,
            with_embeddings=with_embeddings,
            include=include,
            exclude=exclude,
            max_file_bytes=max_file_bytes,
        )

    @server.tool()
    def memory_repo_embed(repo: str, force: bool = False) -> dict[str, Any]:
        """Embed pending repo chunks. Runs automatically during repo index by default."""
        return memory.repo_embed(repo, force=force)

    @server.tool()
    def memory_repo_status(repo: str) -> dict[str, Any] | None:
        """Return exact and semantic index counts for one repo."""
        return memory.repo_status(repo)

    @server.tool()
    def memory_repo_search(
        query: str,
        limit: int = 10,
        repo: str | None = None,
        path: str | None = None,
        mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """Search indexed repositories.

        Modes: `hybrid` fuses chunk embeddings, chunk BM25, and line
        BM25; `vec` is semantic chunk search; `bm25` is keyword chunk
        search; `line` searches the exact per-line index.
        """
        return [
            hit.to_dict()
            for hit in memory.repo_search(
                query, limit=limit, repo=repo, path=path, mode=mode,
            )
        ]

    @server.tool()
    def memory_repo_get_file(
        repo: str,
        path: str,
        start: int | None = None,
        end: int | None = None,
    ) -> dict[str, Any] | None:
        """Return indexed text for one repo file or line range."""
        return memory.repo_get_file(repo, path, start=start, end=end)

    @server.tool()
    def memory_repo_list(limit: int = 100) -> list[dict[str, Any]]:
        """List indexed repositories."""
        return memory.repo_list(limit=limit)

    @server.tool()
    def memory_repo_delete(repo: str, remove_clone: bool = True) -> dict[str, bool]:
        """Delete one indexed repo and optionally remove memo's managed clone."""
        return {"deleted": memory.repo_delete(repo, remove_clone=remove_clone)}

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
    def memory_extract_entities(
        ids: list[str] | None = None, all_: bool = False, force: bool = False,
    ) -> dict[str, int]:
        """Extract named entities (person/project/technology/file/org/concept)
        from memoria bodies via Qwen2.5-3B and write them to the graph DB.

        Args:
            ids: Specific memoria ids to process (full UUID hex). Mutually
                exclusive with `all_`.
            all_: Process every memoria in the store.
            force: Re-extract even if entity links already exist
                (default skips already-indexed memorias).

        Returns counts: `{processed, entities_extracted, links_written, skipped, errors}`.
        Cost: ~0.5-1s per memoria. Use `all_=True` once after a fresh
        install, then incrementally on new memorias.
        """
        return memory.extract_entities(
            ids=ids, all_=all_, skip_already_indexed=not force,
        )

    @server.tool()
    def memory_entities(
        limit: int = 30, type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Top entities in the knowledge graph, ranked by mention count.

        Args:
            limit: Max entities. Default 30.
            type: Optional filter (`person`/`project`/`technology`/
                `file`/`org`/`concept`).
        """
        return memory.graph.top_entities(limit=limit, type_=type)

    @server.tool()
    def memory_entity(name: str, type: str | None = None) -> list[str]:
        """Memoria IDs that mention `name` (and optionally a specific
        entity type). Returns a list of full UUIDs."""
        return memory.graph.entity_memorias(name, type_=type)

    @server.tool()
    def memory_consolidate(
        threshold: float = 0.85, max_clusters: int = 20, type: str | None = None,
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
            threshold=threshold, max_clusters=max_clusters, type_=type,
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
        limit: int = 20, op: str | None = None, id: str | None = None,
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
        limit: int = 10, project: str | None = None,
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
        try:
            history_errors = int(getattr(memory.history, "error_count", 0))
        except Exception:
            pass
        return {
            "total": memory.store.count(),
            "data_dir": str(memory.cfg.data_dir),
            "vault_path": (
                str(memory.cfg.vault_path) if memory.cfg.vault_path else None
            ),
            "db_path": str(memory.cfg.db_path),
            "embedder_model": memory.cfg.embedder_model,
            "history_errors": history_errors,
        }

    # -- temporal reasoning tools ----------------------------------------------

    @server.tool()
    def memory_temporal_contradictions(
        entity: str,
        entity_type: str | None = None,
        confidence_threshold: float = 0.7,
        max_pairs: int = 20,
    ) -> list[dict[str, Any]]:
        """Detect contradictions among memorias mentioning a specific entity.

        Uses the helper LLM to classify pairs of memorias as contradiction,
        evolution, consistent, or unrelated. Returns only contradictions and
        evolutions above the confidence threshold.

        Args:
            entity: The entity name to analyze (e.g. "ollama", "mlx").
            entity_type: Optional entity type filter from graph.
            confidence_threshold: Minimum confidence (0-1). Default 0.7.
            max_pairs: Maximum number of pairs to analyze (LLM is expensive).
        """
        contradictions = memory.temporal.detect_entity_contradictions(
            entity_name=entity,
            entity_type=entity_type,
            confidence_threshold=confidence_threshold,
            max_pairs=max_pairs,
        )
        return [c.__dict__ for c in contradictions]

    @server.tool()
    def memory_temporal_timeline(
        entity: str,
        entity_type: str | None = None,
    ) -> dict[str, Any] | None:
        """Build a chronological timeline of all memorias mentioning an entity.

        Returns a timeline with events ordered by date, including first/last
        seen timestamps. Useful for tracking evolution of decisions or
        opinions over time.

        Args:
            entity: The entity name to analyze.
            entity_type: Optional entity type filter from graph.
        """
        timeline = memory.temporal.build_entity_timeline(
            entity_name=entity,
            entity_type=entity_type,
        )
        if timeline is None:
            return None
        return {
            "entity_name": timeline.entity_name,
            "entity_type": timeline.entity_type,
            "first_seen": timeline.first_seen,
            "last_seen": timeline.last_seen,
            "events": [e.__dict__ for e in timeline.events],
        }

    @server.tool()
    def memory_temporal_stale(
        days_threshold: int = 180,
        min_access_count: int = 0,
    ) -> list[dict[str, Any]]:
        """Find memorias that may be stale based on age and lack of access.

        Returns potentially stale memorias with metadata including days since
        update and access count. Useful for identifying outdated information
        that may need review.

        Args:
            days_threshold: Days since last update to consider stale.
            min_access_count: Minimum access count to exclude (frequently-accessed
                old memorias may still be relevant).
        """
        return memory.temporal.detect_stale_memorias(
            days_threshold=days_threshold,
            min_access_count=min_access_count,
        )

    @server.tool()
    def memory_temporal_patterns() -> dict[str, Any]:
        """Analyze high-level temporal patterns across the entire corpus.

        Returns metrics including:
        - memorias_per_month: histogram of creation activity
        - type_distribution_over_time: how memory types change over time
        - most_active_entities: entities with most temporal churn
        """
        return memory.temporal.detect_temporal_patterns()

    # -- advanced consolidation tools -------------------------------------------

    # -- contradiction radar (persistent triage) -------------------------------

    @server.tool()
    def memory_contradict_scan(
        top_k: int = 5,
        sim_floor: float = 0.55,
        confidence_threshold: float = 0.7,
        min_days_apart: int = 1,
        max_memorias: int = 2000,
        max_pairs: int = 500,
        since: str | None = None,
        type: str | None = None,
    ) -> dict[str, Any]:
        """Scan the corpus for contradiction / evolution pairs and persist them.

        Unlike `memory_temporal_contradictions`, which requires an entity name
        and returns ephemeral results, this walks all memorias, uses vec
        neighborhoods to surface candidate pairs, and persists detected
        contradictions to a sidecar DB. The same pair is never re-classified
        once the user resolves it.

        Args:
            top_k: Neighbors to consider per memoria.
            sim_floor: Cosine floor for candidate pairs.
            confidence_threshold: Min LLM confidence to persist.
            min_days_apart: Skip pairs whose `updated` are within N days.
            max_memorias: Hard cap on memorias visited.
            max_pairs: Hard cap on pairs sent to the LLM.
            since: ISO date; only memorias updated on/after are scanned.
            type: Optional memoria type filter.
        """
        result = memory.contradict_scanner.scan_corpus(
            top_k=top_k,
            sim_floor=sim_floor,
            confidence_threshold=confidence_threshold,
            min_days_apart=min_days_apart,
            max_memorias=max_memorias,
            max_pairs=max_pairs,
            since=since,
            type_=type,
        )
        return {
            "scanned_memorias": result.scanned_memorias,
            "pairs_examined": result.pairs_examined,
            "pairs_inserted": result.pairs_inserted,
            "pairs_refreshed": result.pairs_refreshed,
            "pairs_skipped_resolved": result.pairs_skipped_resolved,
            "contradictions_found": result.contradictions_found,
            "evolutions_found": result.evolutions_found,
        }

    @server.tool()
    def memory_contradict_list(
        status: str = "open",
        limit: int = 20,
        min_confidence: float = 0.0,
        relationship: str | None = None,
    ) -> list[dict[str, Any]]:
        """List contradiction pairs from the sidecar DB.

        Args:
            status: One of open|fused|kept_newer|kept_older|evolved|dismissed.
            limit: Max rows.
            min_confidence: Filter to pairs at/above this LLM confidence.
            relationship: Optional 'contradiction' or 'evolution' filter.
        """
        if status == "open":
            pairs = memory.contradict_store.list_open(
                limit=limit,
                min_confidence=min_confidence,
                relationship=relationship,
            )
        else:
            pairs = memory.contradict_store.list_all(status=status, limit=limit)
            if relationship:
                pairs = [p for p in pairs if p.relationship == relationship]
        return [p.__dict__ for p in pairs]

    @server.tool()
    def memory_contradict_resolve(
        pair_id: int,
        status: str,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Mark a contradiction pair as resolved.

        Valid statuses (excluding `open`):
          - `fused`       — both merged into a new memoria
          - `kept_newer`  — newer side won, older deleted/archived
          - `kept_older`  — older side won
          - `evolved`     — legitimate evolution, both kept
          - `dismissed`   — false positive

        This tool only updates the sidecar; it does NOT itself delete the
        memorias or run a merge. Use `memory_consolidate_apply` or
        `memory_delete` first if the resolution implies destructive ops.

        Args:
            pair_id: The integer pair_id returned by `memory_contradict_list`.
            status: One of the valid statuses listed above.
            note: Optional free-form resolution note.
        """
        ok = memory.contradict_store.resolve(pair_id, status, note=note)
        return {"updated": ok, "pair_id": pair_id, "status": status}

    @server.tool()
    def memory_contradict_stats() -> dict[str, int]:
        """Return counts of contradiction pairs grouped by status."""
        return memory.contradict_store.stats()

    @server.tool()
    def memory_consolidate_propose(
        threshold: float = 0.85,
        max_clusters: int = 20,
        type: str | None = None,
    ) -> dict[str, Any]:
        """Detect clusters and propose merge strategies (read-only).

        Returns a dict with detected clusters and merge proposals. Does not
        modify the corpus. Use `memory_consolidate_apply` to execute merges.

        Args:
            threshold: Cosine similarity threshold (default 0.85).
            max_clusters: Maximum clusters to process (default 20).
            type: Optional filter by memoria type.
        """
        return memory.consolidator.consolidate_all(
            threshold=threshold,
            max_clusters=max_clusters,
            type_=type,
            auto_apply=False,
            dry_run=True,
        )

    @server.tool()
    def memory_consolidate_apply(
        threshold: float = 0.85,
        max_clusters: int = 20,
        type: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Apply merge proposals to consolidate the corpus.

        Executes the consolidation pipeline: detect clusters, propose merges,
        and apply them. Archives old memorias to an `archived/` subdirectory.

        Args:
            threshold: Cosine similarity threshold (default 0.85).
            max_clusters: Maximum clusters to process (default 20).
            type: Optional filter by memoria type.
            dry_run: If True, show what would happen without applying changes.
        """
        return memory.consolidator.consolidate_all(
            threshold=threshold,
            max_clusters=max_clusters,
            type_=type,
            auto_apply=True,
            dry_run=dry_run,
        )

    @server.tool()
    def memory_consolidate_list_archived() -> list[dict[str, Any]]:
        """List all archived memorias.

        Returns a list of archived memoria entries with metadata including
        the replacement memoria ID and archival timestamp.
        """

        import frontmatter

        archival_dir = memory.cfg.memory_dir / "archived"
        if not archival_dir.is_dir():
            return []

        archived = []
        for f in archival_dir.glob("*.md"):
            post = frontmatter.loads(f.read_text(encoding="utf-8"))
            archived.append({
                "id": f.stem,
                "title": post.get("title", ""),
                "archived_for": post.get("archived_for", ""),
                "archived_at": post.get("archived_at", ""),
            })
        return archived

    # -- graph navigation tools ------------------------------------------------

    @server.tool()
    def memory_graph_path(
        source: str,
        target: str,
        max_length: int = 5,
    ) -> dict[str, Any] | None:
        """Find shortest path between two entities in the entity graph.

        Uses BFS to find the shortest path. Two entities are connected if
        they share a memoria. Returns the path including intermediate entities.

        Args:
            source: Source entity name.
            target: Target entity name.
            max_length: Maximum path length to search.
        """
        path = memory.navigator.find_shortest_path(source, target, max_length=max_length)
        return path.__dict__ if path else None

    @server.tool()
    def memory_graph_neighbors(
        entity: str,
        max_neighbors: int = 50,
    ) -> dict[str, Any]:
        """Get direct neighbors of an entity in the graph.

        Returns entities directly connected to the given entity, along with
        the memorias that connect them.

        Args:
            entity: Entity name.
            max_neighbors: Maximum neighbors to return.
        """
        neighbors = memory.navigator.get_neighbors(entity, max_neighbors=max_neighbors)
        return neighbors.__dict__

    @server.tool()
    def memory_graph_communities(
        min_size: int = 2,
    ) -> list[dict[str, Any]]:
        """Detect communities (connected components) in the entity graph.

        Uses connected components to find clusters of related entities.
        Useful for discovering thematic clusters in the knowledge graph.

        Args:
            min_size: Minimum community size to include.
        """
        communities = memory.navigator.detect_communities(min_size=min_size)
        return [c.__dict__ for c in communities]

    @server.tool()
    def memory_graph_centrality(
        top: int = 20,
    ) -> dict[str, Any]:
        """Compute centrality metrics for all entities.

        Returns degree centrality (number of connections) and betweenness
        centrality (how often entity lies on shortest paths). Useful for
        identifying hub entities in the graph.

        Args:
            top: Return top N entities by degree centrality.
        """
        scores = memory.navigator.compute_centrality()
        sorted_by_degree = sorted(scores.degree.items(), key=lambda x: x[1], reverse=True)[:top]
        return {
            "top_entities": [
                {"entity": e, "degree": d, "betweenness": scores.betweenness.get(e, 0.0)}
                for e, d in sorted_by_degree
            ],
            "total_entities": len(scores.degree),
        }

    @server.tool()
    def memory_graph_export(
        format: str = "dot",
        include_memorias: bool = False,
    ) -> dict[str, Any]:
        """Export the entity graph for visualization.

        Returns graph data in the specified format. Use with external tools
        like Graphviz (dot format) or web visualization libraries (JSON format).

        Args:
            format: Either "dot" for Graphviz DOT format or "json" for web UI.
            include_memorias: If True and format is "json", include memoria IDs in edge data.
        """
        if format == "dot":
            dot = memory.navigator.export_graphviz()
            return {"format": "dot", "content": dot}
        else:
            data = memory.navigator.export_json(include_memorias=include_memorias)
            return {"format": "json", "data": data}

    # -- contextual recall tools -------------------------------------------------

    @server.tool()
    def memory_contextual_search(
        query: str,
        limit: int = 10,
        mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """Search with contextual re-ranking based on conversation history.

        Uses conversation context and learned user preferences to re-rank
        search results. Boosts memories that overlap with recent context
        and aligns with user preferences.

        Args:
            query: Search query.
            limit: Max results.
            mode: Search mode (vec, bm25, hybrid).
        """
        results = memory.contextual.search_with_context(
            query=query,
            limit=limit,
            mode=mode,
        )
        return [r.__dict__ for r in results]

    @server.tool()
    def memory_contextual_record_search(
        query: str,
        memoria_ids: list[str],
    ) -> dict[str, Any]:
        """Record a search in the conversation history for learning.

        Use this after each search to build context for future searches.
        The system learns from which memories are recalled to improve
        future contextual ranking.

        Args:
            query: The search query that was used.
            memoria_ids: List of memoria IDs that were recalled.
        """
        memory.contextual.record_search(query, memoria_ids)
        return {"status": "recorded", "count": len(memoria_ids)}

    @server.tool()
    def memory_contextual_record_click(
        memoria_id: str,
    ) -> dict[str, Any]:
        """Record that the user clicked/viewed a memoria (for preference learning).

        Use this when the user explicitly selects a memoria from search results.
        This teaches the system which memory types and entities the user prefers.

        Args:
            memoria_id: The memoria ID that was clicked/viewed.
        """
        memory.contextual.record_click(memoria_id)
        return {"status": "recorded", "memoria_id": memoria_id}

    @server.tool()
    def memory_contextual_preferences() -> dict[str, Any]:
        """Show learned user preferences for memory recall.

        Returns the current preference scores for memory types, entities,
        and recency/diversity weights. Useful for understanding what the
        system has learned about the user's preferences.
        """
        prefs = memory.contextual.context.get_preferences()
        return prefs.__dict__

    @server.tool()
    def memory_contextual_history(
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Show recent conversation history used for contextual recall.

        Returns the N most recent prompts with their recalled memorias.
        This history is used to build context for search re-ranking.

        Args:
            limit: Number of recent prompts to return.
        """
        history = memory.contextual.context.get_recent_context(n=limit)
        return [c.__dict__ for c in history]

    # -- cross-reference tools ----------------------------------------------------

    @server.tool()
    def memory_links_backlinks(
        memoria_id: str,
    ) -> list[dict[str, Any]]:
        """Show all memorias that reference this one.

        Returns backlinks (incoming links) to the specified memoria.
        Useful for understanding how a memoria is connected to others.

        Args:
            memoria_id: The memoria ID to find backlinks for.
        """
        backlinks = memory.crossref.get_backlinks(memoria_id)
        return [b.__dict__ for b in backlinks]

    @server.tool()
    def memory_links_outlinks(
        memoria_id: str,
    ) -> list[dict[str, Any]]:
        """Show all memorias that this one references.

        Returns outlinks (outgoing links) from the specified memoria.
        Useful for understanding what a memoria connects to.

        Args:
            memoria_id: The memoria ID to find outlinks for.
        """
        outlinks = memory.crossref.get_outlinks(memoria_id)
        return [o.__dict__ for o in outlinks]

    @server.tool()
    def memory_links_suggest(
        content: str,
        title: str | None = None,
        tags: list[str] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Suggest links to existing memorias based on content.

        Returns suggested memorias to link to, based on semantic similarity
        and other heuristics. Useful when saving a new memoria to discover
        related existing content.

        Args:
            content: The memoria content being saved.
            title: Optional title of the memoria.
            tags: Optional tags of the memoria.
            limit: Maximum suggestions to return.
        """
        suggestions = memory.link_suggester.suggest_links(
            content=content,
            title=title or "",
            tags=tags or [],
            limit=limit,
        )
        return [s.__dict__ for s in suggestions]

    @server.tool()
    def memory_links_format(
        memoria_id: str,
        title: str | None = None,
    ) -> str:
        """Format a memoria ID as a wikilink.

        Returns a wikilink string like [[memoria-id]] or [[memoria-id|Title]].
        Use this to insert links into memoria content.

        Args:
            memoria_id: The memoria ID to format as a wikilink.
            title: Optional display title for the link.
        """
        return memory.link_suggester.format_wikilink(memoria_id, title)

    # -- lifecycle management helpers ---------------------------------------------

    def memory_lifecycle_report(
        limit: int = 100,
    ) -> dict[str, Any]:
        """Generate a lifecycle report on the corpus.

        Returns statistics on archival candidates, promotion/demotion candidates,
        expiration candidates, and access patterns. Useful for understanding
        the health of the memory corpus.

        Args:
            limit: Maximum memorias to analyze.
        """
        return memory.lifecycle.get_lifecycle_report(limit=limit)

    def memory_lifecycle_apply(
        dry_run: bool = False,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Apply lifecycle rules to the corpus.

        Archives inactive memorias, expires temporary memories, and reports
        promotion/demotion candidates. Use dry_run=True first to preview.

        Args:
            dry_run: If True, only report what would happen without applying changes.
            limit: Maximum memorias to process.
        """
        return memory.lifecycle.apply_lifecycle_rules(dry_run=dry_run, limit=limit)

    def memory_lifecycle_access_count(
        memoria_id: str,
    ) -> dict[str, Any]:
        """Show access count for a specific memoria.

        Returns the number of times a memoria has been accessed (non-save operations).

        Args:
            memoria_id: The memoria ID to check.
        """
        count = memory.lifecycle.get_access_count(memoria_id)
        return {"memoria_id": memoria_id, "access_count": count}

    # -- proactive suggestions helpers --------------------------------------------

    def memory_suggest_analyze(
        recent_turns: list[dict[str, str]],
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Analyze recent conversation turns and suggest memories to save.

        Uses the helper LLM to analyze conversation context and suggest
        potential memories to save, including title, type, tags, and confidence.

        Args:
            recent_turns: List of {"user": "...", "assistant": "..."} turns.
            limit: Maximum suggestions to return.
        """
        suggestions = memory.proactive.analyze_conversation(recent_turns, limit=limit)
        return [s.__dict__ for s in suggestions]

    def memory_suggest_feedback_stats() -> dict[str, Any]:
        """Show statistics on suggestion feedback (acceptance rate).

        Returns the total number of suggestions, accepted/rejected counts,
        and the acceptance rate. Useful for understanding how well the
        proactive suggestion system is working.
        """
        return memory.proactive.get_feedback_stats()

    # -- versioning tools --------------------------------------------------------

    @server.tool()
    def memory_version_history(
        memoria_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Show version history for a memoria.

        Returns a list of all versions of the specified memoria,
        most recent first. Each version includes title, type, tags,
        body snapshot, and change reason.

        Args:
            memoria_id: The memoria ID to get history for.
            limit: Maximum versions to return.
        """
        versions = memory.versioning.get_version_history(memoria_id, limit=limit)
        return [v.__dict__ for v in versions]

    @server.tool()
    def memory_version_diff(
        memoria_id: str,
        version_a: int | None = None,
        version_b: int | None = None,
    ) -> dict[str, Any] | None:
        """Show diff between two versions of a memoria.

        Returns a unified diff between two versions. If version_a or version_b
        is None, uses the latest and latest-1 versions respectively.

        Args:
            memoria_id: The memoria ID to diff.
            version_a: First version ID (or None for latest).
            version_b: Second version ID (or None for latest-1).
        """
        diff = memory.versioning.diff_versions(memoria_id, version_a, version_b)
        return diff.__dict__ if diff else None

    @server.tool()
    def memory_version_rollback(
        memoria_id: str,
        version_id: int,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Rollback a memoria to a previous version.

        Restores the memoria to the specified version by updating its
        content, title, type, and tags to match the version snapshot.

        Args:
            memoria_id: The memoria ID to rollback.
            version_id: The version ID to rollback to.
            reason: Optional reason for the rollback.
        """
        success = memory.versioning.rollback_to_version(memoria_id, version_id, reason)
        return {"success": success, "memoria_id": memoria_id, "version_id": version_id}

    # -- query composition tools --------------------------------------------------

    @server.tool()
    def memory_query_save(
        name: str,
        query_text: str,
        type_filter: str | None = None,
        tags_filter: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        search_mode: str = "hybrid",
        limit: int = 10,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Save a query for reuse.

        Stores a query with its parameters so it can be executed later
        by name. Useful for frequently-used complex searches.

        Args:
            name: Query name (unique).
            query_text: The search query text.
            type_filter: Optional type filter.
            tags_filter: Optional tag filter.
            date_from: Optional start date (ISO format).
            date_to: Optional end date (ISO format).
            search_mode: Search mode (vec, bm25, hybrid).
            limit: Result limit.
            description: Optional description.
        """
        memory.query_composer.query_store.save_query(
            name=name,
            query_text=query_text,
            type_filter=type_filter,
            tags_filter=tags_filter,
            date_from=date_from,
            date_to=date_to,
            search_mode=search_mode,
            limit=limit,
            description=description,
        )
        return {"status": "saved", "name": name}

    @server.tool()
    def memory_query_list() -> list[dict[str, Any]]:
        """List all saved queries.

        Returns a list of all saved queries with their parameters.
        Useful for discovering available queries to execute.
        """
        queries = memory.query_composer.query_store.list_queries()
        return [q.__dict__ for q in queries]

    @server.tool()
    def memory_query_run(
        name: str,
    ) -> dict[str, Any]:
        """Execute a saved query.

        Executes a previously saved query by name and returns the results.

        Args:
            name: The name of the saved query to execute.
        """
        query = memory.query_composer.query_store.get_query(name)
        if not query:
            return {"error": "Query not found", "name": name}

        result = memory.query_composer.execute_query(query)
        return {
            "query_name": result.query_name,
            "count": result.count,
            "executed_at": result.executed_at,
            "results": [r.__dict__ for r in result.results],
        }

    @server.tool()
    def memory_query_delete(
        name: str,
    ) -> dict[str, Any]:
        """Delete a saved query.

        Removes a saved query by name.

        Args:
            name: The name of the query to delete.
        """
        success = memory.query_composer.query_store.delete_query(name)
        return {"success": success, "name": name}

    # -- federation helpers -------------------------------------------------------

    def memory_federation_add_vault(
        name: str,
        path: str,
        weight: float = 1.0,
    ) -> dict[str, Any]:
        """Add a vault to the federation.

        Adds a new vault to the federation configuration so it can be
        included in cross-vault searches.

        Args:
            name: Vault name (unique).
            path: Absolute path to vault data_dir.
            weight: Weight for result ranking (default: 1.0).
        """
        memory.federation.config.add_vault(name, path, weight)
        return {"status": "added", "name": name}

    def memory_federation_list_vaults() -> list[dict[str, Any]]:
        """List all configured vaults in the federation.

        Returns a list of all vaults with their configuration including
        path, weight, and enabled status.
        """
        vaults = memory.federation.config.list_vaults()
        return [v.__dict__ for v in vaults]

    def memory_federation_remove_vault(
        name: str,
    ) -> dict[str, Any]:
        """Remove a vault from the federation.

        Removes a vault from the federation configuration.

        Args:
            name: The name of the vault to remove.
        """
        success = memory.federation.config.remove_vault(name)
        return {"success": success, "name": name}

    def memory_federation_search(
        query: str,
        limit: int = 10,
        mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """Search across all federated vaults.

        Searches across all enabled vaults in the federation and
        aggregates results with deduplication and vault-weighted ranking.

        Args:
            query: Search query.
            limit: Max results per vault before aggregation.
            mode: Search mode (vec, bm25, hybrid).
        """
        results = memory.federation.search(query, limit=limit, mode=mode)
        return [r.__dict__ for r in results]

    # -- sync & backup tools -------------------------------------------------------

    @server.tool()
    def memory_backup_create(
        compress: bool = True,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Create a backup of the entire vault.

        Creates a compressed tar.gz archive of all memoria files and
        databases. Returns metadata including checksum and size.

        Args:
            compress: Whether to compress the backup.
            name: Optional backup name (defaults to timestamp).
        """
        metadata = memory.backup.create_backup(compress=compress, name=name)
        return metadata.__dict__

    @server.tool()
    def memory_backup_list() -> list[dict[str, Any]]:
        """List all available backups.

        Returns a list of all backup archives with their metadata
        including timestamp and size.
        """
        backups = memory.backup.list_backups()
        return [b.__dict__ for b in backups]

    @server.tool()
    def memory_backup_restore(
        backup_name: str,
        restore_memorias: bool = True,
        restore_dbs: bool = True,
    ) -> dict[str, Any]:
        """Restore from a backup.

        Restores memoria files and/or databases from a backup archive.

        Args:
            backup_name: Name of the backup to restore.
            restore_memorias: Whether to restore memoria files.
            restore_dbs: Whether to restore databases.
        """
        success = memory.backup.restore_backup(
            backup_name,
            restore_memorias=restore_memorias,
            restore_dbs=restore_dbs,
        )
        return {"success": success, "backup_name": backup_name}

    @server.tool()
    def memory_sync_diff(
        remote: str | None = None,
    ) -> dict[str, Any]:
        """Compute diff between local and remote vaults.

        Computes the difference between local and remote vaults,
        identifying new, modified, deleted, and conflicted memorias.

        Args:
            remote: Path to remote vault (optional).
        """
        from pathlib import Path
        remote_path = Path(remote) if remote else None

        sync_mgr = memory.sync.__class__(memory, remote_path=remote_path)
        diff = sync_mgr.compute_diff()
        return diff.__dict__

    @server.tool()
    def memory_sync_push(
        remote: str | None = None,
    ) -> dict[str, Any]:
        """Push local changes to remote vault.

        Pushes modified and deleted memorias from local to remote vault.

        Args:
            remote: Path to remote vault (optional).
        """
        from pathlib import Path
        remote_path = Path(remote) if remote else None

        sync_mgr = memory.sync.__class__(memory, remote_path=remote_path)
        diff = sync_mgr.sync(direction="push")
        return diff.__dict__

    @server.tool()
    def memory_sync_pull(
        remote: str | None = None,
    ) -> dict[str, Any]:
        """Pull remote changes to local vault.

        Pulls new and modified memorias from remote to local vault.

        Args:
            remote: Path to remote vault (optional).
        """
        from pathlib import Path
        remote_path = Path(remote) if remote else None

        sync_mgr = memory.sync.__class__(memory, remote_path=remote_path)
        diff = sync_mgr.sync(direction="pull")
        return diff.__dict__

    @server.tool()
    def memory_sync_both(
        remote: str | None = None,
    ) -> dict[str, Any]:
        """Sync both directions (bidirectional).

        Performs bidirectional sync, pulling new changes from remote
        and pushing local changes to remote.

        Args:
            remote: Path to remote vault (optional).
        """
        from pathlib import Path
        remote_path = Path(remote) if remote else None

        sync_mgr = memory.sync.__class__(memory, remote_path=remote_path)
        diff = sync_mgr.sync(direction="both")
        return diff.__dict__

    # -- encryption tools ----------------------------------------------------------

    @server.tool()
    def memory_encrypt_unlock(
        password: str,
    ) -> dict[str, Any]:
        """Unlock the vault with password.

        Derives master key from password using PBKDF2 and stores it
        in memory for subsequent encryption/decryption operations.

        Args:
            password: User password for key derivation.
        """
        success = memory.encryption.unlock(password)
        return {"success": success, "status": "unlocked" if success else "failed"}

    @server.tool()
    def memory_encrypt_lock() -> dict[str, str]:
        """Lock the vault (clear master key from memory).

        Clears the master encryption key from memory, preventing
        further encryption/decryption operations until unlock() is called.
        """
        memory.encryption.lock()
        return {"status": "locked"}

    @server.tool()
    def memory_encrypt_status() -> dict[str, Any]:
        """Check if vault is unlocked.

        Returns the current lock status of the vault and whether
        encryption operations can be performed.
        """
        return {
            "is_unlocked": memory.encryption.is_unlocked(),
            "status": "unlocked" if memory.encryption.is_unlocked() else "locked",
        }

    # -- sharing tools -------------------------------------------------------------

    @server.tool()
    def memory_share_with_user(
        memoria_id: str,
        shared_with: str,
        permission: str = "read",
        expires_days: int | None = None,
    ) -> dict[str, Any]:
        """Share a memoria with a user.

        Shares a memoria with a specific user (by email or username)
        with the specified permission level. Optionally expires after
        a number of days.

        Args:
            memoria_id: The memoria ID to share.
            shared_with: Email or username to share with.
            permission: Permission level (read, comment, edit, admin).
            expires_days: Optional days until expiration.
        """
        share = memory.sharing.share_with_user(
            memoria_id=memoria_id,
            shared_with=shared_with,
            permission=permission,
            expires_days=expires_days,
        )
        return share.__dict__

    @server.tool()
    def memory_share_unshare(
        memoria_id: str,
        shared_with: str,
    ) -> dict[str, Any]:
        """Unshare a memoria from a user.

        Removes a share for a specific user.

        Args:
            memoria_id: The memoria ID.
            shared_with: The user to unshare.
        """
        success = memory.sharing.unshare_with_user(memoria_id, shared_with)
        return {"success": success, "memoria_id": memoria_id, "shared_with": shared_with}

    @server.tool()
    def memory_share_create_link(
        memoria_id: str,
        permission: str = "read",
        expires_hours: int = 24,
        password: str | None = None,
    ) -> dict[str, str]:
        """Create a temporary sharing link.

        Creates a temporary share link with optional password protection
        and expiration time.

        Args:
            memoria_id: The memoria ID.
            permission: Permission level.
            expires_hours: Hours until expiration.
            password: Optional password.
        """
        link = memory.sharing.create_link(
            memoria_id=memoria_id,
            permission=permission,
            expires_hours=expires_hours,
            password=password,
        )
        return {"link": link, "memoria_id": memoria_id}

    @server.tool()
    def memory_share_list(
        memoria_id: str,
    ) -> list[dict[str, Any]]:
        """List all shares for a memoria.

        Returns all shares for the specified memoria with their
        permission levels and expiration dates.

        Args:
            memoria_id: The memoria ID.
        """
        shares = memory.sharing.share_store.get_shares(memoria_id)
        return [s.__dict__ for s in shares]

    @server.tool()
    def memory_share_comment(
        memoria_id: str,
        content: str,
        author: str = "user",
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        """Add a comment to a memoria.

        Adds a comment to a memoria, optionally as a reply to another comment.

        Args:
            memoria_id: The memoria ID.
            content: Comment content.
            author: Comment author.
            parent_id: Optional parent comment ID.
        """
        comment = memory.sharing.add_comment(
            memoria_id=memoria_id,
            author=author,
            content=content,
            parent_id=parent_id,
        )
        return comment.__dict__

    @server.tool()
    def memory_share_comments(
        memoria_id: str,
    ) -> list[dict[str, Any]]:
        """List all comments for a memoria.

        Returns all comments for the specified memoria, including
        reply threads.

        Args:
            memoria_id: The memoria ID.
        """
        comments = memory.sharing.get_comments(memoria_id)
        return [c.__dict__ for c in comments]

    # -- analytics tools ------------------------------------------------------------

    @server.tool()
    def memory_analytics_summary() -> dict[str, Any]:
        """Get analytics summary of the memory corpus.

        Returns comprehensive metrics including total memorias,
        entity counts, growth rate, and type distribution.
        """
        metrics = memory.analytics.compute_corpus_metrics()
        return metrics.__dict__

    @server.tool()
    def memory_analytics_growth(
        days: int = 30,
    ) -> dict[str, Any]:
        """Get growth data over time.

        Returns memoria growth data grouped by date for the
        specified number of days.

        Args:
            days: Number of days to analyze.
        """
        growth = memory.analytics.compute_growth_data(days=days)
        return growth.__dict__

    # -- import/export tools --------------------------------------------------------

    @server.tool()
    def memory_import_json(
        input_path: str,
    ) -> dict[str, Any]:
        """Import memorias from JSON file.

        Imports memorias from a JSON file, creating new entries
        for each item in the file.

        Args:
            input_path: Path to JSON file.
        """
        from pathlib import Path
        result = memory.import_export.import_from(Path(input_path), "json")
        return result.__dict__

    @server.tool()
    def memory_import_csv(
        input_path: str,
    ) -> dict[str, Any]:
        """Import memorias from CSV file.

        Imports memorias from a CSV file, creating new entries
        for each row in the file.

        Args:
            input_path: Path to CSV file.
        """
        from pathlib import Path
        result = memory.import_export.import_from(Path(input_path), "csv")
        return result.__dict__

    @server.tool()
    def memory_export_json(
        output_path: str,
    ) -> dict[str, Any]:
        """Export memorias to JSON file.

        Exports all memorias to a JSON file with complete metadata.

        Args:
            output_path: Path to write JSON file.
        """
        from pathlib import Path
        result = memory.import_export.export_to(Path(output_path), "json")
        return result.__dict__

    @server.tool()
    def memory_export_csv(
        output_path: str,
    ) -> dict[str, Any]:
        """Export memorias to CSV file.

        Exports all memorias to a CSV file with columns for
        id, title, body, tags, type, created, updated.

        Args:
            output_path: Path to write CSV file.
        """
        from pathlib import Path
        result = memory.import_export.export_to(Path(output_path), "csv")
        return result.__dict__

    @server.tool()
    def memory_export_markdown_bundle(
        output_path: str,
    ) -> dict[str, Any]:
        """Export memorias to Markdown bundle (zip).

        Exports all memorias to a zip file containing individual
        .md files with frontmatter metadata.

        Args:
            output_path: Path to write zip file.
        """
        from pathlib import Path
        result = memory.import_export.export_to(Path(output_path), "markdown_bundle")
        return result.__dict__

    # -- source feedback tools -----------------------------------------------------

    @server.tool()
    def memory_feedback_record(
        source_id: str,
        query: str,
        rating: str,
    ) -> dict[str, Any]:
        """Record a 👍 / 👎 vote on a memoria for a given query text.

        Negative votes (rating="down") exclude the source for any future
        query whose embedding cosine-similarity with `query` is >= 0.85.
        Positive votes boost score by ~0.15 per vote (capped). Idempotent
        on (source_id, query, rating). Recording the opposite rating for
        the same (source, query) pair replaces the prior vote.

        Args:
            source_id: meta.id (full or unique prefix >= 4 chars).
            query: query text the feedback applies to. Embedded with the
                asymmetric retrieval prefix so future queries compare
                fairly.
            rating: "up" or "down".
        """
        return memory.feedback_record(source_id, query_text=query, rating=rating)

    @server.tool()
    def memory_feedback_list(
        source_id: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List recorded source-level feedback, newest first.

        Args:
            source_id: optional filter (full id or unique prefix).
            limit: max rows.
        """
        rows = memory.feedback_list(source_id=source_id, limit=limit)
        return {"rows": rows, "count": len(rows)}

    @server.tool()
    def memory_feedback_clear(source_id: str) -> dict[str, Any]:
        """Drop all feedback rows for `source_id`. Returns count deleted."""
        n = memory.feedback_clear(source_id)
        return {"source_id": source_id, "deleted": n}

    # -- OCR tools -----------------------------------------------------------------

    @server.tool()
    def memory_ocr_image(
        image_path: str,
    ) -> dict[str, Any]:
        """Run Apple Vision OCR on an image and return extracted text.

        Cached by SHA256 in `<state_dir>/ocr_cache`. Used by Synapse as a
        chat-time fallback for screenshots that have not been picked up
        by the indexer yet. Returns empty `text` if Vision is unavailable
        (non-macOS or missing PyObjC) or the file is missing.

        Args:
            image_path: Absolute path to the image file. The caller is
                responsible for resolving Obsidian `![[...]]` syntax
                to a real path on disk.
        """
        from pathlib import Path

        from memo.ocr import extract_text_cached, vision_available

        if not vision_available():
            return {"text": "", "cached": False, "error": "vision unavailable"}
        path = Path(image_path)
        if not path.exists():
            return {"text": "", "cached": False, "error": "file not found"}
        cache_dir = memory.cfg.state_dir / "ocr_cache"
        text = extract_text_cached(path, cache_dir=cache_dir)
        cached = (cache_dir / f"{__import__('hashlib').sha256(path.read_bytes()).hexdigest()[:32]}.txt").exists()
        return {"text": text, "cached": cached}

    # -- multi-modal tools (gamechanger #17) ----------------------------------------

    @server.tool()
    def memory_multimodal_add_image(
        image_path: str,
        memoria_id: str | None = None,
    ) -> dict[str, str]:
        """Add image to multi-modal corpus.

        Args:
            image_path: Path to the image file.
            memoria_id: Optional associated memoria ID.
        """
        from pathlib import Path
        content = memory.multimodal.add_image(Path(image_path), memoria_id)
        return {"content_id": content.id, "modality": content.modality}

    @server.tool()
    def memory_multimodal_add_audio(
        audio_path: str,
        memoria_id: str | None = None,
    ) -> dict[str, str]:
        """Add audio to multi-modal corpus.

        Args:
            audio_path: Path to the audio file.
            memoria_id: Optional associated memoria ID.
        """
        from pathlib import Path
        content = memory.multimodal.add_audio(Path(audio_path), memoria_id)
        return {"content_id": content.id, "modality": content.modality}

    @server.tool()
    def memory_multimodal_search_images(
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search with text, find images.

        Args:
            query: Text query.
            limit: Max results.

        Returns:
            List of CrossModalResult objects.
        """
        results = memory.multimodal.search.search_text_find_images(query, limit=limit)
        return [r.__dict__ for r in results]

    @server.tool()
    def memory_multimodal_search_audio(
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search with text, find audio.

        Args:
            query: Text query.
            limit: Max results.

        Returns:
            List of CrossModalResult objects.
        """
        results = memory.multimodal.search.search_text_find_audio(query, limit=limit)
        return [r.__dict__ for r in results]

    @server.tool()
    def memory_multimodal_search_all(
        query: str,
        limit: int = 10,
    ) -> dict[str, list[dict[str, Any]]]:
        """Search across all modalities.

        Args:
            query: Text query.
            limit: Max results per modality.

        Returns:
            Dict with results per modality.
        """
        results = memory.multimodal.search.search_all_modalities(query, limit=limit)
        return {k: [r.__dict__ for r in v] for k, v in results.items()}

    # -- collaborative tools (gamechanger #18) -------------------------------------

    @server.tool()
    def memory_collaborative_share_connection(
        user_id: str,
        entity_a: str,
        entity_b: str,
        relationship: str,
        confidence: float = 0.7,
    ) -> dict[str, str]:
        """Share a discovered connection with the community.

        Args:
            user_id: User ID who discovered the connection.
            entity_a: First entity.
            entity_b: Second entity.
            relationship: Type of relationship.
            confidence: Confidence score.

        Returns:
            Shared connection data.
        """
        conn = memory.collaborative.share_connection(
            user_id=user_id,
            entity_a=entity_a,
            entity_b=entity_b,
            relationship=relationship,
            confidence=confidence,
        )
        return conn.__dict__

    @server.tool()
    def memory_collaborative_connections(
        entity: str,
    ) -> list[dict[str, Any]]:
        """Get shared connections for an entity.

        Args:
            entity: Entity of interest.

        Returns:
            List of SharedConnection objects.
        """
        connections = memory.collaborative.get_shared_connections(entity)
        return [c.__dict__ for c in connections]

    @server.tool()
    def memory_collaborative_recommend(
        entity: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get recommended connections based on collective patterns.

        Args:
            entity: Entity of interest.
            limit: Max results.

        Returns:
            List of recommended SharedConnection objects.
        """
        recommendations = memory.collaborative.get_recommended_connections(entity, limit=limit)
        return [r.__dict__ for r in recommendations]

    @server.tool()
    def memory_collaborative_share_insight(
        user_id: str,
        content: str,
    ) -> dict[str, str]:
        """Share an insight with the community.

        Args:
            user_id: User ID.
            content: Insight content.

        Returns:
            Shared insight data.
        """
        insight = memory.collaborative.share_insight(user_id, content)
        return insight.__dict__

    @server.tool()
    def memory_collaborative_insights(
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get top voted insights from the community.

        Args:
            limit: Max results.

        Returns:
            List of CollectiveInsight objects.
        """
        insights = memory.collaborative.get_top_insights(limit=limit)
        return [i.__dict__ for i in insights]

    # -- time-machine (as-of) ---------------------------------------------------

    @server.tool()
    def memory_search_as_of(
        query: str,
        as_of: str,
        limit: int = 10,
        type: str | None = None,
        mode: str = "hybrid",
    ) -> dict[str, Any]:
        """Search the corpus as it existed on a past date (time-machine).

        Reconstructs a snapshot by replaying `history.db` events in
        reverse from "now", then runs the live search and post-filters
        to the snapshot's record set. Use this when you want to know
        what the user knew (or had decided) at a specific past point.

        Args:
            query: Free-text query.
            as_of: `YYYY-MM-DD` (interpreted as end-of-day UTC) or a
                full ISO-8601 timestamp.
            limit: Max results.
            type: Optional record-type filter.
            mode: `hybrid` (default), `vec`, or `bm25`.

        Returns: `{"as_of", "snapshot_size", "results": [...]}` where
        each result has the same shape as `memory_search` items.
        """
        from memo.time_machine import reconstruct
        snap = reconstruct(memory, as_of=as_of)
        hits = snap.search(query, limit=limit, mode=mode)
        if type:
            hits = [h for h in hits if h.type == type]
        return {
            "as_of": snap.as_of.isoformat(),
            "snapshot_size": len(snap),
            "results": [h.to_dict() for h in hits],
        }

    @server.tool()
    def memory_ask_as_of(question: str, as_of: str, k: int = 5) -> dict[str, Any]:
        """RAG question against a past snapshot.

        Same contract as `memory_ask` but the corpus view is rewound to
        `as_of`. The LLM is told explicitly that the view is historical
        so it doesn't smuggle in facts that only became known later.

        Args:
            question: Free-text question.
            as_of: `YYYY-MM-DD` or full ISO 8601.
            k: Top-k memorias to feed the model as context.
        """
        from memo.time_machine import reconstruct
        snap = reconstruct(memory, as_of=as_of)
        return snap.ask(question, k=k)

    @server.tool()
    def memory_diff(from_ts: str, to_ts: str | None = None) -> dict[str, Any]:
        """Diff the corpus between two snapshots — added, removed, updated.

        Args:
            from_ts: Start date — YYYY-MM-DD or full ISO 8601.
            to_ts: End date (defaults to "now" when omitted).
        """
        from datetime import UTC
        from datetime import datetime as _dt

        from memo.time_machine import diff as _diff
        to_resolved = to_ts or _dt.now(UTC).isoformat()
        d = _diff(memory, from_ts=from_ts, to_ts=to_resolved)
        return {
            "from_ts": d.from_ts.isoformat(),
            "to_ts": d.to_ts.isoformat(),
            "summary": d.summary(),
            "added": [{"id": r.id, "title": r.title, "type": r.type} for r in d.added],
            "removed": [{"id": r.id, "title": r.title, "type": r.type} for r in d.removed],
            "updated": d.updated,
        }

    # -- resources --------------------------------------------------------------
    #
    # Resources let MCP clients (Claude Desktop, Cursor, …) pin/list
    # memorias without paying a tool-call round-trip per access. Each
    # memoria is addressable as `memo://memory/<id>` and the recent
    # feed is at `memo://recent`. Clients show resources in their UI;
    # users can drag them into context.

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
            return (
                f"# Ambiguous id `{id}`\n\nMatches:\n\n"
                + "\n".join(f"- `{m}`" for m in exc.matches)
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
                    question, k=k, type_=type_, history=history, context=context,
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
        host = os.environ.get("MEMO_MCP_HOST", "127.0.0.1").strip() or "127.0.0.1"
        port = int(os.environ.get("MEMO_MCP_PORT", "18768").strip() or "18768")
        server.run(transport=transport, host=host, port=port)
    else:
        server.run()


if __name__ == "__main__":
    main()
