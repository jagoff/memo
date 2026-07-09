"""Read-path operations for `Memory` — search / list / get / resolve_id.

`_SearchOpsMixin` holds the retrieval methods and their helpers (RRF
orchestration, access tracking, cache read-through), moved verbatim from
the former `memory.py` god-file.
"""

from __future__ import annotations

import contextlib
import math
from datetime import UTC, datetime
from typing import Any

from memo.flags import active_flags, flag_bool, flag_float, flag_int
from memo.lifecycle import IS_FORGOTTEN_KEY
from memo.memory._base import _MemoryBase
from memo.memory.record import (
    _RECALL_DECAY_HALFLIFE_DEFAULT,
    AmbiguousIdError,
    MemoryRecord,
    _adaptive_rrf_k,
    _apply_decay,
    _log,
    _rrf_confident_top,
    _rrf_fuse,
    record_from_row,
)
from memo.perf import timer
from memo.tiers import REFERENCE_TYPES


def _record_touches_file(record: MemoryRecord, frag: str) -> bool:
    """True when the memory's capture-stamped tool arrays contain `frag`."""
    extra = record.extra or {}
    for key in ("files_read", "files_modified"):
        vals = extra.get(key)
        if isinstance(vals, list) and any(frag in str(v).lower() for v in vals):
            return True
    return False


def _parse_search_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class _SearchOpsMixin(_MemoryBase):
    # -- search -------------------------------------------------------------

    @timer(log_threshold_ms=50.0)
    def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        type_: str | None = None,
        mode: str = "hybrid",
        load_bodies: bool = True,
        disable_reranker: bool = False,
        recency: bool = False,
        exclude_types: set[str] | None = None,
        exclude_tags: set[str] | None = None,
        include_forgotten: bool = False,
        read_through: bool = False,
        entity_boost: bool | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        quality_rerank: bool | None = None,
        _trace: list[dict[str, Any]] | None = None,
    ) -> list[MemoryRecord]:
        """Top-k search. Three modes:

        - `vec` (semantic only): query embedded via `embed_query`,
          ranked by cosine.
        - `bm25` (keyword only): FTS5 over title+tags+body.
        - `hybrid` (default): reciprocal rank fusion of vec + bm25
          candidates. Picks up both diffuse semantic matches AND
          precise keyword matches (tag names, code snippets, file
          paths) that the small embedder model misses on its own.

        Each result has `.score` populated. For hybrid, `.score` is the
        fused RRF score (not directly comparable to single-mode scores
        but monotonic for ranking).

        Args:
            load_bodies: If False, bodies are not loaded from disk (lazy).
                Useful for reranking/filtering before final formatting.
                Caller must call `_load_body(record.path)` when needed.
            disable_reranker: If True, skip cross-encoder reranking even
                when enabled in config. Useful for chat synthesis where
                RRF is sufficient and reranker adds latency.
            recency: If True, blend a freshness bonus into the final score
                (newer memories rank higher) even when MEMO_SEARCH_DECAY_HALFLIFE
                is unset. The consumer-facing paths (recall hook, ask/chat) pass
                this so stale facts don't crowd out recent ones; the eval
                harness leaves it False to keep a raw, comparable baseline.
            exclude_types: Drop hits whose `type` is in this set, pushed into
                SQL so the candidate pool isn't spent on rows the caller will
                discard. The recall hook + briefing pass `REFERENCE_TYPES`
                (see `memo.tiers`) so bulk vault material stays searchable on
                demand but never drowns durable knowledge in the prompt.
            quality_rerank: Explicit opt-in for quality-aware reranking on
                consumer-facing search paths. Ambient recall leaves this off
                even when the feature flag is enabled.
        """

        def _add_trace(stage: str, **data: Any) -> None:
            if _trace is not None:
                _trace.append({"stage": stage, **data})

        if not query or not query.strip():
            _add_trace("candidate_generation", mode=mode, output_count=0)
            _add_trace("final", output_count=0)
            return []
        limit = limit or self.cfg.search_default_limit
        if date_to and len(date_to) == 10:
            date_to = date_to + "T23:59:59"  # bare date = whole day inclusive
        emb = None  # set in vec/hybrid branches; consumed by feedback boost below

        if mode == "bm25":
            rows = self.store.search_bm25(
                query, limit=limit, type_=type_, exclude_types=exclude_types
            )
            _add_trace(
                "candidate_generation", mode=mode, bm25_count=len(rows), output_count=len(rows)
            )
        elif mode == "exact":
            # Precise keyword lookup: strict AND (no OR loosening) with an
            # elevated tag/title field boost so a term in curated metadata
            # outranks the same term buried in a body. See search_bm25.
            rows = self.store.search_bm25(
                query,
                limit=limit,
                type_=type_,
                exclude_types=exclude_types,
                field_boost="exact",
            )
            _add_trace(
                "candidate_generation", mode=mode, bm25_count=len(rows), output_count=len(rows)
            )
        elif mode == "fuzzy":
            rows = self.store.search_fuzzy(
                query, limit=limit, type_=type_, exclude_types=exclude_types
            )
            _add_trace(
                "candidate_generation", mode=mode, fuzzy_count=len(rows), output_count=len(rows)
            )
        elif mode == "vec":
            # Asymmetric retrieval: queries are embedded WITH the
            # instruction prefix; documents are embedded RAW (in
            # `save()` / `update()`). See `_QUERY_INSTRUCTION_PREFIX`
            # in `embedder.py` for the why.
            emb = self.embedder.embed_query(query)
            rows = self.store.search(
                emb,
                limit=limit,
                type_=type_,
                exclude_types=exclude_types,
                date_from=date_from,
                date_to=date_to,
                exclude_tags=exclude_tags,
            )
            _add_trace(
                "candidate_generation", mode=mode, vec_count=len(rows), output_count=len(rows)
            )
        else:
            emb = None  # set below in hybrid's vec branch; used by feedback boost
            # hybrid — fetch a wider candidate set from each side and
            # fuse with reciprocal rank fusion (RRF). When the reranker
            # is enabled we widen the input pool to `rerank_input_k` so
            # the cross-encoder has more candidates to discriminate
            # between; the final `limit` is applied AFTER rerank.
            input_k = max(self.cfg.rerank_input_k, limit) if self.cfg.reranker_enabled else limit
            k_each = max(input_k * 2, 20)
            try:
                _query_for_embed = query
                # HyDE: generate hypothetical answer doc, embed that instead of raw query
                _hyde_enabled = flag_bool("MEMO_HYDE_ENABLED")
                if _hyde_enabled:
                    _log.debug("HyDE enabled, generating hypothetical doc")
                    _hyde_doc = self._generate_hyde_document(query)
                    if _hyde_doc:
                        _query_for_embed = _hyde_doc
                        _add_trace("hyde", original_query=query, hyde_doc=_hyde_doc[:100] + "…")
                        _log.info("HyDE doc generated: %s", _hyde_doc[:100])
                    else:
                        _log.warning("HyDE returned empty doc, falling back to original query")
                emb = self.embedder.embed_query(_query_for_embed)
                vec_hits = self.store.search(
                    emb,
                    limit=k_each,
                    type_=type_,
                    exclude_types=exclude_types,
                    date_from=date_from,
                    date_to=date_to,
                    exclude_tags=exclude_tags,
                )
            except Exception as exc:
                _log.warning(
                    "embedder unavailable, vec leg disabled in hybrid mode: %s",
                    exc,
                    exc_info=True,
                )
                vec_hits = []
            # Adaptive rerank pool: resize input_k based on the score spread of
            # vec candidates.  High variance → results are diverse, widen the pool
            # so the reranker can pick the best from a richer set.  Low variance
            # → tight cluster, shrink to avoid wasted cross-encoder calls.
            # Thresholds and multiplier are intentionally hardcoded (not flags) to
            # avoid flag proliferation; the outer MEMO_RERANK_ADAPTIVE_POOL gate
            # keeps the eval baseline stable when disabled.
            if self.cfg.reranker_enabled and flag_bool("MEMO_RERANK_ADAPTIVE_POOL") and vec_hits:
                _STDDEV_HIGH = 0.15  # above this → high diversity
                _STDDEV_LOW = 0.05  # below this → tight cluster
                _POOL_MULT = 1.5
                _POOL_CAP = 200
                scores = [h.get("score") or 0.0 for h in vec_hits]
                if len(scores) > 1:
                    _mean = sum(scores) / len(scores)
                    _var = sum((s - _mean) ** 2 for s in scores) / len(scores)
                    _stddev = math.sqrt(_var)
                    if _stddev > _STDDEV_HIGH:
                        input_k = min(int(input_k * _POOL_MULT), _POOL_CAP)
                    elif _stddev < _STDDEV_LOW:
                        input_k = max(limit + 5, 15)
                    # else: medium diversity → keep input_k unchanged
            bm_hits = self.store.search_bm25(
                query, limit=k_each, type_=type_, exclude_types=exclude_types
            )
            exact_hits = self.store.search_bm25(
                query,
                limit=k_each,
                type_=type_,
                exclude_types=exclude_types,
                field_boost="exact",
            )

            # Graph-based candidates (Entity-aware retrieval).
            # Enabled by MEMO_GRAPH_RETRIEVAL_ENABLED or when vec hits fall below
            # MEMO_GRAPH_FALLBACK_MIN_HITS threshold (fallback seeding).
            graph_hits = []
            use_graph = flag_bool("MEMO_GRAPH_RETRIEVAL_ENABLED")
            fallback_threshold = flag_int("MEMO_GRAPH_FALLBACK_MIN_HITS") or 0
            if not use_graph and fallback_threshold > 0 and len(vec_hits) < fallback_threshold:
                use_graph = True  # Fallback: vec is weak, try graph
            if use_graph:
                graph_hits = self._fetch_graph_candidates(
                    query, limit=k_each, type_=type_, exclude_types=exclude_types
                )

            # Fuse all sources. RRF supports multiple ranked lists. `k` is
            # configurable (MEMO_RRF_K, default 60); MEMO_RRF_ADAPTIVE opts
            # into density-driven k (sharper on agreement, softer when the
            # lists diverge) — off by default so the eval baseline holds.
            base_k_flag = flag_int("MEMO_RRF_K")
            base_k = 25 if base_k_flag is None else base_k_flag
            rrf_k = (
                _adaptive_rrf_k([vec_hits, bm_hits, exact_hits, graph_hits], base_k=base_k)
                if flag_bool("MEMO_RRF_ADAPTIVE")
                else base_k
            )

            # Per-leg weights: MEMO_SEARCH_VEC_WEIGHT / MEMO_SEARCH_BM25_WEIGHT
            # allow the user to tilt fusion toward semantic or keyword retrieval.
            # Defaults (0.5 each) preserve the historical equal-weight behaviour.
            # Exact BM25 reuses the keyword weight: it is still lexical evidence,
            # just stricter and metadata-boosted. Graph leg is always weight 1.0
            # (unscaled) because it contributes entity-context candidates, not a
            # competing retrieval signal.
            w_vec = flag_float("MEMO_SEARCH_VEC_WEIGHT")
            w_bm25 = flag_float("MEMO_SEARCH_BM25_WEIGHT")
            # Default is 0.5/0.5; treat None as default (should not happen given
            # registry defaults, but be defensive).
            w_vec = w_vec if w_vec is not None else 0.5
            w_bm25 = w_bm25 if w_bm25 is not None else 0.5
            # Warn when the user has explicitly set both but their sum deviates
            # from 1.0 by more than the 0.05 tolerance.
            _active = active_flags()
            _vec_set = "MEMO_SEARCH_VEC_WEIGHT" in _active
            _bm25_set = "MEMO_SEARCH_BM25_WEIGHT" in _active
            if _vec_set and _bm25_set:
                _weight_sum = w_vec + w_bm25
                if abs(_weight_sum - 1.0) > 0.05:
                    _log.warning(
                        "MEMO_SEARCH_VEC_WEIGHT + MEMO_SEARCH_BM25_WEIGHT = %.2f (expected 1.0)",
                        _weight_sum,
                    )
            # Query-length-adaptive lexical weight (default off): a 1-2 token
            # query is usually an identifier/tag lookup where BM25 is the
            # stronger signal. Explicit user weights always win.
            if (
                flag_bool("MEMO_SEARCH_ADAPTIVE_LEXICAL_WEIGHT")
                and not _vec_set
                and not _bm25_set
                and len(query.split()) <= 2
            ):
                w_vec, w_bm25 = 0.35, 0.65
            # Build weight list aligned with the lists passed to _rrf_fuse:
            # [vec_hits, bm_hits, exact_hits, graph_hits] → [w_vec, w_bm25, w_bm25, 1.0]
            rrf_weights = [w_vec, w_bm25, w_bm25, 1.0]

            rows = _rrf_fuse(
                vec_hits,
                bm_hits,
                exact_hits,
                graph_hits,
                limit=input_k,
                k=rrf_k,
                weights=rrf_weights,
            )
            _add_trace(
                "candidate_generation",
                mode="hybrid",
                vec_count=len(vec_hits),
                bm25_count=len(bm_hits),
                exact_count=len(exact_hits),
                graph_count=len(graph_hits),
                output_count=len(rows),
                rrf_k=rrf_k,
                input_k=input_k,
            )
        if date_from or date_to:
            from_dt = _parse_search_ts(date_from)
            to_dt = _parse_search_ts(date_to)

            def _date_ok(r: dict[str, Any]) -> bool:
                updated_dt = _parse_search_ts(str(r.get("updated") or ""))
                if updated_dt is None:
                    return False
                if from_dt is not None and updated_dt < from_dt:
                    return False
                return not (to_dt is not None and updated_dt > to_dt)

            rows = [r for r in rows if _date_ok(r)]
        if exclude_tags:
            rows = [
                r for r in rows
                if not (exclude_tags & {str(t) for t in (r.get("tags") or [])})
            ]
        out: list[MemoryRecord] = []
        # In hybrid mode the candidate pool can grow large (up to _POOL_CAP) and
        # the reranker trims it to `limit`. Loading every candidate body from
        # disk (open + frontmatter parse per hit) before that trim wastes up to
        # ~200 filesystem round-trips per search. Feed the pool's body text from
        # the FTS index in ONE batched query instead; the canonical .md body is
        # re-resolved from disk only for the surviving `limit` records right
        # before return (see `_resolve_disk_bodies` below). The reranker only
        # needs text, so this is a pure latency win with no ranking change.
        _bodies_from_fts = load_bodies and mode == "hybrid"
        _fts_bodies: dict[str, str] = (
            self.store.get_fts_bodies([r["id"] for r in rows]) if _bodies_from_fts else {}
        )
        for r in rows:
            if not load_bodies:
                body = ""
            elif _bodies_from_fts:
                body = _fts_bodies.get(r["id"], "")
            else:
                body = self._read_body(r["path"])
            out.append(record_from_row(r, body=body))
        _add_trace(
            "materialize", input_count=len(rows), output_count=len(out), load_bodies=load_bodies
        )
        # Drop soft-forgotten memories (forget_after TTL elapsed, see
        # lifecycle.py) before feedback/rerank so they never reach the
        # consumer — recall, ask, chat all route through here. Reversible
        # via `unforget`; pass include_forgotten=True to surface them.
        if out and not include_forgotten:
            before = len(out)
            out = [r for r in out if not (r.extra or {}).get(IS_FORGOTTEN_KEY)]
            _add_trace("forget_filter", input_count=before, output_count=len(out))
        # Source-level feedback (👍 / 👎) — applied AFTER RRF/vec retrieval
        # but BEFORE cross-encoder rerank so the reranker doesn't waste
        # cycles on hits the user already vetoed. Embeds the query once
        # (reusing the vec-mode embedding when available) and consults
        # the `source_feedback` table for each hit. Disabled when
        # MEMO_FEEDBACK_DISABLED=1.
        # Reuse the vec/hybrid-mode query embedding; never force a cold embed
        # here just to score feedback. In bm25/fuzzy mode `emb` is unset and the
        # caller deliberately opted out of the bi-encoder — a cold embed_query
        # on this path would blow the recall hook's 5s budget. Feedback matching
        # is itself vector-based, so applying it only when a query vector already
        # exists is consistent.
        if out and emb is not None and not flag_bool("MEMO_FEEDBACK_DISABLED"):
            try:
                before = len(out)
                # Feedback rows are keyed to the RAW user query embedding; under
                # HyDE `emb` was replaced with a hypothetical-doc vector, which
                # rarely clears the cosine threshold against query-text vectors,
                # silently nullifying every boost AND the thumbs_down exclude.
                # Re-embed the raw query so the match stays apples-to-apples.
                _fb_emb = (
                    self.embedder.embed_query(query)
                    if (mode == "hybrid" and flag_bool("MEMO_HYDE_ENABLED"))
                    else emb
                )
                out = self._apply_source_feedback(out, _fb_emb)
                _add_trace("feedback", input_count=before, output_count=len(out))
            except Exception as exc:
                _log.warning("source_feedback failed: %s", exc, exc_info=True)
        elif out and emb is None:
            _log.debug(
                "Source feedback boost skipped: query embedding unavailable in mode=%s",
                mode,
            )

        # Health scores (pre-rerank): multiply candidate scores by
        # (confidence × roi_score) BEFORE passing to the cross-encoder so
        # low-confidence hits (open contradictions, low-ROI) don't waste
        # reranker compute by surfacing them as strong candidates.
        # _health_applied tracks whether we already ran this pass; the
        # post-pipeline gate below skips a redundant second application.
        _reranker_will_run = (
            mode == "hybrid" and self.cfg.reranker_enabled and not disable_reranker and out
        )
        _health_applied = False
        if _reranker_will_run and not flag_bool("MEMO_HEALTH_SCORES_DISABLED"):
            before = len(out)
            out = self._apply_health_scores(out)
            _health_applied = True
            _add_trace("health_pre_rerank", input_count=before, output_count=len(out))
        # Cross-encoder rerank on hybrid mode only. Skipped for vec/bm25
        # since those callers explicitly opted out of fusion entirely;
        # adding rerank to single-mode searches would surprise users
        # benchmarking the raw bi-encoder or BM25 surfaces.
        # Also skipped when disable_reranker=True (e.g., chat synthesis).
        if _reranker_will_run and flag_bool("MEMO_RERANK_SKIP_CONFIDENT_RRF"):
            min_ratio_flag = flag_float("MEMO_RERANK_SKIP_MIN_RATIO")
            min_gap_flag = flag_float("MEMO_RERANK_SKIP_MIN_GAP")
            decision = _rrf_confident_top(
                out,
                min_ratio=3.0 if min_ratio_flag is None else min_ratio_flag,
                min_gap=0.05 if min_gap_flag is None else min_gap_flag,
            )
            if decision.skip:
                _reranker_will_run = False
                out = out[:limit]
                _add_trace(
                    "rerank_skip",
                    reason="confident_rrf",
                    top_id=decision.top_id,
                    ratio=round(decision.ratio, 6),
                    gap=round(decision.gap, 6),
                    output_count=len(out),
                )
        if _reranker_will_run:
            before = len(out)
            out = self._rerank(query, out, top_n=limit)
            _add_trace("rerank", input_count=before, output_count=len(out))
        else:
            # No reranker ran (disabled per-call — e.g. ask/chat pass
            # disable_reranker=True — or skip-confident already fired): the
            # candidate pool was inflated to `input_k` for the reranker, so clamp
            # it back to `limit` here. Otherwise ask/chat get the whole pool
            # (the `k` contract is violated) and the downstream boosts + per-hit
            # disk reads run over the full pool instead of ≤ limit.
            out = out[:limit]
        # Recency decay: blend a freshness bonus into the score so older
        # memories don't crowd out recent ones. MEMO_SEARCH_DECAY_HALFLIFE
        # (days) sets the halflife explicitly; if unset, the consumer paths
        # (recall/ask/chat) still get a sensible default when they pass
        # `recency=True`, while raw `search()` callers (e.g. the eval harness)
        # stay decay-free for a comparable baseline.
        halflife_days = float(flag_int("MEMO_SEARCH_DECAY_HALFLIFE") or 0)
        if halflife_days <= 0 and recency:
            halflife_days = _RECALL_DECAY_HALFLIFE_DEFAULT
        if halflife_days > 0 and out:
            before = len(out)
            alpha_flag = flag_float("MEMO_SEARCH_DECAY_ALPHA")
            alpha = min(max(alpha_flag if alpha_flag is not None else 0.15, 0.0), 1.0)
            out = _apply_decay(out, halflife_days=halflife_days, alpha=alpha)
            _add_trace(
                "recency_decay",
                input_count=before,
                output_count=len(out),
                halflife_days=halflife_days,
                alpha=alpha,
            )
        # Cache-tier read-through: on a local miss/under-fill, pull from the
        # backing store and materialize locally. OPT-IN per call — only the
        # consumer paths (ask/chat) pass read_through=True. The recall hook
        # never does: a backend subprocess (≤5s) would blow its 5s budget.
        if read_through and self.cache.policy.read_through and len(out) < limit:
            before = len(out)
            out = self._cache_read_through(query, out, limit)
            _add_trace("cache_read_through", input_count=before, output_count=len(out))
        # Entity-aware score boost: if query mentions known entities (persons,
        # technologies, projects), boost chunks whose extra["entities"] overlaps.
        # Gated by MEMO_ENTITY_RETRIEVAL_ENABLED. Best-effort: any failure is
        # silent so entity extraction never breaks the search path.
        # Per-call `entity_boost` overrides the env flag (thread-safe — callers
        # like the MCP entity-search tool no longer mutate global os.environ).
        _entity_on = (
            entity_boost if entity_boost is not None else flag_bool("MEMO_ENTITY_RETRIEVAL_ENABLED")
        )
        if out and _entity_on:
            before = len(out)
            out = self._apply_entity_boost(query, out)
            _add_trace("entity_boost", input_count=before, output_count=len(out))

        # Contradiction penalty: penalise the older side of open contradiction
        # pairs among the retrieved results so stale/superseded memories don't
        # surface at the top. Gated by MEMO_CONTRADICT_PENALTY_ENABLED (default
        # off — the sidecar DB is empty until `memo contradict scan` runs at
        # least once). Only the recall/ask/chat consumer paths benefit; the eval
        # harness and recall-hook budget make this opt-in.
        if out and flag_bool("MEMO_CONTRADICT_PENALTY_ENABLED"):
            before = len(out)
            out = self._apply_contradict_penalty(out)
            _add_trace("contradiction_penalty", input_count=before, output_count=len(out))
        if out and flag_bool("MEMO_GRAPH_EXPANSION_ENABLED"):
            before = len(out)
            out = self._apply_graph_expansion(
                out,
                load_bodies=load_bodies,
                exclude_types=exclude_types,
            )
            _add_trace("graph_expansion", input_count=before, output_count=len(out))
        if out and flag_bool("MEMO_RETRIEVAL_BOOST"):
            before = len(out)
            out = self._apply_retrieval_boost(query, out)
            _add_trace("retrieval_boost", input_count=before, output_count=len(out))
        if out and not flag_bool("MEMO_HEALTH_SCORES_DISABLED") and not _health_applied:
            before = len(out)
            out = self._apply_health_scores(out)
            _add_trace("health", input_count=before, output_count=len(out))
        if out and quality_rerank and flag_bool("MEMO_QUALITY_RERANK"):
            before = len(out)
            out = self._apply_quality_rerank(out)
            _add_trace("quality_rerank", input_count=before, output_count=len(out))
        # Co-recall ranking boost: surface memories relationally associated with
        # the top hit. Read side of MEMO_GRAPH_CO_RECALL; cheap (one graph query),
        # behind the same flag so default behaviour is unchanged.
        if out and flag_bool("MEMO_GRAPH_CO_RECALL"):
            before = len(out)
            out = self._apply_co_recall_boost(out)
            _add_trace("co_recall_boost", input_count=before, output_count=len(out))
        # Reference-tier noise floor: in EXPLICIT retrieval (search/ask/chat)
        # bulk vault chunks compete with durable memories — the recall hook
        # SQL-excludes them (MEMO_RECALL_EXCLUDE_REFERENCE) but the explicit
        # paths don't. When MEMO_REFERENCE_SEARCH_FLOOR > 0, a reference-tier
        # hit must clear the floor on its FINAL (post-boost, mode-dependent)
        # score to stay; durable-tier hits are untouched. Skipped when the
        # caller explicitly asked for the reference tier (type_="reference")
        # — the tier is "searchable on demand" and an explicit ask wins.
        # Applied before _record_access so dropped hits don't log an access.
        _ref_floor = flag_float("MEMO_REFERENCE_SEARCH_FLOOR") or 0.0
        if out and _ref_floor > 0 and type_ not in REFERENCE_TYPES:
            before = len(out)
            out = [
                r for r in out if r.type not in REFERENCE_TYPES or (r.score or 0.0) >= _ref_floor
            ]
            _add_trace(
                "reference_floor",
                input_count=before,
                output_count=len(out),
                floor=_ref_floor,
            )
        # Chunk→parent mapping (K3): a winning reference CHUNK stands in for
        # its parent note. Flag-gated default off; skipped when the caller
        # explicitly asked for the reference tier (same rule as the floor
        # above). See _map_chunks_to_parents.
        if out and flag_bool("MEMO_SEARCH_CHUNK_PARENT") and type_ not in REFERENCE_TYPES:
            before = len(out)
            out = self._map_chunks_to_parents(out)
            _add_trace("chunk_parent", input_count=before, output_count=len(out))
        self._record_access([r.id for r in out])
        # Co-recall graph edges: record which memories surface together.
        # Gated by flag so the graph DB write stays opt-in (off by default).
        if len(out) >= 2 and flag_bool("MEMO_GRAPH_CO_RECALL"):
            try:
                self.graph.record_co_recall([r.id for r in out])
            except Exception as _co_exc:
                _log.debug("co_recall record failed: %s", _co_exc)
        # Re-resolve canonical .md bodies for the survivors. The pool was fed
        # FTS body text (cheap, for rerank); consumers must get the canonical
        # file content — which can differ from the FTS-indexed text (e.g. once
        # contextual-retrieval prepends a situating sentence to the indexed
        # body). Only `len(out)` (≤ limit) disk reads here, not the whole pool.
        if _bodies_from_fts and out:
            import dataclasses

            resolved: list[MemoryRecord] = []
            for r in out:
                disk = self._read_body(r.path)
                resolved.append(dataclasses.replace(r, body=disk) if disk else r)
            out = resolved
        _add_trace("final", output_count=len(out), limit=limit)
        return out

    def search_with_trace(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Run `search()` and return the same hits plus structured pipeline trace."""
        trace: list[dict[str, Any]] = []
        hits = self.search(query, _trace=trace, **kwargs)
        return {"hits": hits, "trace": trace}

    def _map_chunks_to_parents(self, out: list[MemoryRecord]) -> list[MemoryRecord]:
        """Chunk-hit → parent mapping (MEMO_SEARCH_CHUNK_PARENT, default off).

        A chunk row (type=reference, extra.parent_id from MEMO_CHUNK_INGEST)
        that wins retrieval is evidence the PARENT note matters — surface the
        whole note once, at the best chunk's rank and score, instead of a
        fragment (or fragment + parent duplicates). Parents already in the
        list dedup by id; a chunk whose parent was deleted stays as-is.
        Zero hook cost: the recall hook SQL-excludes the reference tier, so
        this loop never sees a chunk row on that path.
        """
        import dataclasses

        mapped: list[MemoryRecord] = []
        seen_ids: set[str] = set()
        for r in out:
            parent_id = (r.extra or {}).get("parent_id")
            if r.type in REFERENCE_TYPES and isinstance(parent_id, str) and parent_id:
                parent = self.get(parent_id)
                if parent is not None:
                    r = dataclasses.replace(parent, score=r.score)
            if r.id in seen_ids:
                continue  # a better-ranked chunk/parent already covers it
            seen_ids.add(r.id)
            mapped.append(r)
        return mapped

    def search_by_file(
        self,
        query: str,
        *,
        file: str,
        limit: int = 10,
        mode: str = "hybrid",
        type_: str | None = None,
        quality_rerank: bool | None = None,
    ) -> list[MemoryRecord]:
        """High-precision by-file lane over the capture-stamped
        `files_read`/`files_modified` arrays (see `capture.collect_tool_files`).

        Over-fetches a wider pool via the normal `search()` — ranking, boosts
        and tier exclusions stay identical — then keeps hits whose stamped
        arrays contain `file` (case-insensitive substring). Falls back to a
        plain `search()` when `file` is empty. Opt-in surface for MCP/CLI
        callers; deliberately NOT wired into the recall hook (5s budget).
        """
        frag = (file or "").strip().lower()
        if not frag:
            return self.search(  # type: ignore[no-any-return]
                query,
                limit=limit,
                mode=mode,
                type_=type_,
                quality_rerank=quality_rerank,
            )
        pool: list[MemoryRecord] = self.search(
            query or file,
            limit=max(limit * 5, 25),
            mode=mode,
            type_=type_,
            quality_rerank=quality_rerank,
        )
        return [r for r in pool if _record_touches_file(r, frag)][:limit]

    def list(
        self,
        *,
        limit: int = 20,
        type_: str | None = None,
        include_forgotten: bool = False,
        updated_since: str | None = None,
    ) -> list[MemoryRecord]:
        """Recent entries by `updated` desc. Body included for each.

        Soft-forgotten memories (see `forget`) are excluded unless
        `include_forgotten=True`. `updated_since` (ISO-8601) filters at the
        DB level — incremental callers (e.g. contradiction scans) get the
        freshest anchors within `limit` instead of post-filtering a page of
        older rows.
        """
        rows = self.store.list_recent(limit=limit, type_=type_, updated_since=updated_since)
        if not include_forgotten:
            rows = [r for r in rows if not (r.get("extra") or {}).get(IS_FORGOTTEN_KEY)]
        return [record_from_row(r, body=self._read_body(r["path"])) for r in rows]

    # -- get ----------------------------------------------------------------

    def resolve_id(self, id_or_prefix: str) -> str | None:
        """Resolve a full id or a unique prefix.

        Returns the canonical 32-char id if `id_or_prefix` matches exactly
        one record (full or prefix), or None if nothing matches. Raises
        `AmbiguousIdError` when 2+ records share the prefix — the caller
        is expected to surface the candidates so the user can disambiguate.

        Why prefix lookup: pasting a 32-char UUID4 from chat is friction.
        Git-style 7-char prefixes are unique with overwhelming probability
        for the corpus sizes memo targets (~thousands).
        """
        if not id_or_prefix:
            return None
        # Fast path: full hex hit.
        if len(id_or_prefix) == 32 and self.store.get(id_or_prefix) is not None:
            return id_or_prefix
        matches = self.store.find_by_prefix(id_or_prefix.lower())
        if len(matches) == 1:
            return str(matches[0])
        if len(matches) > 1:
            raise AmbiguousIdError(id_or_prefix, matches)
        return None

    def get(self, id_: str) -> MemoryRecord | None:
        resolved = self.resolve_id(id_)
        if resolved is None:
            return None
        r = self.store.get(resolved)
        if not r:
            return None
        return record_from_row(r, body=self._read_body(r["path"]))

    def around(self, id_: str, *, before: int = 2, after: int = 2) -> dict[str, Any]:
        """Timeline neighbourhood of one memory — 'what was happening around this'.

        Reference chunks (extra carries parent_path + chunk_seq) expand to
        seq-adjacent siblings of the same source note; durable memories expand
        to created-time neighbours (reference tier excluded so bulk chunks
        don't drown the timeline). Deliberate path only (MCP/CLI) — never
        called from the 5s recall hook."""
        from memo.tiers import REFERENCE_TYPES

        resolved = self.resolve_id(id_)
        row = self.store.get(resolved) if resolved else None
        if row is None:
            return {"anchor": None, "mode": None, "neighbors": []}
        extra = row.get("extra") or {}
        try:
            seq = int(extra["chunk_seq"]) if extra.get("chunk_seq") is not None else None
        except (TypeError, ValueError):
            seq = None
        parent = extra.get("parent_path")
        if parent and seq is not None:
            rows = self.store.chunks_adjacent(str(parent), seq, before=before, after=after)
            mode = "chunk_seq"
        else:
            rows = self.store.records_around_created(
                str(row["created"]), before=before, after=after,
                exclude_types=set(REFERENCE_TYPES),
            )
            mode = "created"
        neighbors: list[dict[str, Any]] = []
        for r in rows:
            if r["id"] == resolved:
                continue
            body = ""
            with contextlib.suppress(Exception):
                body = self._read_body(r["path"])
            neighbors.append({**r, "body_snippet": (body or "")[:400]})
        return {
            "anchor": {"id": resolved, "title": row["title"], "created": row["created"]},
            "mode": mode,
            "neighbors": neighbors,
        }

    def _generate_hyde_document(self, query: str) -> str | None:
        """Generate a hypothetical answer document for HyDE query expansion."""
        max_tokens = 256 if (_mt := flag_int("MEMO_HYDE_MAX_TOKENS")) is None else _mt
        prompt = (
            f"Given the user's question, write a hypothetical ideal answer as a concise "
            f"informational document (like a knowledge-base entry). Focus on the most likely "
            f"correct answer. Write in a factual, matter-of-fact style.\n\n"
            f"Question: {query}\n\n"
            f"Hypothetical Answer:"
        )
        try:
            chat = self._ensure_chat()
            resp = chat.chat(
                model=self.cfg.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "max_tokens": max_tokens},
            )
            return ((resp.get("message") or {}).get("content") or "").strip()
        except Exception as exc:
            _log.warning("HyDE generation failed: %s", exc, exc_info=True)
            return None
