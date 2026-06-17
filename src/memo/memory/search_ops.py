"""Read-path operations for `Memory` — search / list / get / resolve_id.

`_SearchOpsMixin` holds the retrieval methods and their helpers (RRF
orchestration, access tracking, cache read-through), moved verbatim from
the former `memory.py` god-file.
"""

from __future__ import annotations

import builtins
import math
import sqlite3
from dataclasses import replace
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
    _rrf_fuse,
)
from memo.perf import timer


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
        include_forgotten: bool = False,
        read_through: bool = False,
        entity_boost: bool | None = None,
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
                (newer memorias rank higher) even when MEMO_SEARCH_DECAY_HALFLIFE
                is unset. The consumer-facing paths (recall hook, ask/chat) pass
                this so stale facts don't crowd out recent ones; the eval
                harness leaves it False to keep a raw, comparable baseline.
            exclude_types: Drop hits whose `type` is in this set, pushed into
                SQL so the candidate pool isn't spent on rows the caller will
                discard. The recall hook + briefing pass `REFERENCE_TYPES`
                (see `memo.tiers`) so bulk vault material stays searchable on
                demand but never drowns durable knowledge in the prompt.
        """
        def _add_trace(stage: str, **data: Any) -> None:
            if _trace is not None:
                _trace.append({"stage": stage, **data})

        if not query or not query.strip():
            _add_trace("candidate_generation", mode=mode, output_count=0)
            _add_trace("final", output_count=0)
            return []
        limit = limit or self.cfg.search_default_limit

        if mode == "bm25":
            rows = self.store.search_bm25(
                query, limit=limit, type_=type_, exclude_types=exclude_types
            )
            _add_trace("candidate_generation", mode=mode, bm25_count=len(rows), output_count=len(rows))
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
            _add_trace("candidate_generation", mode=mode, bm25_count=len(rows), output_count=len(rows))
        elif mode == "fuzzy":
            rows = self.store.search_fuzzy(
                query, limit=limit, type_=type_, exclude_types=exclude_types
            )
            _add_trace("candidate_generation", mode=mode, fuzzy_count=len(rows), output_count=len(rows))
        elif mode == "vec":
            # Asymmetric retrieval: queries are embedded WITH the
            # instruction prefix; documents are embedded RAW (in
            # `save()` / `update()`). See `_QUERY_INSTRUCTION_PREFIX`
            # in `embedder.py` for the why.
            emb = self.embedder.embed_query(query)
            rows = self.store.search(emb, limit=limit, type_=type_, exclude_types=exclude_types)
            _add_trace("candidate_generation", mode=mode, vec_count=len(rows), output_count=len(rows))
        else:
            # hybrid — fetch a wider candidate set from each side and
            # fuse with reciprocal rank fusion (RRF). When the reranker
            # is enabled we widen the input pool to `rerank_input_k` so
            # the cross-encoder has more candidates to discriminate
            # between; the final `limit` is applied AFTER rerank.
            input_k = self.cfg.rerank_input_k if self.cfg.reranker_enabled else limit
            k_each = max(input_k * 2, 30)
            emb = self.embedder.embed_query(query)
            vec_hits = self.store.search(
                emb, limit=k_each, type_=type_, exclude_types=exclude_types
            )
            # Adaptive rerank pool: resize input_k based on the score spread of
            # vec candidates.  High variance → results are diverse, widen the pool
            # so the reranker can pick the best from a richer set.  Low variance
            # → tight cluster, shrink to avoid wasted cross-encoder calls.
            # Thresholds and multiplier are intentionally hardcoded (not flags) to
            # avoid flag proliferation; the outer MEMO_RERANK_ADAPTIVE_POOL gate
            # keeps the eval baseline stable when disabled.
            if (
                self.cfg.reranker_enabled
                and flag_bool("MEMO_RERANK_ADAPTIVE_POOL")
                and vec_hits
            ):
                _STDDEV_HIGH = 0.15  # above this → high diversity
                _STDDEV_LOW = 0.05   # below this → tight cluster
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

            # Graph-based candidates (Entity-aware retrieval)
            graph_hits = []
            if flag_bool("MEMO_GRAPH_RETRIEVAL_ENABLED"):
                graph_hits = self._fetch_graph_candidates(
                    query, limit=k_each, type_=type_, exclude_types=exclude_types
                )

            # Fuse all sources. RRF supports multiple ranked lists. `k` is
            # configurable (MEMO_RRF_K, default 60); MEMO_RRF_ADAPTIVE opts
            # into density-driven k (sharper on agreement, softer when the
            # lists diverge) — off by default so the eval baseline holds.
            base_k = flag_int("MEMO_RRF_K") or 60
            rrf_k = (
                _adaptive_rrf_k([vec_hits, bm_hits, graph_hits], base_k=base_k)
                if flag_bool("MEMO_RRF_ADAPTIVE")
                else base_k
            )

            # Per-leg weights: MEMO_SEARCH_VEC_WEIGHT / MEMO_SEARCH_BM25_WEIGHT
            # allow the user to tilt fusion toward semantic or keyword retrieval.
            # Defaults (0.5 each) preserve the historical equal-weight behaviour.
            # Graph leg is always weight 1.0 (unscaled) — it contributes
            # entity-context candidates, not a competing retrieval signal.
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
                        "MEMO_SEARCH_VEC_WEIGHT + MEMO_SEARCH_BM25_WEIGHT = %.2f "
                        "(expected 1.0)",
                        _weight_sum,
                    )
            # Build weight list aligned with the lists passed to _rrf_fuse:
            # [vec_hits, bm_hits, graph_hits] → [w_vec, w_bm25, 1.0]
            rrf_weights = [w_vec, w_bm25, 1.0]

            rows = _rrf_fuse(
                vec_hits, bm_hits, graph_hits,
                limit=input_k, k=rrf_k, weights=rrf_weights,
            )
            _add_trace(
                "candidate_generation",
                mode="hybrid",
                vec_count=len(vec_hits),
                bm25_count=len(bm_hits),
                graph_count=len(graph_hits),
                output_count=len(rows),
                rrf_k=rrf_k,
                input_k=input_k,
            )
        out: list[MemoryRecord] = []
        for r in rows:
            body = self._read_body(r["path"]) if load_bodies else ""
            out.append(
                MemoryRecord(
                    id=r["id"],
                    path=r["path"],
                    title=r["title"],
                    type=r["type"],
                    tags=r["tags"],
                    created=r["created"],
                    updated=r["updated"],
                    body=body,
                    extra=r.get("extra") or {},
                    score=r.get("score"),
                ),
            )
        _add_trace("materialize", input_count=len(rows), output_count=len(out), load_bodies=load_bodies)
        # Drop soft-forgotten memorias (forget_after TTL elapsed, see
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
        fb_emb = locals().get("emb")
        if out and fb_emb is not None and not flag_bool("MEMO_FEEDBACK_DISABLED"):
            try:
                before = len(out)
                out = self._apply_source_feedback(out, fb_emb)
                _add_trace("feedback", input_count=before, output_count=len(out))
            except Exception as exc:
                _log.warning("source_feedback failed: %s", exc, exc_info=True)
        elif out and fb_emb is None:
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
        if _reranker_will_run:
            before = len(out)
            out = self._rerank(query, out, top_n=limit)
            _add_trace("rerank", input_count=before, output_count=len(out))
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
            alpha = min(max(flag_float("MEMO_SEARCH_DECAY_ALPHA") or 0.15, 0.0), 1.0)
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
        self._record_access([r.id for r in out])
        _add_trace("final", output_count=len(out), limit=limit)
        return out

    def search_with_trace(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Run `search()` and return the same hits plus structured pipeline trace."""
        trace: list[dict[str, Any]] = []
        hits = self.search(query, _trace=trace, **kwargs)
        return {"hits": hits, "trace": trace}

    def _fetch_graph_candidates(
        self,
        query: str,
        *,
        limit: int = 20,
        type_: str | None = None,
        exclude_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch memorias sharing entities with the query via the knowledge graph.
        Returns a ranked list for RRF fusion.
        """
        try:
            from memo.entity_extractor import extract_entities
            query_entities = extract_entities(query)
            if not query_entities:
                return []

            # Find memorias sharing these entities
            # Score them by how many distinct query entities they match
            memoria_counts: dict[str, int] = {}
            for ent_name in query_entities:
                for mid in self.graph.entity_memorias(ent_name):
                    memoria_counts[mid] = memoria_counts.get(mid, 0) + 1

            if not memoria_counts:
                return []

            # Sort by count desc, then by updated desc (tie-breaker)
            sorted_mids = sorted(memoria_counts.keys(), key=lambda x: memoria_counts[x], reverse=True)

            # Fetch batch from store
            rows = self.store.get_batch(sorted_mids[: limit * 3])

            # Filter by type + exclude_types
            filtered = []
            for r in rows:
                if type_ and r.get("type") != type_:
                    continue
                if exclude_types and r.get("type") in exclude_types:
                    continue
                filtered.append(r)

            # Sort by entity match count descending so that _rrf_fuse, which
            # uses list position as rank, places high-match candidates first.
            # get_batch() returns rows in storage order (not insertion order),
            # so we must re-sort here to get the correct rank ordering.
            filtered.sort(key=lambda r: memoria_counts[r["id"]], reverse=True)
            out = filtered[:limit]

            # Assign a synthetic RRF score so the graph list is on the same
            # scale as vec and BM25 lists.  _rrf_fuse() uses rank (position),
            # not the raw score value, for its own fusion sum; but other
            # post-RRF steps (health multipliers, decay blending) operate on
            # the fused score, so having pre-fusion scores that are integers
            # (1, 2, 3...) rather than [0,1] floats would corrupt any path
            # that inspects the score BEFORE _rrf_fuse runs.
            rrf_k = flag_int("MEMO_RRF_K") or 60
            for rank, r in enumerate(out):
                r["score"] = 1.0 / (rrf_k + rank + 1)
            return out
        except Exception as exc:
            _log.debug("graph_candidates failed: %s", exc)
            return []

    def _apply_graph_expansion(
        self,
        results: list[MemoryRecord],
        *,
        load_bodies: bool = True,
        exclude_types: set[str] | None = None,
    ) -> list[MemoryRecord]:
        """Append graph-adjacent memorias not in the primary result set.

        Walks entity edges from the top-3 results (1-hop) via the knowledge
        graph and appends up to 3 connected memorias scored at 0.6× the
        minimum primary score. Requires entities to have been extracted first
        (`memo extract-entities`). Best-effort: any failure returns `results`
        unchanged so graph availability never breaks the search path.
        """
        try:
            existing_ids = {r.id for r in results}
            min_score = min((r.score or 0.0) for r in results)
            # Floor at a small positive epsilon: when the primaries carry no /
            # zero / negative score (rerank or feedback can push scores ≤ 0), a
            # 0.6× multiplier yields 0.0 and the graph-expanded candidates become
            # unrankable. Keep them strictly positive but below the weakest primary.
            expansion_score = round(max(min_score, 0.01) * 0.6, 6)

            # Collect entity names from top-3 primary results.
            entity_names: list[str] = []
            seen_entity_keys: set[str] = set()
            for r in results[:3]:
                for ent in self.graph.memoria_entities(r.id):
                    key = f"{ent['name']}:{ent['type']}"
                    if key not in seen_entity_keys:
                        seen_entity_keys.add(key)
                        entity_names.append(ent["name"])
            if not entity_names:
                return results

            # Collect connected memoria IDs (1-hop via shared entity membership).
            candidate_ids: list[str] = []
            seen_candidates: set[str] = set(existing_ids)
            for entity_name in entity_names[:5]:
                for mem_id in self.graph.entity_memorias(entity_name):
                    if mem_id not in seen_candidates:
                        seen_candidates.add(mem_id)
                        candidate_ids.append(mem_id)
            if not candidate_ids:
                return results

            rows = self.store.get_batch(candidate_ids[:10])
            if not rows:
                return results

            expanded: list[MemoryRecord] = []
            for r in rows:
                if len(expanded) >= 3:
                    break
                if exclude_types and r.get("type") in exclude_types:
                    continue
                body = self._read_body(r["path"]) if load_bodies else ""
                expanded.append(
                    MemoryRecord(
                        id=r["id"],
                        path=r["path"],
                        title=r["title"],
                        type=r["type"],
                        tags=r["tags"],
                        created=r["created"],
                        updated=r["updated"],
                        body=body,
                        extra={**(r.get("extra") or {}), "graph_expanded": True},
                        score=expansion_score,
                    ),
                )
            return results + expanded
        except Exception as exc:
            _log.debug("graph_expansion failed: %s", exc)
            return results

    def _apply_contradict_penalty(
        self,
        results: list[MemoryRecord],
    ) -> list[MemoryRecord]:
        """Apply score penalty to the older side of open contradiction pairs."""
        penalty = max(0.0, min(1.0, flag_float("MEMO_CONTRADICT_PENALTY") or 0.4))
        ids = [r.id for r in results]
        try:
            pairs = self.contradict_store.pairs_for_ids(ids)
        except Exception as exc:
            _log.debug("contradict_penalty pairs_for_ids failed: %s", exc)
            return results
        if not pairs:
            return results
        # Build a set of IDs to penalise (older side of each contradiction pair).
        penalise: set[str] = set()
        id_to_updated: dict[str, str] = {r.id: r.updated for r in results}
        for pair in pairs:
            if pair.relationship != "contradiction":
                continue
            a_ts = id_to_updated.get(pair.memoria_id_a, "")
            b_ts = id_to_updated.get(pair.memoria_id_b, "")
            if a_ts and b_ts:
                # Both sides in results: penalise the older one.
                penalise.add(pair.memoria_id_a if a_ts < b_ts else pair.memoria_id_b)
            elif a_ts:
                penalise.add(pair.memoria_id_a)
            elif b_ts:
                penalise.add(pair.memoria_id_b)
        if not penalise:
            return results
        penalised = [
            replace(r, score=(r.score or 0.0) * penalty) if r.id in penalise else r for r in results
        ]
        # Re-sort by score descending so penalised entries sink naturally.
        penalised.sort(
            key=lambda r: r.score or 0.0,
            reverse=True,
        )
        return penalised

    def _apply_entity_boost(
        self,
        query: str,
        results: list[MemoryRecord],
    ) -> list[MemoryRecord]:
        """Boost chunks whose stored entities overlap with query entities."""
        try:
            from memo.entity_extractor import entity_match_score, extract_entities

            query_entities = extract_entities(query)
            if not query_entities:
                return results
            boosted: list[MemoryRecord] = []
            changed = False
            for r in results:
                doc_entities: list[str] = (r.extra or {}).get("entities") or []
                boost = entity_match_score(query_entities, doc_entities)
                if boost > 0.0:
                    r = replace(r, score=round((r.score or 0.0) + boost, 6))
                    changed = True
                boosted.append(r)
            if changed:
                boosted.sort(key=lambda r: r.score or 0.0, reverse=True)
            return boosted
        except Exception:
            return results

    def _apply_retrieval_boost(
        self,
        query: str,
        results: list[MemoryRecord],
    ) -> list[MemoryRecord]:
        """Multiply each result's score by a curatorial boost (filename / title /
        heading / tag overlap with the query) so a note whose metadata is the
        answer wins decisively over body-text-only matches. Source-agnostic.
        Mirrors the boost already applied to the repo surface (repo_index).

        Skips records whose health confidence is low (< 0.9): a garbled OCR
        screenshot or low-quality import has an auto-generated title and garbage
        body — NOT a curatorial signal — so it must not earn a metadata boost
        that would undo its quality down-weight. Best-effort: failure → unchanged.
        """
        try:
            from memo.retrieval_boost import boost_for
            health = self.store.get_health_batch([r.id for r in results])
            changed = False
            out: list[MemoryRecord] = []
            for r in results:
                h = health.get(r.id)
                if h is not None and h["confidence"] < 0.9:
                    out.append(r)  # untrusted metadata — no curatorial boost
                    continue
                heading = str(r.extra.get("chunk_heading") or "") if r.extra else ""
                b = boost_for(
                    query=query,
                    filename=r.path or "",
                    title=r.title or "",
                    headings=[heading] if heading else None,
                    tags=r.tags or None,
                )
                if abs(b - 1.0) < 1e-6:
                    out.append(r)
                    continue
                out.append(replace(r, score=round((r.score or 0.0) * b, 6)))
                changed = True
            if changed:
                out.sort(key=lambda r: r.score or 0.0, reverse=True)
            return out
        except Exception as exc:
            _log.debug("retrieval_boost failed: %s", exc)
            return results

    def _apply_health_scores(
        self,
        results: list[MemoryRecord],
    ) -> list[MemoryRecord]:
        """Multiply each result's score by its confidence × roi_score.

        Memories with open contradictions (low confidence) rank lower; frequently
        recalled memories (high roi_score) rank higher. No-op for memorias not yet
        in the health table (missing rows default to 1.0 × 1.0 = neutral).
        Best-effort: any failure returns results unchanged.
        """
        try:
            ids = [r.id for r in results]
            health = self.store.get_health_batch(ids)
            if not health:
                return results
            changed = False
            out = []
            for r in results:
                h = health.get(r.id)
                if h is None:
                    out.append(r)
                    continue
                mult = h["confidence"] * h["roi_score"]
                if abs(mult - 1.0) < 1e-6:
                    out.append(r)
                    continue
                out.append(replace(r, score=round((r.score or 0.0) * mult, 6)))
                changed = True
            if changed:
                out.sort(key=lambda r: r.score or 0.0, reverse=True)
            return out
        except Exception as exc:
            _log.debug("health_scores failed: %s", exc)
            return results

    def _record_access(self, ids: list[str]) -> None:
        """Record read/hits for the surfaced memorias (powers LRU/LFU + the
        promotion/demotion lifecycle).

        Done synchronously on the calling thread so it reuses that thread's
        existing thread-local sqlite connection — spawning a worker thread
        would open (and leak) a fresh connection per recall in the daemon.
        The write is a single batched upsert of ≤limit rows under WAL, so it
        is sub-millisecond and never threatens the recall hook's 5s budget.
        Fully guarded: a lock timeout or any failure just skips tracking
        rather than breaking the read path.
        """
        if not ids:
            return
        try:
            self.store.touch(ids)
            # roi_score boost on access is the LEGACY signal (rewards mere
            # surfacing). When the outcome loop owns roi_score
            # (MEMO_OUTCOME_RANKING_ENABLED), skip it so `reconcile_roi` —
            # driven by real grounding outcomes — stays authoritative instead of
            # being washed out by every-surfacing increments.
            if not flag_bool("MEMO_HEALTH_SCORES_DISABLED") and not flag_bool(
                "MEMO_OUTCOME_RANKING_ENABLED"
            ):
                self.store.boost_roi_batch(ids)
        except sqlite3.Error as exc:  # never let access tracking break a read
            _log.debug("access tracking skipped: %s", exc)

    def _cache_read_through(
        self,
        query: str,
        existing: builtins.list[MemoryRecord],
        limit: int,
    ) -> builtins.list[MemoryRecord]:
        """On a local miss/under-fill, pull candidates from the backing store,
        materialize them locally (so the next read is a hit), and merge.

        Materialized entries come FROM the backing store, so they are clean
        (not dirty) — `save()` would mark them dirty under write-back, so we
        clear that flag after. Dedups against `existing` by id. Best-effort:
        any failure returns `existing` unchanged.
        """
        backend = self._cache_backend()
        if backend is None:
            return existing
        try:
            fetched = backend.fetch(query, limit=limit)
        except Exception as exc:
            _log.debug("cache read-through fetch failed: %s", exc)
            return existing
        if not fetched:
            return existing
        have = {r.id for r in existing}
        added: builtins.list[MemoryRecord] = []
        for cand in fetched:
            body = cand.get("body") or ""
            if not body.strip() or cand.get("id") in have:
                continue
            try:
                # `source=memo-cache-fill` makes save() skip the write policy
                # (this entry already mirrors the backing store — see
                # `_apply_write_policy`), so the fill stays clean.
                saved = self.save(
                    content=body,
                    title=cand.get("title") or "",
                    type_=cand.get("type") or "note",
                    tags=cand.get("tags") or [],
                    extra={"source": "memo-cache-fill"},
                    auto_derive=False,
                )
            except Exception as exc:
                _log.debug("cache read-through materialize failed: %s", exc)
                continue
            added.append(replace(saved, score=cand.get("score")))
            have.add(saved.id)
        return (existing + added)[:limit] if added else existing

    # -- list ---------------------------------------------------------------

    def list(
        self,
        *,
        limit: int = 20,
        type_: str | None = None,
        include_forgotten: bool = False,
        updated_since: str | None = None,
    ) -> list[MemoryRecord]:
        """Recent entries by `updated` desc. Body included for each.

        Soft-forgotten memorias (see `forget`) are excluded unless
        `include_forgotten=True`. `updated_since` (ISO-8601) filters at the
        DB level — incremental callers (e.g. contradiction scans) get the
        freshest anchors within `limit` instead of post-filtering a page of
        older rows.
        """
        rows = self.store.list_recent(limit=limit, type_=type_, updated_since=updated_since)
        if not include_forgotten:
            rows = [r for r in rows if not (r.get("extra") or {}).get(IS_FORGOTTEN_KEY)]
        return [
            MemoryRecord(
                id=r["id"],
                path=r["path"],
                title=r["title"],
                type=r["type"],
                tags=r["tags"],
                created=r["created"],
                updated=r["updated"],
                body=self._read_body(r["path"]),
                extra=r.get("extra") or {},
            )
            for r in rows
        ]

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
        return MemoryRecord(
            id=r["id"],
            path=r["path"],
            title=r["title"],
            type=r["type"],
            tags=r["tags"],
            created=r["created"],
            updated=r["updated"],
            body=self._read_body(r["path"]),
            extra=r.get("extra") or {},
        )
