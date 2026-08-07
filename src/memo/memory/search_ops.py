"""Read-path operations for `Memory` — search / list / get / resolve_id.

`_SearchOpsMixin` holds the retrieval methods and their helpers (RRF
orchestration, access tracking, cache read-through), moved verbatim from
the former `memory.py` god-file.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

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
    is_derived_chunk_id,
    is_memory_id_prefix,
    record_from_row,
)
from memo.perf import timer
from memo.search_deadline import (
    COST_EMBED_MS,
    COST_EXPANSION_MS,
    COST_GRAPH_SIGNAL_MS,
    Deadline,
)
from memo.store.bm25_queries import _normalize_as_of
from memo.tiers import REFERENCE_TYPES, SENSITIVE_TYPES

# Hard ceiling on an explicit search `limit`. Well above any real request
# (default is single digits), it bounds the candidate-pool / rerank math so an
# unbounded caller value cannot exhaust the shared DB or blow the latency budget.
_MAX_SEARCH_LIMIT = 500

# Pool slack for chunk→parent collapsing: a note's chunks can dominate the
# window, so retrieve this many times `limit` and let the collapse refill the
# result with the next distinct documents.
_CHUNK_PARENT_POOL_FACTOR = 4

if TYPE_CHECKING:
    from memo.store.hype_store import HypeStore


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


def _passes_validity_gate(row: dict[str, Any], as_of: str | None) -> bool:
    """Python mirror of the SQL bi-temporal validity gate (`_validity_filter` in
    `store/bm25_queries.py`) for candidate legs that materialize records OUTSIDE
    the SQL predicate — fact-retrieval and hype-fold both fetch via `store.get`,
    which filters only `deleted_at IS NULL` (no validity gate). Without this, a
    superseded record (interval closed via `invalid_at`, index row + fact_edges
    kept) leaks back into recall through those legs, defeating the feature.

    Semantics match the SQL predicate exactly:
      - default (`as_of` None): drop a row whose interval is CLOSED as of now
        (`invalid_at` set and already `<= now`).
      - `as_of=T`: keep iff `COALESCE(valid_at, created) <= T` AND the interval
        is still open at T (`invalid_at IS NULL OR invalid_at > T`) — half-open.

    Timezone consistency with the SQL legs (made local-offset-correct in
    36dc5904): the `as_of` bound is normalized through the SAME `_normalize_as_of`
    (bare-date → end-of-day, offset-preserving) the SQL path binds; both sides
    are then parsed to timezone-AWARE UTC datetimes and compared as INSTANTS —
    robust to offset/precision skew, never a lexicographic TEXT compare that
    could disagree with a different-offset bound.
    """
    invalid_dt = _parse_search_ts(row.get("invalid_at"))
    if as_of is None:
        if invalid_dt is None:
            return True
        return invalid_dt > datetime.now(UTC)
    bound = _parse_search_ts(_normalize_as_of(as_of))
    if bound is None:
        # Malformed as_of — mirror the SQL fallback's leniency: never drop on a
        # boundary we can't parse (and never crash).
        return True
    valid_start = _parse_search_ts(row.get("valid_at")) or _parse_search_ts(row.get("created"))
    if valid_start is not None and valid_start > bound:
        return False
    return invalid_dt is None or invalid_dt > bound


def _compact_fact_edge(fact: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": fact.get("id") or "",
        "subject": fact.get("subject") or "",
        "predicate": fact.get("predicate") or "",
        "object": fact.get("object") or "",
        "source_record_id": fact.get("source_record_id") or "",
        "valid_at": fact.get("valid_at"),
        "invalid_at": fact.get("invalid_at"),
        "expired_at": fact.get("expired_at"),
        "confidence": fact.get("confidence"),
        "score": fact.get("score"),
        "provenance": fact.get("provenance") or {},
        "metadata": fact.get("metadata") or {},
    }


class _SearchOpsMixin(_MemoryBase):
    # Lazily-built HyPE question index (MEMO_HYPE_ENABLED read path). Created
    # on first folded search, reused afterwards; never constructed flag-off.
    _hype_store: HypeStore | None = None
    # Cached result of the one-time variant-mismatch check below — computed
    # once per Memory instance (cheap `stats()` query), not per search.
    _hype_variant_checked: bool = False
    _hype_variant_warning: str | None = None

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
        as_of: str | None = None,
        _track_usage: bool = True,
        _trace: list[dict[str, Any]] | None = None,
        _budget_ms: int | None = None,
        _degraded: list[str] | None = None,
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
            as_of: ISO date/datetime for valid-time recall. When set, the
                candidate seams filter to records whose world-validity interval
                CONTAINS `as_of` (`COALESCE(valid_at, created) <= as_of` and not
                yet invalidated at `as_of`), overriding the default now-gate — so
                a since-superseded fact resurfaces as it stood at that time.
            _track_usage: Internal callers may disable access/co-recall writes
                when search is only a derived processing step rather than a
                user-visible retrieval.
            _degraded: Out-parameter, same shape as `_trace`. Pass a list and
                every stage shed to stay inside the wall-clock budget appends
                its name to it. Degrade and say so — a search that quietly
                dropped its reranker looks identical to one that ran it.
            _budget_ms: Override for MEMO_SEARCH_BUDGET_MS. 0 = no deadline.
        """

        def _add_trace(stage: str, **data: Any) -> None:
            if _trace is not None:
                _trace.append({"stage": stage, **data})

        def _shed(stage: str) -> None:
            """Record one rung of the shed ladder. Only ever called for a stage
            that WOULD have run — a stage held back by its own flag, or absent
            from this mode, must never be reported as budget-degraded."""
            if _degraded is not None:
                _degraded.append(stage)

        # Rung zero: the clock every rung below consults. Started before any
        # stage so the budget covers the whole search, not the tail of it.
        _deadline = Deadline.start(_budget_ms)

        if not query or not query.strip():
            _add_trace("candidate_generation", mode=mode, output_count=0)
            _add_trace("final", output_count=0)
            return []
        # Avoid loading an embedder/reranker merely to prove that an isolated
        # or newly-created store has no candidates. Read-through searches must
        # continue so their backing tier gets a chance to materialize results.
        if not read_through and self.store.count() == 0:
            _add_trace("candidate_generation", mode=mode, output_count=0)
            _add_trace("final", output_count=0)
            return []
        # Credentials are managed only through the explicit secret API. Legacy
        # `type: secret` rows must never be returned by any search mode, even if
        # a caller asks for that type or forgets to pass an exclusion set.
        exclude_types = set(exclude_types or ()) | set(SENSITIVE_TYPES)
        limit = limit or self.cfg.search_default_limit
        # Clamp an explicit caller `limit` to a hard ceiling. It drives pool-
        # widening math (input_k → k_each) that becomes the LIMIT of three
        # sqlite-vec/BM25 scans plus the cross-encoder rerank top_n, so an
        # unbounded value (e.g. `memo_search(limit=1_000_000)` — the MCP tool has
        # no ge/le, unlike the HTTP route) is a self-inflicted resource-exhaustion
        # vector against the shared DB / recall daemon and blows the latency budget.
        limit = max(1, min(int(limit), _MAX_SEARCH_LIMIT))
        if date_to and len(date_to) == 10:
            date_to = date_to + "T23:59:59"  # bare date = whole day inclusive
        emb = None  # set in vec/hybrid branches; consumed by feedback boost below

        if mode == "bm25":
            rows = self.store.search_bm25(
                query, limit=limit, type_=type_, exclude_types=exclude_types, as_of=as_of
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
                as_of=as_of,
            )
            _add_trace(
                "candidate_generation", mode=mode, bm25_count=len(rows), output_count=len(rows)
            )
        elif mode == "fuzzy":
            rows = self.store.search_fuzzy(
                query, limit=limit, type_=type_, exclude_types=exclude_types, as_of=as_of
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
                as_of=as_of,
            )
            _add_trace(
                "candidate_generation", mode=mode, vec_count=len(rows), output_count=len(rows)
            )
            # HyPE fold: match the query against the nightly question-space
            # index and max-fold the results into the doc candidates (raises
            # scores AND appends memories the doc vector alone missed).
            # Default OFF — with the flag unset this branch is a single
            # flag_bool check, no HypeStore construction, no extra reads.
            if flag_bool("MEMO_HYPE_ENABLED"):
                before_hype = len(rows)
                rows = self._hype_fold_candidates(
                    rows, emb, limit, type_=type_, exclude_types=exclude_types, as_of=as_of
                )
                variant_warning = self._hype_variant_mismatch_warning()
                _add_trace(
                    "hype_fold",
                    input_count=before_hype,
                    output_count=len(rows),
                    **({"warning": variant_warning} if variant_warning else {}),
                )
        else:
            emb = None  # set below in hybrid's vec branch; used by feedback boost
            # hybrid — fetch a wider candidate set from each side and
            # fuse with reciprocal rank fusion (RRF). When the reranker
            # is enabled we widen the input pool to `rerank_input_k` so
            # the cross-encoder has more candidates to discriminate
            # between; the final `limit` is applied AFTER rerank.
            input_k = max(self.cfg.rerank_input_k, limit) if self.cfg.reranker_enabled else limit
            # Chunk→parent collapsing removes hits by design (eight chunks of
            # one note become one). Without slack in the pool the caller gets
            # fewer results than it asked for, so widen it enough that the
            # collapse can refill from the next distinct documents.
            if flag_bool("MEMO_SEARCH_CHUNK_PARENT") and type_ not in REFERENCE_TYPES:
                input_k = max(input_k, limit * _CHUNK_PARENT_POOL_FACTOR)
            k_each = max(input_k * 2, 20)
            try:
                _query_for_embed = query
                # HyDE: generate hypothetical answer doc, embed that instead of raw query
                _hyde_enabled = flag_bool("MEMO_HYDE_ENABLED")
                # Rung two: query expansion. HyDE is an LLM generation layered ON
                # TOP of the embed it feeds, so affording it means affording both
                # — generating a hypothetical document and then having no budget
                # left to embed it is the worst of both. Hence the summed cost,
                # which also makes this rung shed strictly before rung four:
                # under pressure the search loses the enrichment first and the
                # semantic leg only after that.
                if _hyde_enabled and not _deadline.afford(COST_EXPANSION_MS + COST_EMBED_MS):
                    _hyde_enabled = False
                    _add_trace("hyde_skip", reason="budget")
                    _shed("expansion_skipped")
                if _hyde_enabled:
                    _log.debug("HyDE enabled, generating hypothetical doc")
                    _hyde_doc = self._generate_hyde_document(query)
                    if _hyde_doc:
                        _query_for_embed = _hyde_doc
                        _add_trace("hyde", original_query=query, hyde_doc=_hyde_doc[:100] + "…")
                        _log.info("HyDE doc generated: %s", _hyde_doc[:100])
                    else:
                        _log.warning("HyDE returned empty doc, falling back to original query")
                # Rung four: the vec leg's embed. BM25-only is the fallback the
                # `except` below already reaches for when the embedder is
                # unavailable — the difference is that this one is a decision,
                # not an accident, and says so via `_degraded`. An MLX embed
                # that cannot finish inside the budget is the 300s hang.
                #
                # The check is taken HERE, at the decision, and never hoisted
                # above the HyDE branch. `_generate_hyde_document` is an
                # un-timeboxed `chat.chat()` (the ChatBackend protocol in
                # llm.py takes no deadline), so it can burn arbitrary
                # wall-clock. A snapshot taken before it would be guaranteed
                # True on this branch — rung two only lets HyDE run when
                # `afford(COST_EXPANSION_MS + COST_EMBED_MS)` held, which
                # implies `afford(COST_EMBED_MS)` — making rung four
                # unreachable exactly when the budget was actually spent. Same
                # discipline as `_rerank`, which re-derives `budget_s` from a
                # fresh `remaining_ms()` after the reranker's cold load.
                if not _deadline.afford(COST_EMBED_MS):
                    vec_hits = []
                    _add_trace("embed_skip", mode="hybrid", reason="budget")
                    _shed("embed_skipped_bm25_only")
                else:
                    emb = self.embedder.embed_query(_query_for_embed)
                    vec_hits = self.store.search(
                        emb,
                        limit=k_each,
                        type_=type_,
                        exclude_types=exclude_types,
                        date_from=date_from,
                        date_to=date_to,
                        exclude_tags=exclude_tags,
                        as_of=as_of,
                    )
                    if flag_bool("MEMO_HYPE_ENABLED"):
                        before_hype = len(vec_hits)
                        vec_hits = self._hype_fold_candidates(
                            vec_hits,
                            emb,
                            k_each,
                            type_=type_,
                            exclude_types=exclude_types,
                            exclude_tags=exclude_tags,
                            as_of=as_of,
                        )
                        variant_warning = self._hype_variant_mismatch_warning()
                        _add_trace(
                            "hype_fold",
                            mode="hybrid",
                            input_count=before_hype,
                            output_count=len(vec_hits),
                            **({"warning": variant_warning} if variant_warning else {}),
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
                query, limit=k_each, type_=type_, exclude_types=exclude_types, as_of=as_of
            )
            exact_hits = self.store.search_bm25(
                query,
                limit=k_each,
                type_=type_,
                exclude_types=exclude_types,
                field_boost="exact",
                as_of=as_of,
            )

            fact_hits = (
                self._fetch_fact_candidates(
                    query,
                    limit=k_each,
                    type_=type_,
                    exclude_types=exclude_types,
                    exclude_tags=exclude_tags,
                    as_of=as_of,
                )
                if flag_bool("MEMO_FACT_RETRIEVAL_ENABLED")
                else []
            )

            # Fuse all sources. RRF supports multiple ranked lists. `k` is
            # configurable (MEMO_RRF_K, default 60); MEMO_RRF_ADAPTIVE opts
            # into density-driven k (sharper on agreement, softer when the
            # lists diverge) — off by default so the eval baseline holds.
            base_k_flag = flag_int("MEMO_RRF_K")
            base_k = 25 if base_k_flag is None else base_k_flag
            rrf_lists = [vec_hits, bm_hits, exact_hits]
            if fact_hits:
                rrf_lists.append(fact_hits)
            rrf_k = (
                _adaptive_rrf_k(rrf_lists, base_k=base_k)
                if flag_bool("MEMO_RRF_ADAPTIVE")
                else base_k
            )

            # Per-leg weights: MEMO_SEARCH_VEC_WEIGHT / MEMO_SEARCH_BM25_WEIGHT
            # allow the user to tilt fusion toward semantic or keyword retrieval.
            # Defaults (0.5 each) preserve the historical equal-weight behaviour.
            # Exact BM25 reuses the keyword weight: it is still lexical evidence,
            # just stricter and metadata-boosted.
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
            fact_weight = flag_float("MEMO_FACT_RETRIEVAL_WEIGHT")
            # Build weight list aligned with the lists passed to _rrf_fuse:
            # [vec, bm25, exact, facts] → [w_vec, w_bm25, w_bm25, fact_weight]
            rrf_weights = [w_vec, w_bm25, w_bm25]
            if fact_hits:
                rrf_weights.append(fact_weight or 0.0)

            rows = _rrf_fuse(
                *rrf_lists,
                limit=input_k,
                k=rrf_k,
                weights=rrf_weights,
            )
            # Vector search always returns nearest neighbours, even for
            # nonsense queries. Keep vector-only hits only when they clear a
            # calibrated similarity floor; lexical/fact evidence corroborates
            # a result and therefore bypasses the floor.
            min_vec_score = flag_float("MEMO_HYBRID_MIN_VEC_SCORE")
            # Short keyword-like prompts benefit from semantic fallback (and
            # usually have an exact lexical signal); abstention targets the
            # multi-token, open-ended/nonsense queries that create false hits.
            if min_vec_score is not None and min_vec_score > 0 and len(query.split()) >= 3:
                lexical_ids = {
                    str(hit.get("id"))
                    for hit in (*bm_hits, *exact_hits, *fact_hits)
                    if hit.get("id")
                }
                vec_scores = {
                    str(hit.get("id")): float(hit.get("score") or 0.0)
                    for hit in vec_hits
                    if hit.get("id")
                }
                before_abstention = len(rows)
                rows = [
                    row
                    for row in rows
                    if str(row.get("id")) in lexical_ids
                    or vec_scores.get(str(row.get("id")), 0.0) >= min_vec_score
                ]
                _add_trace(
                    "abstention",
                    input_count=before_abstention,
                    output_count=len(rows),
                    dropped_count=before_abstention - len(rows),
                    min_vec_score=min_vec_score,
                )
            _add_trace(
                "candidate_generation",
                mode="hybrid",
                vec_count=len(vec_hits),
                bm25_count=len(bm_hits),
                exact_count=len(exact_hits),
                fact_count=len(fact_hits),
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
            rows = [r for r in rows if not (exclude_tags & {str(t) for t in (r.get("tags") or [])})]
        out: list[MemoryRecord] = []
        # In hybrid mode the candidate pool can grow large (up to _POOL_CAP) and
        # the reranker trims it to `limit`. Loading every candidate body from
        # disk (open + frontmatter parse per hit) before that trim wastes up to
        # ~200 filesystem round-trips per search. Feed the pool's body text from
        # the FTS index in ONE batched query instead; the canonical .md body is
        # re-resolved from disk only for the surviving `limit` records right
        # before return (see `_resolve_disk_bodies` below). The reranker only
        # needs text, so this is a pure latency win with no ranking change.
        # `load_bodies=False` defers the per-candidate DISK read; it must not
        # starve the scoring stages of text. It used to gate this batched FTS
        # fetch too, so ask/chat handed the cross-encoder empty bodies and it
        # ranked on titles alone — on the live corpus that pushed an answer
        # from rank 2 to rank 21 and `memo ask` refused a question it had the
        # answer for. Bodies are blanked again before return (below) so the
        # lazy contract still holds for the caller.
        _bodies_from_fts = mode == "hybrid"
        _fts_bodies: dict[str, str] = (
            self.store.get_fts_bodies([r["id"] for r in rows]) if _bodies_from_fts else {}
        )
        for r in rows:
            if _bodies_from_fts:
                body = _fts_bodies.get(r["id"], "")
            elif not load_bodies:
                body = ""
            else:
                body = self._read_body(r["path"])
            out.append(record_from_row(r, body=body))
        _add_trace(
            "materialize", input_count=len(rows), output_count=len(out), load_bodies=load_bodies
        )
        if out and flag_bool("MEMO_FACT_SURFACE_ENABLED"):
            before = len(out)
            out = self._attach_related_fact_edges(query, out)
            _add_trace("fact_surface", input_count=before, output_count=len(out))
        # Drop soft-forgotten memories (forget_after TTL elapsed, see
        # lifecycle.py) before feedback/rerank so they never reach the
        # consumer — recall, ask, chat all route through here. Reversible
        # via `unforget`; pass include_forgotten=True to surface them.
        if out and not include_forgotten:
            before = len(out)
            out = [r for r in out if not (r.extra or {}).get(IS_FORGOTTEN_KEY)]
            _add_trace("forget_filter", input_count=before, output_count=len(out))
        # Chunk→parent mapping (MEMO_SEARCH_CHUNK_PARENT, default off) runs on
        # the WIDE pool, before rerank and the trim to `limit`. Collapsing after
        # the trim could only shrink the result: a long note whose eight chunks
        # all rank well returned eight near-identical hits, and enabling the
        # flag turned those eight into one instead of one plus the next seven
        # distinct documents. Collapsing first also stops the reranker from
        # spending its window on fragments of a single note.
        if out and flag_bool("MEMO_SEARCH_CHUNK_PARENT") and type_ not in REFERENCE_TYPES:
            before = len(out)
            out = self._map_chunks_to_parents(out)
            _add_trace("chunk_parent", input_count=before, output_count=len(out))
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
            # Rung one: the most expensive stage, and the only one reached here
            # — `_reranker_will_run` already proved the reranker was going to
            # run, so a skip inside `_rerank` is always a budget decision.
            out = self._rerank(query, out, top_n=limit, deadline=_deadline, degraded=_degraded)
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
        if out and flag_bool("MEMO_VERIFICATION_STATE_TRACKING"):
            before = len(out)
            out = self._apply_verification_decay(out)
            _add_trace("verification_decay", input_count=before, output_count=len(out))
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
        out = self._apply_curated_graph_order(
            query, out, _add_trace, deadline=_deadline, degraded=_degraded
        )
        if _track_usage:
            self._record_access([r.id for r in out])
        # Co-recall graph edges: record which memories surface together.
        # Gated by flag so the graph DB write stays opt-in (off by default).
        if _track_usage and len(out) >= 2 and flag_bool("MEMO_GRAPH_CO_RECALL"):
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

            if not load_bodies:
                # The caller asked to defer body loading: the FTS text was for
                # scoring only, so hand back unloaded records and let it resolve
                # the canonical .md for whatever it keeps.
                out = [dataclasses.replace(r, body="") for r in out]
            else:
                resolved: list[MemoryRecord] = []
                for r in out:
                    disk = self._read_body(r.path)
                    resolved.append(dataclasses.replace(r, body=disk) if disk else r)
                out = resolved
        # Judged relations are compact, derived annotations. Pending candidates
        # stay out of normal recall and are visible only in review surfaces.
        if out:
            out = self.annotate_relations(out)
        _add_trace("final", output_count=len(out), limit=limit)
        return out

    def _apply_curated_graph_order(
        self,
        query: str,
        out: list[MemoryRecord],
        trace: Any,
        *,
        deadline: Deadline | None = None,
        degraded: list[str] | None = None,
    ) -> list[MemoryRecord]:
        """Apply one identity-safe graph ordering pass without changing scores.

        `deadline`, when given, is rung three of the shed ladder: the budget
        check sits AFTER both enable gates, so a stage held back by its own
        flag is never reported as budget-degraded. Skipping costs the ordering
        pass only — the fused order the records already carry is served.
        """
        if not out or not flag_bool("MEMO_GRAPH_SIGNAL_ENABLED"):
            return out
        if not flag_bool("MEMO_GRAPH_PROJECTION_ENABLED"):
            trace("graph_signal", enabled=True, touched_count=0, skipped="projection_disabled")
            return out
        if deadline is not None and not deadline.afford(COST_GRAPH_SIGNAL_MS):
            trace("graph_signal", enabled=True, touched_count=0, skipped="budget")
            if degraded is not None:
                degraded.append("graph_signal_skipped")
            return out
        try:
            from memo.graph_reason import build_graph_reason
            from memo.graph_signal import collect_graph_signal, config_from_flags

            config = config_from_flags()
            model = self.graph.projection.read_model(config.max_age_hours)
            result = collect_graph_signal(
                model,
                query,
                [record.id for record in out],
                config=config,
            )
            trace(
                "graph_signal",
                enabled=result.enabled,
                projection_version=model.version,
                query_nodes=list(result.query_nodes),
                touched_count=len(result.signals),
                skipped=result.skipped,
                elapsed_ms=round(result.elapsed_ms, 3),
            )
            if not result.signals:
                return out
            by_id = {record.id: record for record in out}
            ordered = [by_id[memory_id] for memory_id in result.ordered_ids]
            if not flag_bool("MEMO_GRAPH_REASON_ENABLED"):
                return ordered
            relations_by_id: dict[str, list[dict[str, Any]]] = {}
            if flag_bool("MEMO_GRAPH_SEMANTIC_RELATIONS"):
                relations_by_id = {
                    memory_id: self.graph.semantic_relations_for(
                        source_id=memory_id,
                        limit=10,
                    )
                    for memory_id in result.traces
                }
            return [
                dataclasses.replace(
                    record,
                    extra={
                        **(record.extra or {}),
                        "graph_reason": build_graph_reason(
                            record.id,
                            result.traces[record.id],
                            relations=relations_by_id.get(record.id),
                        ),
                    },
                )
                if record.id in result.traces
                else record
                for record in ordered
            ]
        except Exception as exc:
            _log.debug("graph_signal failed: %s", exc)
            trace("graph_signal", enabled=True, touched_count=0, skipped="error")
            return out

    def _hype_fold_candidates(
        self,
        rows: list[dict[str, Any]],
        emb: list[float],
        limit: int,
        *,
        type_: str | None = None,
        exclude_types: set[str] | None = None,
        exclude_tags: set[str] | None = None,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        """Max-fold HyPE question-space candidates into the vec doc hits.

        Lazily constructs the `HypeStore` once per `Memory` instance (imports
        deferred so the flag-off hot path pays nothing beyond the flag check).
        Best-effort: any failure degrades to the unfolded doc hits — the
        question index is derived data and must never break search.

        `type_`/`exclude_types` are the same filters `search()` applied at the
        SQL level for the doc hits — `fetch_meta` must re-apply them for
        fold-appended candidates (question-space matches not already in
        `rows`), since those are materialized straight from `self.store.get`
        with no filter awareness of its own. Existing doc hits are untouched:
        `hype_fold` only calls `fetch_meta` for candidates NOT already present.
        """
        try:
            from memo.hype_fold import hype_fold
            from memo.store.hype_store import HypeStore

            store = self._hype_store
            if store is None:
                store = HypeStore(
                    self.cfg.db_path,
                    self.cfg.embedder_dims,
                    embedder_model=self.store.embedder_model,
                )
                self._hype_store = store
            pool = flag_int("MEMO_HYPE_FOLD_POOL") or 30
            oversample = flag_int("MEMO_HYPE_FOLD_OVERSAMPLE") or 2

            def _fetch_meta_filtered(memory_id: str) -> dict[str, Any] | None:
                row: dict[str, Any] | None = self.store.get(memory_id)
                if row is None:
                    return None
                # Same validity gate the vec doc leg applies in SQL — a
                # fold-appended candidate is materialized straight from
                # `store.get` (deleted_at-only), so re-check it here.
                if not _passes_validity_gate(row, as_of):
                    return None
                if type_ and row.get("type") != type_:
                    return None
                if exclude_types and row.get("type") in exclude_types:
                    return None
                if exclude_tags and exclude_tags.intersection(row.get("tags") or ()):
                    return None
                return row

            return hype_fold(
                rows,
                emb,
                store,
                _fetch_meta_filtered,
                pool=pool * max(1, oversample),
                limit=limit,
            )
        except Exception as exc:
            _log.warning("hype_fold failed, using doc hits: %s", exc, exc_info=True)
            return rows

    def _hype_variant_mismatch_warning(self) -> str | None:
        """One cheap `stats()` query (cached for the life of this `Memory`
        instance) checking whether every HyPE row uses the currently active
        `MEMO_HYPE_EMBED_RAW` embedding variant.

        A mismatch means some stored question vectors were embedded on a
        different scale (query-prefixed vs raw) than what live queries use —
        the fold still runs (never hard-fails search), this only surfaces a
        trace note so the mismatch is diagnosable instead of silently
        degrading scores. Resolved by `memo dream hype --reembed`.
        """
        if self._hype_variant_checked:
            return self._hype_variant_warning
        self._hype_variant_checked = True
        try:
            from memo.dream_hype import _active_variant

            store = self._hype_store
            if store is None:
                return None  # not yet constructed this call; nothing to check
            by_variant = store.stats().get("by_variant", {})
            if not by_variant:
                return None
            active = _active_variant()
            stale_variants = sorted(variant for variant in by_variant if variant != active)
            if stale_variants:
                counts = ", ".join(
                    f"{variant}={by_variant[variant]}" for variant in sorted(by_variant)
                )
                warning = (
                    f"hype store variants ({counts}) include values != active={active!r} "
                    f"(run `memo dream hype --reembed`)"
                )
                self._hype_variant_warning = warning
                _log.warning(warning)
                return warning
            return None
        except (sqlite3.Error, OSError, RuntimeError, TypeError, ValueError) as exc:
            _log.debug("hype variant check skipped: %s", exc)
            return None

    def _fetch_fact_candidates(
        self,
        query: str,
        *,
        limit: int,
        type_: str | None = None,
        exclude_types: set[str] | None = None,
        exclude_tags: set[str] | None = None,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        try:
            # `as_of` time-travels the fact-edge selection (its own bi-temporal
            # columns), mirroring how the vec/bm25 legs thread it into their SQL.
            facts = self.fact_edges.search_text(query, as_of=as_of, limit=limit * 3)
        except Exception as exc:
            _log.debug("fact retrieval skipped: %s", exc)
            return []
        best_by_id: dict[str, dict[str, Any]] = {}
        facts_by_id: dict[str, list[dict[str, Any]]] = {}
        for fact in facts:
            rid = str(fact.get("source_record_id") or "")
            if not rid:
                continue
            row = self.store.get(rid)
            if row is None:
                continue
            # `store.get` filters only `deleted_at IS NULL`; re-apply the SAME
            # validity gate the SQL legs enforce so a superseded (or not-yet /
            # no-longer valid at `as_of`) source record can't leak in here.
            if not _passes_validity_gate(row, as_of):
                continue
            if type_ and row.get("type") != type_:
                continue
            if exclude_types and row.get("type") in exclude_types:
                continue
            tags = {str(t) for t in (row.get("tags") or [])}
            if exclude_tags and (exclude_tags & tags):
                continue
            facts_by_id.setdefault(rid, []).append(_compact_fact_edge(fact))
            score = float(fact.get("score") or 0.0) * float(fact.get("confidence") or 1.0)
            if rid not in best_by_id or score > float(best_by_id[rid].get("score") or 0.0):
                candidate = dict(row)
                extra = dict(candidate.get("extra") or {})
                extra["fact_edge_matched"] = True
                candidate["extra"] = extra
                candidate["score"] = score
                best_by_id[rid] = candidate
        out = sorted(best_by_id.values(), key=lambda row: row.get("score") or 0.0, reverse=True)
        for row in out:
            rid = str(row.get("id") or "")
            extra = dict(row.get("extra") or {})
            extra["related_fact_edges"] = facts_by_id.get(rid, [])[:5]
            row["extra"] = extra
        return out[:limit]

    def _attach_related_fact_edges(
        self, query: str, records: list[MemoryRecord]
    ) -> list[MemoryRecord]:
        try:
            facts = self.fact_edges.search_text(query, limit=max(len(records) * 5, 20))
        except Exception as exc:
            _log.debug("fact surface skipped: %s", exc)
            return records
        by_source: dict[str, list[dict[str, Any]]] = {}
        for fact in facts:
            rid = str(fact.get("source_record_id") or "")
            if rid:
                by_source.setdefault(rid, []).append(_compact_fact_edge(fact))
        out: list[MemoryRecord] = []
        for rec in records:
            related = by_source.get(rec.id)
            if not related:
                out.append(rec)
                continue
            extra = dict(rec.extra or {})
            existing = extra.get("related_fact_edges")
            merged = list(existing) if isinstance(existing, list) else []
            seen = {str(item.get("id") or "") for item in merged if isinstance(item, dict)}
            for fact in related:
                fid = str(fact.get("id") or "")
                if fid and fid in seen:
                    continue
                merged.append(fact)
                if fid:
                    seen.add(fid)
            extra["related_fact_edges"] = merged[:5]
            extra["fact_edge_matched"] = bool(extra["related_fact_edges"])
            out.append(dataclasses.replace(rec, extra=extra))
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
        # Derived reference chunks intentionally append a controlled suffix to
        # their canonical parent id. They are exact-only (never prefix-resolved)
        # but remain first-class anchors for get/around.
        if is_derived_chunk_id(id_or_prefix):
            return id_or_prefix if self.store.get(id_or_prefix) is not None else None
        if not is_memory_id_prefix(id_or_prefix):
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
            raw_seq = extra.get("chunk_seq", extra.get("chunk_index"))
            seq = int(raw_seq) if raw_seq is not None else None
        except (TypeError, ValueError):
            seq = None
        parent = extra.get("parent_path")
        if parent and seq is not None:
            rows = self.store.chunks_adjacent(str(parent), seq, before=before, after=after)
            mode = "chunk_seq"
        else:
            rows = self.store.records_around_created(
                str(row["created"]),
                before=before,
                after=after,
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
