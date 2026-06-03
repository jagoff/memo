"""Read-path operations for `Memory` — search / list / get / resolve_id.

`_SearchOpsMixin` holds the retrieval methods and their helpers (RRF
orchestration, access tracking, cache read-through), moved verbatim from
the former `memory.py` god-file.
"""

from __future__ import annotations

import builtins
import os
import sqlite3
from dataclasses import replace

from memo.perf import timer

from memo.lifecycle import IS_FORGOTTEN_KEY
from memo.memory._base import _MemoryBase
from memo.memory.record import (
    _RECALL_DECAY_HALFLIFE_DEFAULT,
    AmbiguousIdError,
    MemoryRecord,
    _apply_decay,
    _log,
    _rrf_fuse,
)


class _SearchOpsMixin(_MemoryBase):
    # -- search -------------------------------------------------------------

    @timer(log_threshold_ms=50.0)
    def search(
        self, query: str, *, limit: int | None = None, type_: str | None = None,
        mode: str = "hybrid", load_bodies: bool = True, disable_reranker: bool = False,
        recency: bool = False, exclude_types: set[str] | None = None,
        include_forgotten: bool = False, read_through: bool = False,
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
        if not query or not query.strip():
            return []
        limit = limit or self.cfg.search_default_limit

        if mode == "bm25":
            rows = self.store.search_bm25(query, limit=limit, type_=type_, exclude_types=exclude_types)
        elif mode == "fuzzy":
            rows = self.store.search_fuzzy(query, limit=limit, type_=type_, exclude_types=exclude_types)
        elif mode == "vec":
            # Asymmetric retrieval: queries are embedded WITH the
            # instruction prefix; documents are embedded RAW (in
            # `save()` / `update()`). See `_QUERY_INSTRUCTION_PREFIX`
            # in `embedder.py` for the why.
            emb = self.embedder.embed_query(query)
            rows = self.store.search(emb, limit=limit, type_=type_, exclude_types=exclude_types)
        else:
            # hybrid — fetch a wider candidate set from each side and
            # fuse with reciprocal rank fusion (RRF). When the reranker
            # is enabled we widen the input pool to `rerank_input_k` so
            # the cross-encoder has more candidates to discriminate
            # between; the final `limit` is applied AFTER rerank.
            input_k = self.cfg.rerank_input_k if self.cfg.reranker_enabled else limit
            k_each = max(input_k * 2, 30)
            emb = self.embedder.embed_query(query)
            vec_hits = self.store.search(emb, limit=k_each, type_=type_, exclude_types=exclude_types)
            bm_hits = self.store.search_bm25(query, limit=k_each, type_=type_, exclude_types=exclude_types)
            rows = _rrf_fuse(vec_hits, bm_hits, limit=input_k)
        out: list[MemoryRecord] = []
        for r in rows:
            body = self._read_body(r["path"]) if load_bodies else ""
            out.append(
                MemoryRecord(
                    id=r["id"], path=r["path"], title=r["title"], type=r["type"],
                    tags=r["tags"], created=r["created"], updated=r["updated"],
                    body=body, extra=r.get("extra") or {}, score=r.get("score"),
                ),
            )
        # Drop soft-forgotten memorias (forget_after TTL elapsed, see
        # lifecycle.py) before feedback/rerank so they never reach the
        # consumer — recall, ask, chat all route through here. Reversible
        # via `unforget`; pass include_forgotten=True to surface them.
        if out and not include_forgotten:
            out = [r for r in out if not (r.extra or {}).get(IS_FORGOTTEN_KEY)]
        # Source-level feedback (👍 / 👎) — applied AFTER RRF/vec retrieval
        # but BEFORE cross-encoder rerank so the reranker doesn't waste
        # cycles on hits the user already vetoed. Embeds the query once
        # (reusing the vec-mode embedding when available) and consults
        # the `source_feedback` table for each hit. Disabled when
        # MEMO_FEEDBACK_DISABLED=1.
        if out and os.environ.get("MEMO_FEEDBACK_DISABLED") != "1":
            try:
                fb_emb = locals().get("emb")
                if fb_emb is None:
                    fb_emb = self.embedder.embed_query(query)
                out = self._apply_source_feedback(out, fb_emb)
            except Exception as exc:
                _log.warning("source_feedback failed: %s", exc, exc_info=True)

        # Cross-encoder rerank on hybrid mode only. Skipped for vec/bm25
        # since those callers explicitly opted out of fusion entirely;
        # adding rerank to single-mode searches would surprise users
        # benchmarking the raw bi-encoder or BM25 surfaces.
        # Also skipped when disable_reranker=True (e.g., chat synthesis).
        if mode == "hybrid" and self.cfg.reranker_enabled and not disable_reranker and out:
            out = self._rerank(query, out, top_n=limit)
        # Recency decay: blend a freshness bonus into the score so older
        # memories don't crowd out recent ones. MEMO_SEARCH_DECAY_HALFLIFE
        # (days) sets the halflife explicitly; if unset, the consumer paths
        # (recall/ask/chat) still get a sensible default when they pass
        # `recency=True`, while raw `search()` callers (e.g. the eval harness)
        # stay decay-free for a comparable baseline.
        halflife_days = float(os.environ.get("MEMO_SEARCH_DECAY_HALFLIFE", "0") or 0)
        if halflife_days <= 0 and recency:
            halflife_days = _RECALL_DECAY_HALFLIFE_DEFAULT
        if halflife_days > 0 and out:
            alpha = min(max(float(os.environ.get("MEMO_SEARCH_DECAY_ALPHA", "0.15")), 0.0), 1.0)
            out = _apply_decay(out, halflife_days=halflife_days, alpha=alpha)
        # Cache-tier read-through: on a local miss/under-fill, pull from the
        # backing store and materialize locally. OPT-IN per call — only the
        # consumer paths (ask/chat) pass read_through=True. The recall hook
        # never does: a backend subprocess (≤5s) would blow its 5s budget.
        if read_through and self.cache.policy.read_through and len(out) < limit:
            out = self._cache_read_through(query, out, limit)
        # Entity-aware score boost: if query mentions known entities (persons,
        # technologies, projects), boost chunks whose extra["entities"] overlaps.
        # Gated by MEMO_ENTITY_RETRIEVAL_ENABLED. Best-effort: any failure is
        # silent so entity extraction never breaks the search path.
        if out and os.environ.get("MEMO_ENTITY_RETRIEVAL_ENABLED") == "1":
            out = self._apply_entity_boost(query, out)

        # Contradiction penalty: penalise the older side of open contradiction
        # pairs among the retrieved results so stale/superseded memories don't
        # surface at the top. Gated by MEMO_CONTRADICT_PENALTY_ENABLED (default
        # off — the sidecar DB is empty until `memo contradict scan` runs at
        # least once). Only the recall/ask/chat consumer paths benefit; the eval
        # harness and recall-hook budget make this opt-in.
        if out and os.environ.get("MEMO_CONTRADICT_PENALTY_ENABLED") == "1":
            out = self._apply_contradict_penalty(out)
        self._record_access([r.id for r in out])
        return out

    def _apply_contradict_penalty(
        self, results: list[MemoryRecord],
    ) -> list[MemoryRecord]:
        """Apply score penalty to the older side of open contradiction pairs."""
        raw_penalty = os.environ.get("MEMO_CONTRADICT_PENALTY", "0.4")
        try:
            penalty = max(0.0, min(1.0, float(raw_penalty)))
        except ValueError:
            penalty = 0.4
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
            replace(r, score=(r.score or 0.0) * penalty) if r.id in penalise else r
            for r in results
        ]
        # Re-sort by score descending so penalised entries sink naturally.
        penalised.sort(
            key=lambda r: (r.score or 0.0),
            reverse=True,
        )
        return penalised

    def _apply_entity_boost(
        self, query: str, results: list[MemoryRecord],
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
                boosted.sort(key=lambda r: (r.score or 0.0), reverse=True)
            return boosted
        except Exception:
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
        except sqlite3.Error as exc:  # never let access tracking break a read
            _log.debug("access tracking skipped: %s", exc)

    def _cache_read_through(
        self, query: str, existing: builtins.list[MemoryRecord], limit: int,
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
        self, *, limit: int = 20, type_: str | None = None,
        include_forgotten: bool = False,
    ) -> list[MemoryRecord]:
        """Recent entries by `updated` desc. Body included for each.

        Soft-forgotten memorias (see `forget`) are excluded unless
        `include_forgotten=True`.
        """
        rows = self.store.list_recent(limit=limit, type_=type_)
        if not include_forgotten:
            rows = [r for r in rows if not (r.get("extra") or {}).get(IS_FORGOTTEN_KEY)]
        return [
            MemoryRecord(
                id=r["id"], path=r["path"], title=r["title"], type=r["type"],
                tags=r["tags"], created=r["created"], updated=r["updated"],
                body=self._read_body(r["path"]), extra=r.get("extra") or {},
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
            id=r["id"], path=r["path"], title=r["title"], type=r["type"],
            tags=r["tags"], created=r["created"], updated=r["updated"],
            body=self._read_body(r["path"]), extra=r.get("extra") or {},
        )

