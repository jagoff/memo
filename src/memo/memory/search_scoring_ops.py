"""Post-retrieval candidate processing for `Memory` search.

`_SearchScoringMixin` holds the helpers the read path applies AFTER the
initial vec/BM25 fetch: knowledge-graph candidate expansion, contradiction
penalty, entity/retrieval/health re-scoring, access tracking, and cache
read-through. Split out of `search_ops.py` to keep both files under the
repo's 800-line limit. `Memory` inherits this mixin alongside
`_SearchOpsMixin`, so every `self._apply_*` call resolves unchanged via MRO.
"""

from __future__ import annotations

import builtins
import sqlite3
from contextlib import suppress
from dataclasses import replace
from typing import Any

from memo.flags import flag_bool, flag_float, flag_int
from memo.memory._base import _MemoryBase
from memo.memory.record import (
    MemoryRecord,
    _log,
)


class _SearchScoringMixin(_MemoryBase):
    def _fetch_graph_candidates(
        self,
        query: str,
        *,
        limit: int = 20,
        type_: str | None = None,
        exclude_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch memories sharing entities with the query via the knowledge graph.
        Returns a ranked list for RRF fusion.
        """
        try:
            from memo.entity_extractor import extract_entities

            query_entities = extract_entities(query)
            if not query_entities:
                return []

            # Find memories sharing these entities, scored by the summed IDF
            # (rarity) of the shared entities — NOT a raw match count. Raw
            # counting lets a memory that shares several *ubiquitous* entities
            # ("memo", "synapse") outrank one sharing a single discriminating
            # one, flooding the graph leg with generic junk. IDF weighting fixes
            # that: an entity in every memory contributes 0, a rare one a lot.
            # A ubiquitous entity (idf 0) is skipped entirely, so we never even
            # fetch its (huge, useless) memory list.
            from memo.graph_proximity import _idf

            n_docs = 0
            try:
                n_docs = int(self.graph.total_indexed_memories())
            except Exception:
                n_docs = 0
            doc_freqs: dict[str, float] = {}
            if n_docs > 0:
                try:
                    doc_freqs = self.graph.entity_doc_freqs(query_entities)
                except Exception:
                    doc_freqs = {}

            memoria_scores: dict[str, float] = {}
            for ent_name in query_entities:
                w = _idf(doc_freqs.get(ent_name.strip().lower(), 0.0), n_docs) if n_docs > 0 else 1.0
                if w <= 0.0:  # ubiquitous / unknown entity carries no signal
                    continue
                for mid in self.graph.entity_memories(ent_name):
                    memoria_scores[mid] = memoria_scores.get(mid, 0.0) + w

            if not memoria_scores:
                return []

            # Sort by summed-IDF desc (strongest discriminating overlap first).
            sorted_mids = sorted(
                memoria_scores.keys(), key=lambda x: memoria_scores[x], reverse=True
            )

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

            # Sort by summed-IDF descending so that _rrf_fuse, which uses list
            # position as rank, places the most discriminating candidates first.
            # get_batch() returns rows in storage order (not insertion order),
            # so we must re-sort here to get the correct rank ordering.
            filtered.sort(key=lambda r: memoria_scores[r["id"]], reverse=True)
            out = filtered[:limit]

            # Assign a synthetic RRF score so the graph list is on the same
            # scale as vec and BM25 lists.  _rrf_fuse() uses rank (position),
            # not the raw score value, for its own fusion sum; but other
            # post-RRF steps (health multipliers, decay blending) operate on
            # the fused score, so having pre-fusion scores that are integers
            # (1, 2, 3...) rather than [0,1] floats would corrupt any path
            # that inspects the score BEFORE _rrf_fuse runs.
            rrf_k = flag_int("MEMO_RRF_K") or 60
            density_boost = flag_float("MEMO_GRAPH_DENSITY_BOOST") or 0.0
            for rank, r in enumerate(out):
                base_score = 1.0 / (rrf_k + rank + 1)
                # Boost well-connected memories (density reranking).
                if density_boost > 0:
                    with suppress(Exception):
                        degree = self.graph.memory_degree(r["id"])
                        base_score *= 1.0 + (density_boost * degree)
                r["score"] = base_score
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
        """Append graph-adjacent memories not in the primary result set.

        Walks entity edges from the top-3 results (1-hop) via the knowledge
        graph and appends up to 3 connected memories scored at 0.6× the
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
                for ent in self.graph.memory_entities(r.id):
                    key = f"{ent['name']}:{ent['type']}"
                    if key not in seen_entity_keys:
                        seen_entity_keys.add(key)
                        entity_names.append(ent["name"])
            if not entity_names:
                return results

            # Collect connected memory IDs (1-hop via shared entity membership).
            candidate_ids: list[str] = []
            seen_candidates: set[str] = set(existing_ids)
            for entity_name in entity_names[:5]:
                for mem_id in self.graph.entity_memories(entity_name):
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
        """Demote the older side of contradiction AND evolution pairs.

        Contradiction: the older side is likely WRONG — penalise it even when
        only one side surfaced (the strong, default 0.4 penalty).
        Evolution: the older side is merely SUPERSEDED — penalise it (softer,
        default 0.7) ONLY when the newer side also surfaced, so a superseded
        memory still answers when nothing fresher was retrieved. This routes
        the temporal engine's already-detected 'evolved' verdicts into ranking
        instead of letting known-stale facts compete at full score.
        """
        contradict_penalty = max(0.0, min(1.0, flag_float("MEMO_CONTRADICT_PENALTY") or 0.4))
        evolution_penalty = max(0.0, min(1.0, flag_float("MEMO_EVOLUTION_PENALTY") or 0.7))
        ids = [r.id for r in results]
        try:
            pairs = self.contradict_store.pairs_for_ids(ids)
            pairs += self.contradict_store.pairs_for_ids(ids, status="evolved")
        except Exception as exc:
            _log.debug("contradict_penalty pairs_for_ids failed: %s", exc)
            return results
        if not pairs:
            return results
        present = {r.id for r in results}
        id_to_updated: dict[str, str] = {r.id: r.updated for r in results}
        # id -> multiplier; keep the strongest (lowest) penalty if an id is in
        # multiple pairs.
        mult: dict[str, float] = {}

        def _demote(mid: str, pen: float) -> None:
            mult[mid] = min(mult.get(mid, 1.0), pen)

        for pair in pairs:
            rel = (pair.relationship or "").lower()
            a, b = pair.memory_id_a, pair.memory_id_b
            a_ts, b_ts = id_to_updated.get(a, ""), id_to_updated.get(b, "")
            if "contrad" in rel:
                # Only demote when BOTH sides carry a timestamp — otherwise we
                # can't tell which is older and would risk sinking the newer one.
                if a_ts and b_ts:
                    _demote(a if a_ts < b_ts else b, contradict_penalty)
            elif "evolu" in rel and a in present and b in present and a_ts and b_ts:
                # Only when BOTH sides surfaced can we safely demote the older.
                _demote(a if a_ts < b_ts else b, evolution_penalty)
        if not mult:
            return results
        penalised = [
            replace(r, score=(r.score or 0.0) * mult[r.id]) if r.id in mult else r for r in results
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
            _log.exception("retrieval boost failed, returning unboosted results")
            return results

    def _apply_co_recall_boost(self, results: list[MemoryRecord]) -> list[MemoryRecord]:
        """Boost candidates frequently co-recalled with the top hit.

        Memories that have surfaced together in past searches reinforce each
        other: the more often a candidate was co-recalled with the current top
        result, the larger its bump (scaled by the strongest co-recall edge in
        the set, capped at MEMO_CO_RECALL_BOOST_WEIGHT). This is the read side
        of MEMO_GRAPH_CO_RECALL — the write side records the edges in `search`.
        Best-effort: any failure returns the unboosted results.
        """
        if len(results) < 3:
            return results
        try:
            anchor = results[0]
            counts = self.graph.co_recall_counts(anchor.id, [r.id for r in results[1:]])
            max_c = max(counts.values(), default=0)
            if max_c <= 0:
                return results
            weight = flag_float("MEMO_CO_RECALL_BOOST_WEIGHT") or 0.1
            boosted: list[MemoryRecord] = [anchor]
            for r in results[1:]:
                c = counts.get(r.id, 0)
                if c > 0:
                    bump = weight * (c / max_c)
                    r = replace(r, score=round((r.score or 0.0) + bump, 6))
                boosted.append(r)
            boosted.sort(key=lambda r: r.score or 0.0, reverse=True)
            return boosted
        except Exception:
            _log.exception("co-recall boost failed, returning unboosted results")
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
        recalled memories (high roi_score) rank higher. No-op for memories not yet
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
        """Record read/hits for the surfaced memories (powers LRU/LFU + the
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
