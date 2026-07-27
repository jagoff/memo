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
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, cast

from memo.flags import flag_bool, flag_float
from memo.memory._base import _MemoryBase
from memo.memory.record import (
    MemoryRecord,
    _log,
    _state_decay_factor,
)


def _older_id(a: str, a_ts: str, b: str, b_ts: str) -> str | None:
    """Return the id of the older side of a pair by aware-datetime compare.

    ``updated`` is LOCAL-offset ISO (``record._now_iso`` uses ``.astimezone()``),
    so a raw lexicographic ``a_ts < b_ts`` inverts across differing UTC
    offsets/DST and would demote the wrong (newer) side. Parse both and compare
    timezone-aware instants instead (naive strings are assumed UTC). Returns
    ``None`` when either timestamp is empty/unparseable so the caller can skip
    the pair rather than guess which side is older. A tie picks ``b`` — matching
    the prior ``a if a_ts < b_ts else b`` behavior when the two are equal.
    """

    def _parse(ts: str) -> datetime | None:
        try:
            dt = datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            return None
        return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt

    a_dt, b_dt = _parse(a_ts), _parse(b_ts)
    if a_dt is None or b_dt is None:
        return None
    return a if a_dt < b_dt else b


class _SearchScoringMixin(_MemoryBase):
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
        # Both flags are registered, bounded floats.  Cast away the generic
        # accessor's optional type without an ``or`` fallback, which would
        # incorrectly discard the explicitly supported value 0.0.
        contradict_penalty = cast(float, flag_float("MEMO_CONTRADICT_PENALTY"))
        evolution_penalty = cast(float, flag_float("MEMO_EVOLUTION_PENALTY"))
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
            mult[mid] = min(mult[mid], pen) if mid in mult else pen

        # declare-disputes: when both sides of a competing/open pair surface in
        # the same result set, keep BOTH at full score instead of silently
        # demoting the older side — the dispute is exposed via the hit
        # dossier's ⚔ marker (Task 1), so no duplicate state is written here.
        declared: set[str] = set()
        if flag_bool("MEMO_DECLARE_DISPUTES"):
            try:
                declare_pairs = self.contradict_store.pairs_for_ids(
                    ids, status="competing"
                ) + self.contradict_store.pairs_for_ids(ids, status="open")
            except Exception as exc:
                _log.debug("declare_disputes pairs_for_ids failed: %s", exc)
                declare_pairs = []
            for _p in declare_pairs:
                if _p.memory_id_a in present and _p.memory_id_b in present:
                    declared.add(_p.memory_id_a)
                    declared.add(_p.memory_id_b)

        for pair in pairs:
            rel = str(pair.relationship).lower()
            a, b = pair.memory_id_a, pair.memory_id_b
            a_ts, b_ts = id_to_updated.get(a), id_to_updated.get(b)
            if "contrad" in rel:
                # Only demote when BOTH sides carry a timestamp — otherwise we
                # can't tell which is older and would risk sinking the newer one.
                if a_ts and b_ts:
                    target = _older_id(a, a_ts, b, b_ts)
                    if target is not None and target not in declared:
                        _demote(target, contradict_penalty)
            elif "evolu" in rel and a in present and b in present and a_ts and b_ts:
                # Only when BOTH sides surfaced can we safely demote the older.
                target = _older_id(a, a_ts, b, b_ts)
                if target is not None and target not in declared:
                    _demote(target, evolution_penalty)
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

    def _apply_quality_rerank(self, results: list[MemoryRecord]) -> list[MemoryRecord]:
        """Quality-aware reranking for explicit search/ask paths.

        Default-off via MEMO_QUALITY_RERANK. Best-effort: malformed optional
        quality metadata never breaks retrieval.
        """
        if not flag_bool("MEMO_QUALITY_RERANK"):
            return results
        try:
            from memo.quality import apply_quality_rerank

            return apply_quality_rerank(results)
        except Exception as exc:
            _log.debug("quality_rerank failed: %s", exc)
            return results

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
            weight = cast(float, flag_float("MEMO_CO_RECALL_BOOST_WEIGHT"))
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

    def _apply_verification_decay(
        self,
        results: list[MemoryRecord],
    ) -> list[MemoryRecord]:
        """Multiply each result's score by its verification-state decay factor
        (VERIFIED≈1.0, STALE 0.7, UNVERIFIED 0.8 — see `_state_decay_factor`),
        so freshly-verified facts outrank stale/unverified ones.

        `verification_state` + `verified_at` ride on the MemoryRecord (the meta
        SELECT already carries them), so this is a pure in-memory pass with no
        extra lookups — safe on the recall hot path. No-op for an all-UNVERIFIED
        corpus (uniform 0.8 → order unchanged). Gated by the caller
        (MEMO_VERIFICATION_STATE_TRACKING). Best-effort: any failure returns
        results unchanged.
        """
        try:
            changed = False
            out = []
            for r in results:
                factor = _state_decay_factor(r)
                if abs(factor - 1.0) < 1e-6:
                    out.append(r)
                    continue
                out.append(replace(r, score=round((r.score or 0.0) * factor, 6)))
                changed = True
            if changed:
                out.sort(key=lambda r: r.score or 0.0, reverse=True)
            return out
        except Exception as exc:
            _log.debug("verification_decay failed: %s", exc)
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
        if len(existing) >= limit:
            return existing[: max(limit, 0)]
        try:
            fetched = backend.fetch(query, limit=limit)
        except Exception as exc:
            _log.debug("cache read-through fetch failed: %s", exc)
            return existing
        if not fetched:
            return existing
        have = {r.id for r in existing}
        have_backend_ids = {
            str(backend_id)
            for r in existing
            if (backend_id := (r.extra or {}).get("cache_backend_id"))
        }
        added: builtins.list[MemoryRecord] = []
        remaining_slots = limit - len(existing)
        for cand in fetched:
            if len(added) >= remaining_slots:
                break
            existing, materialized = self._materialize_cache_candidate(
                cand,
                existing=existing,
                have=have,
                have_backend_ids=have_backend_ids,
            )
            if materialized is not None:
                added.append(materialized)
        return (existing + added)[:limit] if added else existing

    def _materialize_cache_candidate(
        self,
        cand: dict[str, Any],
        *,
        existing: builtins.list[MemoryRecord],
        have: set[str],
        have_backend_ids: set[str],
    ) -> tuple[builtins.list[MemoryRecord], MemoryRecord | None]:
        """Materialize one backend hit and durably reconcile id collisions."""

        body = cand.get("body") or ""
        backend_id = str(cand.get("id") or "")
        if (
            not body.strip()
            or backend_id in have
            or (backend_id and backend_id in have_backend_ids)
        ):
            return existing, None
        try:
            # `source=memo-cache-fill` makes save() skip the write policy
            # (this entry already mirrors the backing store — see
            # `_apply_write_policy`), so the fill stays clean.
            saved = self.save(
                content=body,
                title=cand.get("title") or "",
                type_=cand.get("type") or "note",
                tags=cand.get("tags") or [],
                extra={
                    "source": "memo-cache-fill",
                    **({"cache_backend_id": backend_id} if backend_id else {}),
                },
                auto_derive=False,
            )
            if saved.id in have:
                if backend_id:
                    merged_extra = dict(saved.extra or {})
                    merged_extra["cache_backend_id"] = backend_id
                    persisted = self.update(saved.id, extra=merged_extra)
                    if persisted is not None:
                        existing = [
                            replace(persisted, score=record.score)
                            if record.id == saved.id
                            else record
                            for record in existing
                        ]
                    have_backend_ids.add(backend_id)
                return existing, None
        except Exception as exc:
            _log.debug("cache read-through materialize failed: %s", exc)
            return existing, None

        have.add(saved.id)
        if backend_id:
            have_backend_ids.add(backend_id)
        return existing, replace(saved, score=cand.get("score"))

    # -- list ---------------------------------------------------------------
