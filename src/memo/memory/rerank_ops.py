"""Reranking + source-feedback operations for `Memory`.

`_RerankOpsMixin` holds the cross-encoder rerank surface and the per-source
👍/👎 feedback methods (record / list / clear) plus their helpers, moved
verbatim from the former `memory.py` god-file.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, ClassVar

from memo.flags import flag_bool, flag_float
from memo.memory._base import _MemoryBase
from memo.memory.record import (
    AmbiguousIdError,
    MemoryRecord,
    _log,
)
from memo.tiers import VerificationState


def _feedback_recency_weight(
    created_at: str, *, halflife_days: float, now: datetime | None = None
) -> float:
    """Half-life weight for a feedback vote based on its age.

    `0.5 ** (age_days / halflife_days)` — a vote at one half-life counts half.
    Returns 1.0 (no decay) when `halflife_days <= 0` or `created_at` can't be
    parsed, so a malformed timestamp never silently zeroes a vote.
    """
    if halflife_days <= 0:
        return 1.0
    try:
        ts = datetime.fromisoformat(created_at)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return 1.0
    current = now or datetime.now(tz=UTC)
    age_days = max(0.0, (current - ts).total_seconds() / 86400.0)
    return 0.5 ** (age_days / halflife_days)


def _state_decay_factor(memory_record: MemoryRecord) -> float:
    """Compute decay factor based on verification state + age.

    Verified memories score higher; STALE and UNVERIFIED memories are penalized.
    Returns a float multiplier (0.7–1.0) applied to hit scores.

    - VERIFIED & fresh (< 7 days): 1.0 (no penalty)
    - VERIFIED & old (7+ days): 0.95 (5% penalty)
    - STALE: 0.7 (30% penalty)
    - UNVERIFIED: 0.8 (20% penalty)
    """
    if not memory_record.verified_at:
        return 0.8  # UNVERIFIED: 20% penalty

    now = int(time.time())
    days_since_verified = (now - memory_record.verified_at) / 86400.0

    if memory_record.verification_state == VerificationState.VERIFIED:
        return 1.0 if days_since_verified < 7 else 0.95
    elif memory_record.verification_state == VerificationState.STALE:
        return 0.7  # STALE: 30% penalty
    else:  # UNVERIFIED
        return 0.8


class _RerankOpsMixin(_MemoryBase):
    # -- source feedback (public) ------------------------------------------

    def feedback_record(
        self,
        source_id: str,
        *,
        query_text: str,
        rating: str,
        only_if_absent: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Public wrapper around store.record_source_feedback.

        Accepts `rating` as "up"/"down"/"click"/"ignore" or "+1"/"-1".
        Resolves a short `source_id` prefix to a full meta.id when possible
        (errors on ambiguity). Embeds `query_text` with the asymmetric
        retrieval prefix so future queries compare on equal footing.

        Signal semantics:
          "click"      — implicit positive (user used this result); soft boost.
          "thumbs_up"  — explicit positive; stronger boost.
          "ignore"     — implicit negative (user skipped); soft score penalty.
          "thumbs_down"— explicit rejection; hard exclude from future results.

        `only_if_absent=True` (used by the outcome loop) skips the write if any
        feedback already exists for this (source, query) — never overrides a
        manual vote.
        """
        rating_norm, signal = self._normalize_rating(rating)
        resolved = self._resolve_source_id(source_id)
        if not query_text or not query_text.strip():
            raise ValueError("query_text is required")
        emb = self.embedder.embed_query(query_text)
        fid = self.store.record_source_feedback(
            source_id=resolved,
            query_text=query_text,
            query_emb=list(emb),
            rating=rating_norm,
            extra={**(extra or {}), "signal": signal},
            only_if_absent=only_if_absent,
        )
        return {
            "feedback_id": fid,
            "source_id": resolved,
            "query_text": query_text,
            "rating": "up" if rating_norm > 0 else "down",
            "signal": signal,
        }

    def feedback_list(
        self,
        *,
        source_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if source_id:
            source_id = self._resolve_source_id(source_id)
        return self.store.list_source_feedback(source_id=source_id, limit=limit)

    def feedback_clear(self, source_id: str) -> int:
        resolved = self._resolve_source_id(source_id)
        return self.store.clear_source_feedback(resolved)

    @staticmethod
    def _normalize_rating(rating: str | int) -> tuple[int, str]:
        """Return (db_rating, canonical_signal).

        Canonical signals and their semantics:
          thumbs_up  (rating=1)  — explicit positive vote; 0.15 score boost.
          click      (rating=1)  — implicit positive (user viewed/used); 0.08 boost.
          thumbs_down (rating=-1) — explicit rejection; hard-exclude from results.
          ignore     (rating=-1) — implicit negative (user skipped); soft 0.7× penalty.
        """
        raw = str(rating).strip().lower()
        if raw in {"up", "+1", "1", "thumbs_up", "positive", "pos"}:
            return 1, "thumbs_up"
        if raw in {"down", "-1", "thumbs_down", "negative", "neg"}:
            return -1, "thumbs_down"
        if raw == "click":
            return 1, "click"
        if raw == "ignore":
            return -1, "ignore"
        raise ValueError(f"unknown rating {rating!r}; expected up/down/click/ignore")

    def _resolve_source_id(self, source_id: str) -> str:
        sid = (source_id or "").strip()
        if not sid:
            raise ValueError("source_id is required")
        # Already a full id (32 hex chars) — accept only if it exists, so a
        # fabricated id can't write an orphan feedback row.
        if len(sid) >= 32:
            if self.store.get(sid) is None:
                raise ValueError(f"no memory matches source_id {sid!r}")
            return sid
        # Prefix lookup. Must match exactly one row.
        matches = self.store.find_by_prefix(sid, limit=2)
        if not matches:
            raise ValueError(f"no memory matches source_id prefix {sid!r}")
        if len(matches) > 1:
            raise AmbiguousIdError(sid, matches)
        return matches[0]

    # Signal strengths: how much each feedback type affects the score.
    # thumbs_up → full boost; click → half boost (implicit positive signal).
    # thumbs_down → hard exclude; ignore → soft penalty (score × 0.7).
    _SIGNAL_BOOST: ClassVar[dict[str, float]] = {"thumbs_up": 0.15, "click": 0.08}
    _SIGNAL_IGNORE_FACTOR = 0.7  # multiplied into score for "ignore" signals

    def _apply_source_feedback(
        self,
        hits: list[MemoryRecord],
        query_emb: list[float],
        *,
        sim_threshold: float = 0.85,
        boost_per_vote: float = 0.15,
        boost_cap: float = 0.6,
    ) -> list[MemoryRecord]:
        """Filter/boost hits using prior votes for the user query.

        Signal semantics (stored in extra_json.signal):
          thumbs_up  → score += 0.15 per vote, capped at boost_cap.
          click      → score += 0.08 per vote (implicit positive, softer).
          thumbs_down→ hard exclude (user explicitly rejected this source).
          ignore     → score *= 0.7 (user skipped — soft penalty, no hard exclude).

        Legacy rows without a signal field are treated as thumbs_up / thumbs_down
        based on their integer rating so backward compatibility is preserved.

        Tunables (env, optional):
          MEMO_FEEDBACK_SIM_THRESHOLD (default 0.85)
          MEMO_FEEDBACK_BOOST_PER_VOTE (default 0.15)
          MEMO_FEEDBACK_BOOST_CAP (default 0.6)
        """
        import json

        from memo.flags import flag_float

        # Knobs live in the flags registry (typed + validated by `memo config
        # validate`); the kwargs above are the in-code defaults the registry
        # mirrors. `flag_float` returns env-or-registry-default — never raw
        # os.environ (CLAUDE.md rule). The `or <kwarg>` guard keeps a caller's
        # explicit override winning over a 0/None.
        sim_threshold = flag_float("MEMO_FEEDBACK_SIM_THRESHOLD") or sim_threshold
        boost_per_vote = flag_float("MEMO_FEEDBACK_BOOST_PER_VOTE") or boost_per_vote
        boost_cap = flag_float("MEMO_FEEDBACK_BOOST_CAP") or boost_cap
        # Temporal decay: a positive vote's boost fades with its age (half-life
        # MEMO_FEEDBACK_HALFLIFE_DAYS, default 180; 0 disables). Keeps recent
        # feedback authoritative without letting a year-old 👍 pin a stale
        # source. thumbs_down (exclusion) and ignore are NOT decayed — an
        # explicit rejection shouldn't quietly expire.
        halflife_days = flag_float("MEMO_FEEDBACK_HALFLIFE_DAYS")
        if halflife_days is None:
            halflife_days = 180.0
        _now = datetime.now(tz=UTC)
        # Existence pre-filter: most memories have zero feedback rows, so a
        # single IN-list lookup tells us which hits are even worth the kNN vec
        # scan below. Collapses a per-hit N+1 into 1 query for the common case.
        try:
            with_feedback = self.store.sources_with_feedback([h.id for h in hits])
        except sqlite3.Error:
            with_feedback = {h.id for h in hits}  # degrade to per-hit scan
        out: list[MemoryRecord] = []
        for h in hits:
            if h.id not in with_feedback:
                out.append(h)
                continue
            try:
                fb = self.store.find_feedback_for_source(
                    h.id,
                    query_emb,
                    threshold=sim_threshold,
                )
            except sqlite3.Error:
                fb = []
            if not fb:
                out.append(h)
                continue
            score = h.score or 0.0
            hard_exclude = False
            total_boost = 0.0
            ignore_factor = 1.0
            for r in fb:
                extra_raw = r.get("extra_json") or ""
                try:
                    extra = json.loads(extra_raw) if extra_raw else {}
                except (json.JSONDecodeError, TypeError):
                    extra = {}
                signal = str(extra.get("signal") or "")
                # Determine canonical signal from stored value or fall back to rating.
                if not signal:
                    signal = "thumbs_up" if r["rating"] > 0 else "thumbs_down"
                if signal == "thumbs_down":
                    hard_exclude = True
                    break
                if signal == "ignore":
                    ignore_factor = min(ignore_factor, self._SIGNAL_IGNORE_FACTOR)
                else:
                    per = self._SIGNAL_BOOST.get(signal, boost_per_vote)
                    weight = _feedback_recency_weight(
                        str(r.get("created_at") or ""),
                        halflife_days=halflife_days,
                        now=_now,
                    )
                    total_boost += per * weight
            if hard_exclude:
                continue
            score = score * ignore_factor
            if total_boost > 0:
                score = score + min(boost_cap, total_boost)
            h = replace(h, score=round(score, 6))
            out.append(h)
        return out

    def rerank_hits(
        self,
        query: str,
        hits: list[dict[str, Any]],
        *,
        top_n: int | None = None,
        body_chars: int = 1200,
    ) -> list[dict[str, Any]]:
        """Score externally-supplied hit dicts with the cross-encoder.

        Mirrors the ``memo rerank`` CLI but reuses THIS instance's cached
        reranker (`self._reranker`), so a long-lived server (memo-mcp HTTP
        daemon) pays the Qwen3-Reranker load only once. This is the warm
        equivalent of the per-process CLI used by Synapse's `memo_ce` rerank.

        Each hit is scored on ``"{title}\\n\\n{snippet|body}"`` (truncated to
        ``body_chars``); returns the list reordered with a ``rerank_score``
        field added per hit, original fields preserved. Pass-through (input
        order, no scores) when reranking is disabled in this install.
        """
        if not query or not hits:
            return list(hits or [])
        if not self.cfg.reranker_enabled:
            return list(hits)
        reranker = self._ensure_reranker()
        scored: list[tuple[float, dict[str, Any]]] = []
        for h in hits:
            if not isinstance(h, dict):
                continue
            title = str(h.get("title") or "")
            body_src = str(h.get("snippet") or h.get("body") or "")[: max(0, body_chars)]
            doc = f"{title}\n\n{body_src}" if body_src else title
            try:
                p = float(reranker.score(query, doc))
            except Exception as exc:
                _log.error(
                    "reranker score failed (model=%s, revision=%s, hit_id=%s): %s",
                    self.cfg.reranker_model,
                    self.cfg.reranker_revision,
                    h.get("id"),
                    exc,
                )
                p = 0.0
            new = dict(h)
            new["rerank_score"] = p
            scored.append((p, new))
        scored.sort(key=lambda t: t[0], reverse=True)
        out = [h for _p, h in scored]
        if top_n is not None and top_n > 0:
            out = out[:top_n]
        return out

    def _rerank(
        self,
        query: str,
        hits: list[MemoryRecord],
        *,
        top_n: int,
    ) -> list[MemoryRecord]:
        """Apply the cross-encoder to `hits`, return top-N reordered.

        Score fusion: the final ranking blends the reranker's `P(yes)`
        with the original RRF position so a single noisy cross-encoder
        score can't promote a candidate the bi-encoder + BM25 fusion
        had ranked far down. Position bonus is `1 - i / N` where `i`
        is the original 0-indexed RRF rank — top-of-RRF gets +1.0,
        bottom gets ~0.

        Lazy-loads the reranker on first call. Failures are absorbed:
        if MLX runs into a Metal hiccup mid-rerank we fall back to the
        original RRF order so search never goes dark on the user.
        """
        reranker = self._ensure_reranker()

        # Snapshot original RRF positions BEFORE rerank rewrites the
        # `score` field. Index by id rather than object identity so the
        # reranker can return new replaced records without losing the
        # mapping.
        n = len(hits)
        rrf_pos: dict[str, int] = {h.id: i for i, h in enumerate(hits)}

        try:
            reranked = reranker.rerank(query, hits, top_n=None)
        except Exception as exc:
            _log.error(
                "reranker failed (model=%s, revision=%s): %s",
                self.cfg.reranker_model,
                self.cfg.reranker_revision,
                exc,
            )
            _log.info("falling back to RRF order (no cross-encoder reranking)")
            return hits[:top_n]

        alpha = self.cfg.rerank_fusion_alpha
        fused: list[MemoryRecord] = []
        for h in reranked:
            rerank_score = h.score or 0.0
            pos = rrf_pos.get(h.id, n - 1)
            rrf_bonus = 1.0 - (pos / max(n - 1, 1))
            final = alpha * rerank_score + (1.0 - alpha) * rrf_bonus
            fused.append(replace(h, score=final))
        fused.sort(key=lambda h: h.score or 0.0, reverse=True)
        return fused[:top_n]

    def _rerank_logic(
        self,
        hits: list[dict[str, Any]],
        query: str,
        rerank_candidates: int,
    ) -> list[dict[str, Any]]:
        """Apply distance decay + verification state weighting to hits.

        Takes pre-scored hits and applies inverse-distance decay and/or
        verification state decay if enabled, returning the reordered list.
        Distance decay penalizes memories far from base facts in the knowledge
        graph via BFS distance. State decay prioritizes VERIFIED memories.

        Args:
            hits: List of hit dicts with "id" and "score" fields
            query: Query text (for context, unused in this version)
            rerank_candidates: Maximum hits to return

        Returns:
            Reordered hits with decayed scores and debug fields
            (when decay is enabled)
        """
        scored_hits = list(hits or [])  # Copy to avoid mutation

        # Apply distance decay if enabled
        if flag_bool("MEMO_GRAPH_DISTANCE_DECAY"):
            decay_rate = flag_float("MEMO_GRAPH_DISTANCE_DECAY_RATE")
            for hit in scored_hits:
                mem_id = hit.get("id")
                if mem_id:
                    distance = self.graph.distance_to_nearest_fact(mem_id)
                    # Decay: score *= 1 / (1 + rate * distance)
                    decay_factor = 1.0 / (1.0 + decay_rate * distance)
                    current_score = hit.get("score", 0.0)
                    hit["score"] = current_score * decay_factor
                    hit["_distance"] = distance  # Debug: track distance

        # Apply verification state decay if enabled
        if flag_bool("MEMO_VERIFICATION_STATE_TRACKING"):
            for hit in scored_hits:
                mem_id = hit.get("id")
                if mem_id and mem_id in self.memory_map:
                    mem = self.memory_map[mem_id]
                    state_decay = _state_decay_factor(mem)
                    current_score = hit.get("score", 0.0)
                    hit["score"] = current_score * state_decay
                    hit["_verification_state"] = mem.verification_state.value  # Debug

        # Sort by score descending
        scored_hits.sort(key=lambda h: h.get("score", 0.0), reverse=True)

        # Return top-N if rerank_candidates is specified
        if rerank_candidates > 0:
            return scored_hits[:rerank_candidates]
        return scored_hits
