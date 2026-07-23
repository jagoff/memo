"""Reranking + source-feedback operations for `Memory`.

`_RerankOpsMixin` holds the cross-encoder rerank surface and the per-source
👍/👎 feedback methods (record / list / clear) plus their helpers, moved
verbatim from the former `memory.py` god-file.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, ClassVar

from memo.flags import flag_float
from memo.memory._base import _MemoryBase
from memo.memory.record import (
    AmbiguousIdError,
    MemoryRecord,
    _log,
)


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


def _feedback_knob(value: float | None, name: str, default: float) -> float:
    """Resolve a feedback knob without treating an explicit zero as missing."""
    if value is not None:
        return value
    configured = flag_float(name)
    return default if configured is None else configured


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

    def feedback_flag(
        self,
        source_id: str,
        *,
        kind: str,
        superseded_by: str | None = None,
    ) -> dict[str, Any]:
        """Typed *content-lifecycle* feedback on a surfaced memory.

        Distinct from feedback_record (which teaches the *retriever* per
        query): this acts on the memory itself when the agent judges the
        content no longer true. Modeled on Memoria's typed feedback
        (useful/irrelevant/outdated/wrong) — memo already covers the
        useful/irrelevant axis via up/down/click/ignore ranking votes; this
        adds the outdated/wrong axis that routes to the lifecycle:

          kind="outdated" — archive the memory (stale but not contradicted).
          kind="wrong"    — archive it; if `superseded_by` names a replacement
                            memory, record the supersede link.

        Archive is reversible (the same primitive `memo maintain`/`dream` use),
        so an over-eager flag is recoverable — never a hard delete. Accepts a
        short id prefix for `source_id` / `superseded_by`. Returns the action.
        """
        kind_norm = (kind or "").strip().lower()
        if kind_norm not in ("outdated", "wrong"):
            raise ValueError("kind must be 'outdated' or 'wrong'")
        resolved = self.resolve_id(source_id)
        if not resolved:
            raise ValueError(f"no memory matches {source_id!r}")
        replacement: str | None = None
        if superseded_by:
            replacement = self.resolve_id(superseded_by)
            if not replacement:
                raise ValueError(f"no memory matches superseded_by={superseded_by!r}")
        archived = bool(self.lifecycle.archive_memory(resolved, superseded_by=replacement))
        return {
            "source_id": resolved,
            "kind": kind_norm,
            "action": "superseded" if replacement else "archived",
            "superseded_by": replacement,
            "archived": archived,
        }

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
        sim_threshold: float | None = None,
        boost_per_vote: float | None = None,
        boost_cap: float | None = None,
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

        # Knobs live in the flags registry (typed + validated by `memo config
        # validate`); the registry defaults mirror the documented
        # 0.85 / 0.15 / 0.6. `flag_float` returns env-or-registry-default —
        # never raw os.environ (CLAUDE.md rule). Kwargs default to None so
        # resolution is: caller override > env flag > registry default.
        sim_threshold = _feedback_knob(sim_threshold, "MEMO_FEEDBACK_SIM_THRESHOLD", 0.85)
        boost_per_vote = _feedback_knob(boost_per_vote, "MEMO_FEEDBACK_BOOST_PER_VOTE", 0.15)
        boost_cap = _feedback_knob(boost_cap, "MEMO_FEEDBACK_BOOST_CAP", 0.6)
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
        equivalent of the per-process ``memo rerank`` invocation.

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
