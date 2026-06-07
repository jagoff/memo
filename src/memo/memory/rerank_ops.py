"""Reranking + source-feedback operations for `Memory`.

`_RerankOpsMixin` holds the cross-encoder rerank surface and the per-source
👍/👎 feedback methods (record / list / clear) plus their helpers, moved
verbatim from the former `memory.py` god-file.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import replace
from typing import Any, ClassVar

from memo.memory._base import _MemoryBase
from memo.memory.record import (
    AmbiguousIdError,
    MemoryRecord,
    _log,
)


class _RerankOpsMixin(_MemoryBase):
    # -- source feedback (public) ------------------------------------------

    def feedback_record(
        self, source_id: str, *, query_text: str, rating: str,
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
            extra={"signal": signal},
        )
        return {
            "feedback_id": fid,
            "source_id": resolved,
            "query_text": query_text,
            "rating": "up" if rating_norm > 0 else "down",
            "signal": signal,
        }

    def feedback_list(
        self, *, source_id: str | None = None, limit: int = 50,
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
        # Already a full id (32 hex chars) — accept as-is.
        if len(sid) >= 32:
            return sid
        # Prefix lookup. Must match exactly one row.
        matches = self.store.find_by_prefix(sid, limit=2)
        if not matches:
            raise ValueError(f"no memoria matches source_id prefix {sid!r}")
        if len(matches) > 1:
            raise AmbiguousIdError(sid, matches)
        return matches[0]

    # Signal strengths: how much each feedback type affects the score.
    # thumbs_up → full boost; click → half boost (implicit positive signal).
    # thumbs_down → hard exclude; ignore → soft penalty (score × 0.7).
    _SIGNAL_BOOST: ClassVar[dict[str, float]] = {"thumbs_up": 0.15, "click": 0.08}
    _SIGNAL_IGNORE_FACTOR = 0.7  # multiplied into score for "ignore" signals

    def _apply_source_feedback(
        self, hits: list[MemoryRecord], query_emb: list[float],
        *, sim_threshold: float = 0.85, boost_per_vote: float = 0.15,
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
        sim_threshold = float(os.environ.get("MEMO_FEEDBACK_SIM_THRESHOLD") or sim_threshold)
        boost_per_vote = float(os.environ.get("MEMO_FEEDBACK_BOOST_PER_VOTE") or boost_per_vote)
        boost_cap = float(os.environ.get("MEMO_FEEDBACK_BOOST_CAP") or boost_cap)
        out: list[MemoryRecord] = []
        for h in hits:
            try:
                fb = self.store.find_feedback_for_source(
                    h.id, query_emb, threshold=sim_threshold,
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
                    total_boost += per
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
            except Exception:
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
        self, query: str, hits: list[MemoryRecord], *, top_n: int,
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

