"""Reranking + source-feedback operations for `Memory`.

`_RerankOpsMixin` holds the cross-encoder rerank surface and the per-source
👍/👎 feedback methods (record / list / clear) plus their helpers, moved
verbatim from the former `memory.py` god-file.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

from memo.memory._base import _MemoryBase
from memo.memory.record import (
    _log,
    AmbiguousIdError,
    MemoryRecord,
)


class _RerankOpsMixin(_MemoryBase):
    # -- source feedback (public) ------------------------------------------

    def feedback_record(
        self, source_id: str, *, query_text: str, rating: str,
    ) -> dict[str, Any]:
        """Public wrapper around store.record_source_feedback.

        Accepts `rating` as "up"/"down" or "+1"/"-1". Resolves a short
        `source_id` prefix to a full meta.id when possible (errors on
        ambiguity). Embeds `query_text` with the asymmetric retrieval
        prefix so future queries — which use the same prefix — can be
        compared on equal footing.
        """
        rating_norm = self._normalize_rating(rating)
        resolved = self._resolve_source_id(source_id)
        if not query_text or not query_text.strip():
            raise ValueError("query_text is required")
        emb = self.embedder.embed_query(query_text)
        fid = self.store.record_source_feedback(
            source_id=resolved,
            query_text=query_text,
            query_emb=list(emb),
            rating=rating_norm,
        )
        return {
            "feedback_id": fid,
            "source_id": resolved,
            "query_text": query_text,
            "rating": "up" if rating_norm > 0 else "down",
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
    def _normalize_rating(rating: str | int) -> int:
        raw = str(rating).strip().lower()
        if raw in {"up", "+1", "1", "thumbs_up", "positive", "pos"}:
            return 1
        if raw in {"down", "-1", "thumbs_down", "negative", "neg"}:
            return -1
        raise ValueError(f"unknown rating {rating!r}; expected up/down")

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

    def _apply_source_feedback(
        self, hits: list[MemoryRecord], query_emb: list[float],
        *, sim_threshold: float = 0.85, boost_per_vote: float = 0.15,
        boost_cap: float = 0.6,
    ) -> list[MemoryRecord]:
        """Filter/boost hits using prior 👍/👎 votes for the user query.

        For each hit, look up `source_feedback` rows on `hit.id` whose
        query embedding is cosine-similar to `query_emb` at >=
        `sim_threshold`. Then:

        - Any negative match → drop the hit (hard exclude). User said
          this source is wrong for this kind of query; trust them.
        - Positive matches → score += `boost_per_vote * n`, capped at
          `boost_cap`. Doesn't replace ranking entirely — just lifts
          well-reviewed sources up the list.
        - No relevant feedback → hit passes through unchanged.

        Tunables (env, optional):
        - `MEMO_FEEDBACK_SIM_THRESHOLD` (default 0.85)
        - `MEMO_FEEDBACK_BOOST_PER_VOTE` (default 0.15)
        - `MEMO_FEEDBACK_BOOST_CAP` (default 0.6)
        """
        sim_threshold = float(os.environ.get("MEMO_FEEDBACK_SIM_THRESHOLD") or sim_threshold)
        boost_per_vote = float(os.environ.get("MEMO_FEEDBACK_BOOST_PER_VOTE") or boost_per_vote)
        boost_cap = float(os.environ.get("MEMO_FEEDBACK_BOOST_CAP") or boost_cap)
        from dataclasses import replace
        out: list[MemoryRecord] = []
        for h in hits:
            try:
                fb = self.store.find_feedback_for_source(
                    h.id, query_emb, threshold=sim_threshold,
                )
            except Exception:
                fb = []
            if not fb:
                out.append(h)
                continue
            if any(r["rating"] < 0 for r in fb):
                # Hard exclude — user vetoed this source for similar queries.
                continue
            pos = sum(1 for r in fb if r["rating"] > 0)
            if pos > 0:
                boost = min(boost_cap, boost_per_vote * pos)
                h = replace(h, score=(h.score or 0.0) + boost)
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
        reranker = self._reranker
        if reranker is None:
            from memo.reranker import MLXReranker
            reranker = MLXReranker(
                model_path=self.cfg.reranker_model,
                revision=self.cfg.reranker_revision,
            )
            self._reranker = reranker
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
        reranker = self._reranker
        if reranker is None:
            from memo.reranker import MLXReranker
            reranker = MLXReranker(
                model_path=self.cfg.reranker_model,
                revision=self.cfg.reranker_revision,
            )
            self._reranker = reranker

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

