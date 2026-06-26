from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any

from ._base import _StoreBase
from .rows import _row_to_dict
from .schema import (
    _BM25_ES_STOPWORDS,
    _BM25_FTS_BODY_WEIGHT,
    _BM25_FTS_TAGS_WEIGHT,
    _BM25_FTS_TITLE_WEIGHT,
    _BM25_UNINDEXED_WEIGHT,
)

_log = logging.getLogger(__name__)

# When a type filter is active, BM25/fuzzy candidates are over-fetched so the
# post-filter against the meta table can still honour `limit`. A heavy filter
# (e.g. one type out of a dense, diverse top-K) can drop most candidates, so the
# multiplier is generous to avoid silently under-filling below `limit`.
_TYPE_FILTER_CANDIDATE_MULT = 20


def _env_float(name: str, default: float, min_val: float | None = None, max_val: float | None = None) -> float:
    """Parse a float env var, falling back to `default` when unset/blank/bad.

    The store layer is a foundation module and cannot import memo.flags, so
    these tuning knobs (registered there for `memo config validate`) are read
    directly from the environment here. `min_val`/`max_val` duplicate the flag
    spec bounds so `memo config validate` and runtime agree.
    """
    raw = os.environ.get(name)
    if not raw or not raw.strip():
        return default
    try:
        v = float(raw)
        if min_val is not None and v < min_val:
            return default
        if max_val is not None and v > max_val:
            return default
        return v
    except ValueError:
        return default


class _BM25QueriesMixin(_StoreBase):
    """BM25/FTS5 and fuzzy search methods.

    Extracted from `_QueriesMixin` to keep each file under 800 lines.
    `VecStore` inherits these via `_QueriesMixin(_BM25QueriesMixin, ...)`.
    """

    def search_bm25(
        self,
        query: str,
        limit: int = 10,
        type_: str | None = None,
        exclude_types: set[str] | None = None,
        field_boost: str | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 keyword search. Dispatches to tantivy when available, FTS5 otherwise.

        Returns rows shaped like `search()` (vec) — metadata dict with `score`
        in [0,1] where higher = more relevant.

        `field_boost="exact"` forces the FTS5 path (so the elevated tag/title
        weights apply deterministically regardless of backend) and runs a
        strict AND with no OR fallback.
        """
        if not query or not query.strip():
            return []
        if field_boost == "exact":
            return self._search_bm25_fts5(
                query, limit, type_, exclude_types, field_boost="exact"
            )
        t = self._get_tantivy()
        if t is not None:
            return self._search_bm25_tantivy(query, limit, type_, exclude_types, t)
        return self._search_bm25_fts5(query, limit, type_, exclude_types)

    def _search_bm25_tantivy(
        self,
        query: str,
        limit: int,
        type_: str | None,
        exclude_types: set[str] | None,
        t: Any,
    ) -> list[dict[str, Any]]:
        # Fetch more candidates when filtering by type so we can honour `limit`
        # after post-filtering against the meta table.
        candidate_k = limit * _TYPE_FILTER_CANDIDATE_MULT if (type_ or exclude_types) else limit
        hits = t.search_bm25(query, candidate_k)
        if not hits:
            return []
        # Resolve metadata from sqlite in one batched query.
        id_score = {h["id"]: h["score"] for h in hits}
        placeholders = ",".join("?" for _ in id_score)
        sql = (
            "SELECT id, path, title, type, tags, created, updated, body_hash, extra_json "  # noqa: S608
            f"FROM meta WHERE id IN ({placeholders})"
        )
        params: list[Any] = list(id_score.keys())
        if type_:
            sql += " AND type = ?"
            params.append(type_)
        if exclude_types:
            sql += f" AND type NOT IN ({','.join('?' for _ in exclude_types)})"
            params.extend(sorted(exclude_types))
        rows = self._conn.execute(sql, params).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = _row_to_dict(r)
            d["score"] = id_score.get(d["id"], 0.0)
            out.append(d)
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:limit]

    def _search_bm25_fts5(
        self,
        query: str,
        limit: int,
        type_: str | None,
        exclude_types: set[str] | None,
        field_boost: str | None = None,
    ) -> list[dict[str, Any]]:
        # FTS5 needs an explicit MATCH expression. Pre-2026-05-07 we wrapped
        # the whole query in `"..."` (phrase match) to dodge FTS5 syntax
        # collisions on hyphens/colons. Side-effect: multi-word queries
        # required the EXACT consecutive sequence, killing recall on
        # natural Spanish queries — "Astor terapia ocupacional" would NOT
        # match a doc titled "Informe Terapia Ocupacional — Astor Ferrari"
        # because the words don't appear consecutively in that order.
        #
        # Fix: tokenize via \w+ regex (drops punctuation, keeps Unicode
        # letters via Python's \w), wrap each token in its own phrase
        # quotes, join with whitespace (FTS5's implicit AND). Result:
        # `"Astor" "terapia" "ocupacional"` — matches any doc containing
        # all 3 words anywhere, in any order.
        import re as _re

        _raw_tokens = [t for t in _re.findall(r"\w+", query, flags=_re.UNICODE) if t]
        if not _raw_tokens:
            return []
        # Strip Spanish stopwords, but only if doing so keeps ≥ 2 tokens.
        # Single-token queries like "Grecia" or stopword-heavy short
        # questions ("que es eso?") must still produce a match expression.
        _filtered = [t for t in _raw_tokens if t.lower() not in _BM25_ES_STOPWORDS]
        _tokens = _filtered if len(_filtered) >= 2 else _raw_tokens

        # `exact` mode applies a preconfigured field boost favouring curated
        # metadata (title/tags) over body prose. Weights are tunable via
        # MEMO_EXACT_TITLE_WEIGHT / MEMO_EXACT_TAGS_WEIGHT (registered in
        # flags.py for `memo config validate`; read here via os.environ because
        # the store layer is a foundation module and cannot import memo.flags).
        # KEEP DEFAULTS IN SYNC WITH flags_search.py (MEMO_EXACT_TITLE_WEIGHT=10.0,
        # MEMO_EXACT_TAGS_WEIGHT=8.0).
        if field_boost == "exact":
            title_w = _env_float("MEMO_EXACT_TITLE_WEIGHT", 10.0, min_val=0.0)
            tags_w = _env_float("MEMO_EXACT_TAGS_WEIGHT", 8.0, min_val=0.0)
            body_w = _BM25_FTS_BODY_WEIGHT
        else:
            title_w = _BM25_FTS_TITLE_WEIGHT
            tags_w = _BM25_FTS_TAGS_WEIGHT
            body_w = _BM25_FTS_BODY_WEIGHT

        def _run(tokens: list[str], joiner: str) -> list[Any]:
            expr = joiner.join(f'"{t}"' for t in tokens)
            candidate_k = limit * _TYPE_FILTER_CANDIDATE_MULT if (type_ or exclude_types) else limit
            sql = (
                "SELECT fts.id AS id, "
                "       bm25(fts, ?, ?, ?, ?) AS bm25_score, "
                "       meta.path, meta.title, meta.type, meta.tags, "
                "       meta.created, meta.updated, meta.body_hash, meta.extra_json "
                "FROM fts JOIN meta ON meta.id = fts.id "
                "WHERE fts MATCH ? "
            )
            params: list[Any] = [
                _BM25_UNINDEXED_WEIGHT,
                title_w,
                tags_w,
                body_w,
                expr,
            ]
            if type_:
                sql += "AND meta.type = ? "
                params.append(type_)
            if exclude_types:
                sql += f"AND meta.type NOT IN ({','.join('?' for _ in exclude_types)}) "
                params.extend(sorted(exclude_types))
            sql += "ORDER BY bm25_score ASC LIMIT ?"
            params.append(candidate_k)
            try:
                return list(self._conn.execute(sql, params).fetchall())
            except sqlite3.OperationalError as _bm25_err:
                # Malformed FTS expression (e.g. unbalanced quotes after
                # escape). Fall back to no results — Memory.search_hybrid
                # treats this as "no BM25 signal" and uses pure vec.
                _log.warning("BM25 search failed (falling back to vec-only): %s", _bm25_err)
                return []

        rows = _run(_tokens, " ")
        # AND-of-tokens zero-recall fallback: only when the strict AND
        # match returns nothing on a multi-token query, retry with OR.
        # Triggering on `<limit` (partial recall) caused RRF rank
        # washing — OR brings in popular single-token matches that
        # demote the AND-matched correct doc once fused with the vec
        # leg. Triggering only on zero is the safe floor: it cannot
        # make a successful AND query worse.
        # exact mode is strict: never loosen AND into OR. A missing term means
        # no match, by design.
        if not rows and len(_tokens) >= 2 and field_boost != "exact":
            rows = _run(_tokens, " OR ")
        out: list[dict[str, Any]] = []
        for r in rows:
            d = _row_to_dict(r)
            # bm25() returns a NEGATIVE score for sqlite-fts5 (lower =
            # better). Transform into [0, 1] where higher = better match:
            # 1 - 1/(1 + |bm|) — more negative BM25 → score near 1.0.
            bm = float(r["bm25_score"])
            d["score"] = 1.0 - 1.0 / (1.0 + abs(bm)) if bm < 0 else 0.0
            out.append(d)
        return out[:limit]

    def search_fuzzy(
        self,
        query: str,
        limit: int = 10,
        type_: str | None = None,
        exclude_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fuzzy (typo-tolerant) BM25 search via tantivy.

        Falls back to `_search_bm25_fts5` when tantivy is not available
        (no true fuzzy without tantivy, but better than nothing).
        Returns rows shaped like `search()`.
        """
        if not query or not query.strip():
            return []
        t = self._get_tantivy()
        if t is None:
            return self._search_bm25_fts5(query, limit, type_, exclude_types)
        candidate_k = limit * _TYPE_FILTER_CANDIDATE_MULT if (type_ or exclude_types) else limit
        hits = t.search_fuzzy(query, candidate_k)
        if not hits:
            return []
        id_score = {h["id"]: h["score"] for h in hits}
        placeholders = ",".join("?" for _ in id_score)
        sql = (
            "SELECT id, path, title, type, tags, created, updated, body_hash, extra_json "  # noqa: S608
            f"FROM meta WHERE id IN ({placeholders})"
        )
        params: list[Any] = list(id_score.keys())
        if type_:
            sql += " AND type = ?"
            params.append(type_)
        if exclude_types:
            sql += f" AND type NOT IN ({','.join('?' for _ in exclude_types)})"
            params.extend(sorted(exclude_types))
        rows = self._conn.execute(sql, params).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = _row_to_dict(r)
            d["score"] = id_score.get(d["id"], 0.0)
            out.append(d)
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:limit]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM meta").fetchone()[0]
