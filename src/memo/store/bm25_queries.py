from __future__ import annotations

import logging
import os
import sqlite3
from datetime import UTC, datetime
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
# Mirrors `queries.py`'s `k_fetch` widening for the same reason: a post-filter
# that can drop rows needs a bigger candidate pool than `limit`.
_VALIDITY_CANDIDATE_MULT = 4

_META_SELECT_COLUMNS = (
    "id, path, title, type, tags, created, updated, body_hash, extra_json, "
    "review_after, verification_state, verified_at, valid_at, invalid_at"
)


def _now_iso_local() -> str:
    """Wall-clock *now* in the machine's LOCAL UTC offset, ms precision — the
    EXACT shape ``memo.memory.record._now_iso()`` stamps ``created`` /
    ``valid_at`` / ``invalid_at`` with. Replicated here rather than imported
    because the store is a foundation layer that must not import up into
    ``memo.memory`` (that would cycle: ``memory.facade`` → ``store`` →
    ``memory.record``); the sibling store modules (``episode_store`` etc.) keep
    their own ``_now_iso`` for the same reason. Matching the stored columns'
    offset+precision is what makes ``_validity_filter``'s lexicographic TEXT
    compare correct — see its tz note.
    """
    return datetime.now(tz=UTC).astimezone().isoformat(timespec="milliseconds")


def _normalize_as_of(as_of: str) -> str:
    """Normalize a caller-supplied ``as_of`` into the local-offset ISO shape the
    validity columns are stamped in, so the lexicographic TEXT compare in
    ``_validity_filter`` is correct on THIS machine.

    - **Bare date** (``YYYY-MM-DD``, no time) → the END of that day
      (``T23:59:59.999999``) so a fact that became valid *later that same day*
      (``valid_at="…-06-15T14:00…"``) is still included — "as of end of day D".
    - **Naive datetime** (no offset) → assumed local wall-clock time.
    - **Offset-aware** value → converted instant-preservingly to the local
      offset (``…T14:00:00+00:00`` on a UTC-3 box → ``…T11:00:00-03:00``).

    Unparseable input falls back to the raw string (old behaviour — no crash on
    a malformed boundary), which then simply matches lexicographically as-is.
    """
    s = as_of.strip()
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return as_of
    if "T" not in s and " " not in s:  # bare date → end of that day
        dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
    # naive → assume local; aware → convert to the machine's local offset.
    return dt.astimezone().isoformat()


def _validity_filter(
    prefix: str, include_invalid: bool, as_of: str | None = None
) -> tuple[str, list[Any]]:
    """Temporal gate (record-level bi-temporal validity).

    Three modes, in precedence order:

    - **``as_of`` set → valid-time.** The interval must CONTAIN ``as_of``:
      ``COALESCE(valid_at, created) <= ? AND (invalid_at IS NULL OR invalid_at
      > ?)`` (``as_of`` bound twice). This REPLACES the default now-gate, so a
      fact superseded since ``as_of`` still resurfaces. ``COALESCE(valid_at,
      created)`` resolves legacy/un-backfilled rows (NULL ``valid_at``) by
      learned-time.
    - **``include_invalid`` set, no ``as_of`` → no filter** (``("", [])``);
      every row, closed intervals included.
    - **default (neither) → now-gate.** Excludes rows whose interval is already
      closed as of *now* — ``(invalid_at IS NULL OR invalid_at > ?)`` — so a
      contradiction-superseded fact stays in-index (recoverable, as-of
      queryable) but drops out of normal recall.

    **Timezone (single-machine consistent).** The columns are TEXT and compared
    lexicographically, so the bound value must carry the SAME UTC offset the
    columns were stamped in. Stored ``valid_at``/``invalid_at``/``created`` use
    ``record._now_iso()`` = the machine's LOCAL offset (e.g. ``-03:00``), so the
    bound is normalized to that same local offset here — ``now`` via
    ``_now_iso_local()``, ``as_of`` via ``_normalize_as_of()`` — NOT a fixed
    ``+00:00``. Binding ``+00:00`` skewed the boundary by the machine's UTC
    offset (superseded facts resurfacing / future interval-ends hiding early).
    KNOWN LIMITATION: cross-machine git sync can still land values stamped in a
    different offset into one DB; that pre-existing mixed-offset case (shared
    with ``created``/``updated`` recency sorting) is out of scope.

    Index-friendly via the ``idx_meta_invalid_at`` partial index. ``prefix``
    qualifies the columns for JOINed queries (``"meta."``) vs single-table
    selects (``""``). Qmark binding matches the store's positional param style.
    """
    if as_of is not None:
        bound = _normalize_as_of(as_of)
        return (
            f" AND COALESCE({prefix}valid_at, {prefix}created) <= ?"
            f" AND ({prefix}invalid_at IS NULL OR {prefix}invalid_at > ?)",
            [bound, bound],
        )
    if include_invalid:
        return "", []
    return f" AND ({prefix}invalid_at IS NULL OR {prefix}invalid_at > ?)", [_now_iso_local()]


def _env_float(
    name: str, default: float, min_val: float | None = None, max_val: float | None = None
) -> float:
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
        include_invalid: bool = False,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 keyword search. Dispatches to tantivy when available, FTS5 otherwise.

        Returns rows shaped like `search()` (vec) — metadata dict with `score`
        in [0,1] where higher = more relevant.

        `field_boost="exact"` forces the FTS5 path (so the elevated tag/title
        weights apply deterministically regardless of backend) and runs a
        strict AND with no OR fallback.

        `include_invalid=False` (default) hides rows whose validity interval is
        closed as of now; `include_invalid=True` bypasses the gate. `as_of=T`
        overrides both with a valid-time predicate (rows valid at T).
        """
        if not query or not query.strip():
            return []
        if field_boost == "exact":
            return self._search_bm25_fts5(
                query,
                limit,
                type_,
                exclude_types,
                field_boost="exact",
                include_invalid=include_invalid,
                as_of=as_of,
            )
        t = self._get_tantivy()
        if t is not None:
            return self._search_bm25_tantivy(
                query,
                limit,
                type_,
                exclude_types,
                t,
                include_invalid=include_invalid,
                as_of=as_of,
            )
        return self._search_bm25_fts5(
            query,
            limit,
            type_,
            exclude_types,
            include_invalid=include_invalid,
            as_of=as_of,
        )

    def _tantivy_candidate_k(
        self,
        limit: int,
        *,
        type_: str | None,
        exclude_types: set[str] | None,
        include_invalid: bool,
        as_of: str | None,
    ) -> int:
        """Candidate pool for the tantivy legs, widened per post-filter.

        Type filters were already handled; validity was not. `update_validity` /
        supersede never touch the tantivy index, so superseded rows stay in it,
        take up the `limit` slots, and are then dropped by `_validity_filter` on
        the meta join — returning a thinner BM25 leg than the corpus actually
        holds, with no signal to the caller. (FTS5 has no such problem: its
        validity filter is inside the SQL, before LIMIT.)
        """
        if type_ or exclude_types:
            return limit * _TYPE_FILTER_CANDIDATE_MULT
        # Same predicate (and same `idx_meta_invalid_at` partial index) as
        # `_QueriesMixin._index_has_invalid`, inlined here for the same reason
        # `_deleted_filter_sql` is duplicated: this mixin only sees
        # `_StoreBase`.
        has_invalid = bool(
            self._conn.execute(
                "SELECT EXISTS(SELECT 1 FROM meta WHERE invalid_at IS NOT NULL)"
            ).fetchone()[0]
        )
        validity_can_drop = (not include_invalid) and (as_of is not None or has_invalid)
        return limit * _VALIDITY_CANDIDATE_MULT if validity_can_drop else limit

    def _search_bm25_tantivy(
        self,
        query: str,
        limit: int,
        type_: str | None,
        exclude_types: set[str] | None,
        t: Any,
        include_invalid: bool = False,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        # Fetch more candidates when a post-filter can drop rows, so we can
        # still honour `limit` after the meta-table join.
        candidate_k = self._tantivy_candidate_k(
            limit,
            type_=type_,
            exclude_types=exclude_types,
            include_invalid=include_invalid,
            as_of=as_of,
        )
        hits = t.search_bm25(query, candidate_k)
        if not hits:
            return []
        # Resolve metadata from sqlite in one batched query.
        id_score = {h["id"]: h["score"] for h in hits}
        placeholders = ",".join("?" for _ in id_score)
        valid_sql, valid_params = _validity_filter("", include_invalid, as_of)
        sql = (
            f"SELECT {_META_SELECT_COLUMNS} "  # noqa: S608
            f"FROM meta WHERE id IN ({placeholders}){self._deleted_filter_sql()}{valid_sql}"
        )
        params: list[Any] = [*id_score.keys(), *valid_params]
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
        include_invalid: bool = False,
        as_of: str | None = None,
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
                "       meta.created, meta.updated, meta.body_hash, meta.extra_json, "
                "       meta.verification_state, meta.verified_at, "
                "       meta.valid_at, meta.invalid_at "
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
            valid_sql, valid_params = _validity_filter("meta.", include_invalid, as_of)
            sql += valid_sql + " "
            params.extend(valid_params)
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
        include_invalid: bool = False,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fuzzy (typo-tolerant) BM25 search via tantivy.

        Falls back to `_search_bm25_fts5` when tantivy is not available
        (no true fuzzy without tantivy, but better than nothing).
        Returns rows shaped like `search()`.

        `include_invalid`/`as_of` mirror `search_bm25`: default hides closed
        intervals; `as_of=T` applies a valid-time (valid-at-T) predicate.
        """
        if not query or not query.strip():
            return []
        t = self._get_tantivy()
        if t is None:
            return self._search_bm25_fts5(
                query,
                limit,
                type_,
                exclude_types,
                include_invalid=include_invalid,
                as_of=as_of,
            )
        candidate_k = self._tantivy_candidate_k(
            limit,
            type_=type_,
            exclude_types=exclude_types,
            include_invalid=include_invalid,
            as_of=as_of,
        )
        hits = t.search_fuzzy(query, candidate_k)
        if not hits:
            return []
        id_score = {h["id"]: h["score"] for h in hits}
        placeholders = ",".join("?" for _ in id_score)
        valid_sql, valid_params = _validity_filter("", include_invalid, as_of)
        sql = (
            f"SELECT {_META_SELECT_COLUMNS} "  # noqa: S608
            f"FROM meta WHERE id IN ({placeholders}){self._deleted_filter_sql()}{valid_sql}"
        )
        params: list[Any] = [*id_score.keys(), *valid_params]
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

    def _deleted_filter_sql(self) -> str:
        """`" AND (deleted_at IS NULL OR deleted_at = '')"` when meta has the
        soft-delete column, else "" (pre-migration DBs where the deleted_at
        ALTER was skipped, e.g. suppressed lock error). Same guard as
        `count()` below; `_QueriesMixin` carries an identical definition for
        its own read surfaces."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(meta)").fetchall()}
        return " AND (deleted_at IS NULL OR deleted_at = '')" if "deleted_at" in cols else ""

    def count(self) -> int:
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(meta)").fetchall()}
        if "deleted_at" in cols:
            return self._conn.execute(
                "SELECT COUNT(*) FROM meta WHERE deleted_at IS NULL"
            ).fetchone()[0]
        return self._conn.execute("SELECT COUNT(*) FROM meta").fetchone()[0]

    def count_by_type(self) -> dict[str, int]:
        """Return active record counts grouped by their public type."""

        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(meta)").fetchall()}
        where = " WHERE deleted_at IS NULL" if "deleted_at" in cols else ""
        rows = self._conn.execute(
            f"SELECT type, COUNT(*) AS n FROM meta{where} GROUP BY type ORDER BY type"  # noqa: S608
        ).fetchall()
        return {str(row["type"]): int(row["n"]) for row in rows}
