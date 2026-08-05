from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import UTC, datetime
from typing import Any

from ..identity import (
    canonical_topic_key,
    namespace_for_index,
)
from ..identity import (
    normalized_content_hash as content_identity_hash,
)
from ..identity import (
    normalized_title as canonical_title,
)
from ..sqlite_compat import import_sqlite_vec
from ..util import safe_operation
from .bm25_queries import _BM25QueriesMixin, _validity_filter
from .rows import _row_to_dict
from .signal_queries import _SignalQueriesMixin

serialize_float32 = import_sqlite_vec().serialize_float32

_log = logging.getLogger(__name__)

META_SELECT_COLUMNS = (
    "id, path, title, type, tags, created, updated, body_hash, extra_json, "
    "review_after, verification_state, verified_at, valid_at, invalid_at, topic_key, normalized_hash, "
    "namespace, normalized_title, normalized_content_hash"
)


def _parse_filter_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class _QueriesMixin(_BM25QueriesMixin, _SignalQueriesMixin):
    # -- public CRUD --------------------------------------------------------

    def _validate_embedding(self, id_: str, embedding: list[float]) -> None:
        """Validate one vector before any transaction mutates index rows."""
        if len(embedding) != self.dims:
            raise ValueError(
                f"Embedding dimension mismatch: got {len(embedding)}, store expects {self.dims}. "
                f"The embedder produced a vector of the wrong length — usually a swapped "
                f"model/MEMO_EMBEDDER_DIMS (e.g. 1024 vs 2560). A wrong-dims vector can still be "
                f"unit-norm, so the norm check alone won't catch this. "
                f"Run 'memo reindex --rebuild' after restoring the correct model/dims.",
            )
        norm = sum(x * x for x in embedding) ** 0.5
        if norm != norm or norm == float("inf") or norm == float("-inf"):
            raise ValueError(
                f"Embedding contains NaN or Inf (norm={norm}). "
                f"The embedder produced corrupt output — re-embed or check MLX model."
            )
        if not (0.5 < norm < 1.5):
            raise ValueError(
                f"Embedding norm {norm:.4f} out of L2-normalised range (expected ≈ 1.0) "
                f"[id={id_[:8]}] — re-embed or check MLX model output."
            )
        if not (0.95 < norm < 1.05):
            _log.warning(
                "Embedding norm %.4f outside expected [0.95, 1.05] for id=%s"
                " — possible model mismatch or quantization issue; "
                "search quality may be degraded.",
                norm,
                id_[:8],
            )

    def _derive_identity_values(
        self,
        *,
        path: str,
        title: str,
        type_: str,
        tags: list[str],
        body_text: str,
        topic_key: str | None,
        namespace: str | None,
        normalized_title: str | None,
        normalized_content_hash: str | None,
    ) -> tuple[str | None, str | None, str, str]:
        del type_  # Included in the caller's exact-identity tuple, not normalized.
        return (
            namespace if namespace is not None else namespace_for_index(tags, path=path),
            canonical_topic_key(topic_key),
            normalized_title if normalized_title is not None else canonical_title(title),
            (
                normalized_content_hash
                if normalized_content_hash is not None
                else content_identity_hash(body_text)
            ),
        )

    def _upsert_memory_row(
        self,
        cx: sqlite3.Connection,
        *,
        id_: str,
        path: str,
        title: str,
        type_: str,
        tags: list[str],
        created: str,
        updated: str,
        body_hash: str,
        embedding: list[float],
        extra: dict[str, Any] | None = None,
        body_text: str = "",
        topic_key: str | None = None,
        normalized_hash: str | None = None,
        namespace: str | None = None,
        normalized_title: str | None = None,
        normalized_content_hash: str | None = None,
        valid_at: str | None = None,
        invalid_at: str | None = None,
    ) -> None:
        """Write one complete memory row using the caller's transaction.

        `valid_at`/`invalid_at` (world-validity interval) are written on INSERT
        but intentionally left OUT of the ON CONFLICT update set — like
        `created`, they are set once and preserved across a re-save/edit;
        dedicated statements (contradiction-supersede, reindex-fold) own their
        later mutation.
        """
        namespace, topic_key, normalized_title, normalized_content_hash = (
            self._derive_identity_values(
                path=path,
                title=title,
                type_=type_,
                tags=tags,
                body_text=body_text,
                topic_key=topic_key,
                namespace=namespace,
                normalized_title=normalized_title,
                normalized_content_hash=normalized_content_hash,
            )
        )
        if self._has_identity_cols:
            cx.execute(
                "INSERT INTO meta (id, path, title, type, tags, created, updated, "
                "body_hash, extra_json, topic_key, normalized_hash, namespace, "
                "normalized_title, normalized_content_hash, valid_at, invalid_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "path=excluded.path, title=excluded.title, type=excluded.type, "
                "tags=excluded.tags, updated=excluded.updated, body_hash=excluded.body_hash, "
                "deleted_at=NULL, extra_json=excluded.extra_json, topic_key=excluded.topic_key, "
                "normalized_hash=excluded.normalized_hash, namespace=excluded.namespace, "
                "normalized_title=excluded.normalized_title, "
                "normalized_content_hash=excluded.normalized_content_hash",
                (
                    id_,
                    path,
                    title,
                    type_,
                    json.dumps(tags),
                    created,
                    updated,
                    body_hash,
                    json.dumps(extra, default=str) if extra is not None else None,
                    topic_key,
                    normalized_hash,
                    namespace,
                    normalized_title,
                    normalized_content_hash,
                    valid_at,
                    invalid_at,
                ),
            )
        elif self._has_pattern_cols:
            cx.execute(
                "INSERT INTO meta (id, path, title, type, tags, created, updated, body_hash, extra_json, topic_key, normalized_hash, valid_at, invalid_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "path=excluded.path, title=excluded.title, type=excluded.type, "
                "tags=excluded.tags, updated=excluded.updated, body_hash=excluded.body_hash, "
                "deleted_at=NULL, extra_json=excluded.extra_json, topic_key=excluded.topic_key, normalized_hash=excluded.normalized_hash",
                (
                    id_,
                    path,
                    title,
                    type_,
                    json.dumps(tags),
                    created,
                    updated,
                    body_hash,
                    json.dumps(extra, default=str) if extra is not None else None,
                    topic_key,
                    normalized_hash,
                    valid_at,
                    invalid_at,
                ),
            )
        else:
            cx.execute(
                "INSERT INTO meta (id, path, title, type, tags, created, updated, body_hash, extra_json, valid_at, invalid_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "path=excluded.path, title=excluded.title, type=excluded.type, "
                "tags=excluded.tags, updated=excluded.updated, body_hash=excluded.body_hash, "
                "deleted_at=NULL, extra_json=excluded.extra_json",
                (
                    id_,
                    path,
                    title,
                    type_,
                    json.dumps(tags),
                    created,
                    updated,
                    body_hash,
                    json.dumps(extra, default=str) if extra is not None else None,
                    valid_at,
                    invalid_at,
                ),
            )
        cx.execute("DELETE FROM vec WHERE id = ?", (id_,))
        cx.execute(
            f"INSERT INTO vec (id, embedding, type) VALUES (?, {self._vec_bind_new()}, ?)",
            (id_, serialize_float32(embedding), type_),
        )
        cx.execute("DELETE FROM fts WHERE id = ?", (id_,))
        cx.execute(
            "INSERT INTO fts (id, title, tags, body) VALUES (?, ?, ?, ?)",
            (id_, title, " ".join(tags), body_text),
        )
        cx.execute(
            "INSERT OR IGNORE INTO access (id, access_count, last_accessed) VALUES (?, 0, ?)",
            (id_, updated),
        )
        cx.execute(
            "INSERT OR IGNORE INTO memory_health (id, confidence, roi_score, updated_at) "
            "VALUES (?, 1.0, 1.0, datetime('now'))",
            (id_,),
        )

    def upsert(
        self,
        *,
        id_: str,
        path: str,
        title: str,
        type_: str,
        tags: list[str],
        created: str,
        updated: str,
        body_hash: str,
        embedding: list[float],
        extra: dict[str, Any] | None = None,
        body_text: str = "",
        topic_key: str | None = None,
        normalized_hash: str | None = None,
        namespace: str | None = None,
        normalized_title: str | None = None,
        normalized_content_hash: str | None = None,
        valid_at: str | None = None,
        invalid_at: str | None = None,
    ) -> None:
        self._validate_embedding(id_, embedding)
        # Dual-write: the tantivy write lock spans the sqlite commit AND the
        # tantivy write so concurrent same-id writers can't commit to sqlite
        # in one order and index tantivy in the other (which would leave
        # tantivy permanently serving the older version).
        with self._tantivy_write_lock:
            with self._tx() as cx:
                self._upsert_memory_row(
                    cx,
                    id_=id_,
                    path=path,
                    title=title,
                    type_=type_,
                    tags=tags,
                    created=created,
                    updated=updated,
                    body_hash=body_hash,
                    embedding=embedding,
                    extra=extra,
                    body_text=body_text,
                    topic_key=topic_key,
                    normalized_hash=normalized_hash,
                    namespace=namespace,
                    normalized_title=normalized_title,
                    normalized_content_hash=normalized_content_hash,
                    valid_at=valid_at,
                    invalid_at=invalid_at,
                )
            tantivy = self._get_tantivy()
            if tantivy is not None:
                try:
                    tantivy.delete_document(id_)
                    tantivy.add_document(id_, title, " ".join(tags), body_text)
                    tantivy.commit()
                except Exception as exc:
                    _log.warning("tantivy upsert failed (FTS5 still current): %s", exc)
                    self._mark_tantivy_unhealthy()

    def upsert_replacing_path_owner(
        self,
        *,
        stale_id: str,
        id_: str,
        path: str,
        title: str,
        type_: str,
        tags: list[str],
        created: str,
        updated: str,
        body_hash: str,
        embedding: list[float],
        extra: dict[str, Any] | None = None,
        body_text: str = "",
        topic_key: str | None = None,
        normalized_hash: str | None = None,
        namespace: str | None = None,
        normalized_title: str | None = None,
        normalized_content_hash: str | None = None,
        verification_state: str = "unverified",
        verified_at: int | None = None,
        review_after: str | None = None,
        valid_at: str | None = None,
        invalid_at: str | None = None,
    ) -> None:
        """Atomically transfer one canonical path from ``stale_id`` to ``id_``.

        Reindex uses this when a hand-edited/restored Markdown file keeps its
        path but changes frontmatter id.  Embedding happens before this method;
        the old row and its signals are deleted in the same transaction that
        inserts the replacement, so either the complete transfer commits or the
        previous searchable row survives.
        """
        self._validate_embedding(id_, embedding)
        replaced_id: str | None = None
        # Lock spans sqlite commit + tantivy write — see upsert().
        with self._tantivy_write_lock:
            with self._tx() as cx:
                owner = cx.execute(
                    "SELECT id FROM meta WHERE path = ?",
                    (path,),
                ).fetchone()
                if owner is not None and str(owner["id"]) != id_:
                    replaced_id = str(owner["id"])
                    if replaced_id != stale_id:
                        raise RuntimeError(
                            f"path owner changed during reindex: {path!r} "
                            f"({stale_id[:8]} -> {replaced_id[:8]})"
                        )
                    cx.execute("DELETE FROM meta WHERE id = ?", (replaced_id,))
                    cx.execute("DELETE FROM vec WHERE id = ?", (replaced_id,))
                    cx.execute("DELETE FROM fts WHERE id = ?", (replaced_id,))
                    cx.execute("DELETE FROM access WHERE id = ?", (replaced_id,))
                    cx.execute("DELETE FROM memory_health WHERE id = ?", (replaced_id,))
                    cx.execute(
                        "DELETE FROM source_feedback_vec WHERE source_id = ?", (replaced_id,)
                    )
                    cx.execute("DELETE FROM source_feedback WHERE source_id = ?", (replaced_id,))
                self._upsert_memory_row(
                    cx,
                    id_=id_,
                    path=path,
                    title=title,
                    type_=type_,
                    tags=tags,
                    created=created,
                    updated=updated,
                    body_hash=body_hash,
                    embedding=embedding,
                    extra=extra,
                    body_text=body_text,
                    topic_key=topic_key,
                    normalized_hash=normalized_hash,
                    namespace=namespace,
                    normalized_title=normalized_title,
                    normalized_content_hash=normalized_content_hash,
                    valid_at=valid_at,
                    invalid_at=invalid_at,
                )
                cx.execute(
                    "UPDATE meta SET review_after = ?, verification_state = ?, verified_at = ? "
                    "WHERE id = ?",
                    (review_after, verification_state, verified_at, id_),
                )

            tantivy = self._get_tantivy()
            if tantivy is not None:
                try:
                    if replaced_id is not None:
                        tantivy.delete_document(replaced_id)
                    tantivy.delete_document(id_)
                    tantivy.add_document(id_, title, " ".join(tags), body_text)
                    tantivy.commit()
                except Exception as exc:
                    _log.warning(
                        "tantivy path-owner replacement failed (FTS5 still current): %s",
                        exc,
                    )
                    self._mark_tantivy_unhealthy()

    def _reset_sidecar_vec_tables(
        self,
        cx: sqlite3.Connection,
        *,
        dimensions_changed: bool,
        present_sidecars: list[tuple[str, str, str, str]],
        stamp_identity: bool,
        current_model: str,
    ) -> None:
        """Invalidate HyPE / episode sidecar vectors + metadata on a rebuild that
        changed the vector space. They share this DB file but keep independent
        watermarks, so equal body/content hashes would otherwise make their
        rebuilders skip incompatible old vectors."""
        for vec_table, meta_table, schema_table, id_column in present_sidecars:
            if dimensions_changed:
                cx.execute(f"DROP TABLE {vec_table}")
                cx.execute(
                    f"CREATE VIRTUAL TABLE {vec_table} USING vec0("
                    f"{id_column} TEXT PRIMARY KEY, "
                    f"embedding FLOAT[{self.dims}] distance_metric=cosine)"
                )
            else:
                cx.execute(f"DELETE FROM {vec_table}")
            cx.execute(f"DELETE FROM {meta_table}")
            if stamp_identity:
                cx.execute(
                    f"CREATE TABLE IF NOT EXISTS {schema_table} ("
                    "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                cx.execute(
                    f"INSERT INTO {schema_table} (key, value) "
                    "VALUES ('embedder_model', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (current_model,),
                )
                cx.execute(
                    f"INSERT INTO {schema_table} (key, value) "
                    "VALUES ('embedder_dims', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (str(self.dims),),
                )

    def replace_memory_index(self, rows: list[dict[str, Any]]) -> int:
        """Atomically replace markdown-derived meta/vector/FTS rows.

        All vectors are validated before the write transaction starts. If any
        row insert fails, SQLite rolls the deletes and every prior insert back,
        leaving the previous searchable index intact.

        A rebuild is also the supported model/profile migration boundary.  When
        dimensionality changes, all three vec0 tables must be recreated because
        their ``FLOAT[N]`` shape is part of the SQLite schema.  When only the
        model identity changes, repo/feedback vectors are invalidated even when
        their dimensions happen to match; retaining them would silently mix two
        embedding spaces.  Their non-vector source rows remain intact and can be
        re-embedded lazily/by their normal rebuild paths.
        """
        prepared_rows: list[dict[str, Any]] = []
        topic_identities: set[tuple[str, str]] = set()
        topic_conflict = False
        for source_row in rows:
            row = dict(source_row)
            self._validate_embedding(str(row["id_"]), list(row["embedding"]))
            namespace, topic_key, norm_title, content_hash = self._derive_identity_values(
                path=str(row["path"]),
                title=str(row["title"]),
                type_=str(row["type_"]),
                tags=list(row["tags"]),
                body_text=str(row.get("body_text") or ""),
                topic_key=row.get("topic_key"),
                namespace=row.get("namespace"),
                normalized_title=row.get("normalized_title"),
                normalized_content_hash=row.get("normalized_content_hash"),
            )
            row["namespace"] = namespace
            row["topic_key"] = topic_key
            row["normalized_title"] = norm_title
            row["normalized_content_hash"] = content_hash
            if namespace is not None and topic_key is not None:
                identity = (namespace, topic_key)
                if identity in topic_identities:
                    topic_conflict = True
                topic_identities.add(identity)
            prepared_rows.append(row)

        stored_model_row = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'embedder_model'"
        ).fetchone()
        stored_dims_row = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'embedder_dims'"
        ).fetchone()
        stored_model = str(stored_model_row["value"]) if stored_model_row else ""
        try:
            stored_dims = int(stored_dims_row["value"]) if stored_dims_row else None
        except (TypeError, ValueError):
            stored_dims = None

        current_model = str(self.embedder_model or "")
        stamp_identity = bool(current_model and "stub" not in current_model.lower())
        # Missing legacy metadata is an unknown vector space, not proof of a
        # match. Invalidating dependent vectors once is safer than silently
        # retaining repo/feedback embeddings under an assumed identity.
        model_changed = bool(stamp_identity and stored_model != current_model)
        derived_sidecars = [
            ("hype_vec", "hype_questions", "hype_schema_meta", "question_id"),
            ("episode_vec", "episode_meta", "episode_schema_meta", "id"),
        ]
        sidecar_dims: dict[str, int | None] = {}
        present_sidecars: list[tuple[str, str, str, str]] = []
        for sidecar in derived_sidecars:
            vec_table = sidecar[0]
            ddl_row = self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                (vec_table,),
            ).fetchone()
            if ddl_row is None:
                continue
            present_sidecars.append(sidecar)
            match = re.search(r"FLOAT\[(\d+)\]", str(ddl_row["sql"] or ""), re.IGNORECASE)
            sidecar_dims[vec_table] = int(match.group(1)) if match else None
        dimensions_changed = any(
            actual is not None and actual != self.dims
            for actual in (
                self._vec_table_dims("vec"),
                self._vec_table_dims("repo_vec"),
                self._vec_table_dims("source_feedback_vec"),
                *(sidecar_dims[sidecar[0]] for sidecar in present_sidecars),
            )
        ) or (stored_dims is not None and stored_dims != self.dims)
        # A quant flip (float32 <-> int8) is baked into the `vec` vec0 column
        # TYPE, so — like a dims change — the table must be recreated. But it
        # touches ONLY `vec`; repo_vec/source_feedback_vec stay float32.
        quant_changed = self._vec_table_dtype("vec") != self.vec_quant

        with self._tx() as cx:
            previous_count = int(cx.execute("SELECT COUNT(*) FROM meta").fetchone()[0])
            if topic_conflict:
                cx.execute("DROP INDEX IF EXISTS idx_meta_active_topic_unique")
            cx.execute("DELETE FROM meta")
            cx.execute("DELETE FROM fts")
            if dimensions_changed:
                # vec0 dimensionality is encoded in virtual-table DDL. SQLite
                # DDL is transactional, so a later row failure restores the old
                # tables and vectors together with meta/FTS.
                cx.execute("DROP TABLE vec")
                cx.execute("DROP TABLE repo_vec")
                cx.execute("DROP TABLE source_feedback_vec")
                self._create_vec_tables(cx)
            else:
                if quant_changed:
                    # Recreate ONLY `vec` at the new precision; _create_vec_tables
                    # is IF NOT EXISTS, so repo_vec/source_feedback_vec are left
                    # intact (the 62 MB reference/repo tier is not wiped).
                    cx.execute("DROP TABLE vec")
                    self._create_vec_tables(cx)
                else:
                    cx.execute("DELETE FROM vec")
                if model_changed:
                    # Same width does not imply the same vector space.
                    cx.execute("DELETE FROM repo_vec")
                    cx.execute("DELETE FROM source_feedback_vec")
            if dimensions_changed or model_changed:
                self._reset_sidecar_vec_tables(
                    cx,
                    dimensions_changed=dimensions_changed,
                    present_sidecars=present_sidecars,
                    stamp_identity=stamp_identity,
                    current_model=current_model,
                )
            for source_row in prepared_rows:
                row = dict(source_row)
                verification_state = str(row.pop("verification_state", "unverified"))
                verified_at = row.pop("verified_at", None)
                review_after = row.pop("review_after", None)
                self._upsert_memory_row(cx, **row)
                cx.execute(
                    "UPDATE meta SET review_after = ?, verification_state = ?, verified_at = ? "
                    "WHERE id = ?",
                    (review_after, verification_state, verified_at, row["id_"]),
                )
            if stamp_identity:
                cx.execute(
                    "INSERT INTO schema_meta (key, value) VALUES ('embedder_model', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (current_model,),
                )
                cx.execute(
                    "INSERT INTO schema_meta (key, value) VALUES ('embedder_dims', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (str(self.dims),),
                )
            identity_capability = "blocked" if topic_conflict else "enabled"
            if not topic_conflict:
                cx.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_meta_active_topic_unique "
                    "ON meta(namespace, topic_key) WHERE namespace IS NOT NULL "
                    "AND topic_key IS NOT NULL AND (deleted_at IS NULL OR deleted_at = '')"
                )
            cx.execute(
                "INSERT INTO schema_meta(key, value) VALUES "
                "('identity_topic_unique', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (identity_capability,),
            )

        # Hold the write lock so per-row dual-writers can't interleave between
        # the rebuild's fts SELECT and the tantivy rebuild (their doc would be
        # wiped from tantivy while still present in sqlite/FTS5).
        with self._tantivy_write_lock:
            try:
                self._rebuild_tantivy_from_sqlite()
            except Exception as exc:
                _log.warning("tantivy rebuild failed after atomic sqlite replace: %s", exc)
                self._mark_tantivy_unhealthy()
        return previous_count

    def update_verification(
        self,
        *,
        id_: str,
        verification_state: str,
        verified_at: int | None,
    ) -> bool:
        """Update Markdown-derived verification metadata for one memory."""
        with self._tx() as cx:
            cur = cx.execute(
                "UPDATE meta SET verification_state = ?, verified_at = ? WHERE id = ?",
                (verification_state, verified_at, id_),
            )
        return cur.rowcount > 0

    def update_validity(
        self,
        *,
        id_: str,
        valid_at: str | None,
        invalid_at: str | None,
    ) -> bool:
        """Set the world-validity interval on one memory.

        `valid_at`/`invalid_at` are intentionally excluded from the `upsert()`
        ON CONFLICT update set (see `_upsert_memory_row`), so a re-save/edit
        never touches them. This dedicated statement is how the reindex-fold and
        the contradiction-supersede own their mutation. Derived chunks mirror
        their parent's interval in the same transaction so an invalid parent
        can never leave searchable chunk ghosts behind."""
        with self._tx() as cx:
            cur = cx.execute(
                "UPDATE meta SET valid_at = ?, invalid_at = ? WHERE id = ?",
                (valid_at, invalid_at, id_),
            )
            if cur.rowcount:
                cx.execute(
                    "UPDATE meta SET valid_at = ?, invalid_at = ? "
                    "WHERE json_extract(extra_json, '$.parent_id') = ?",
                    (valid_at, invalid_at, id_),
                )
        return cur.rowcount > 0

    def verification_candidates(self) -> list[dict[str, Any]]:
        """Rows `{id, verification_state, verified_at}` for memories in a
        decayable state (VERIFIED or STALE) with a `verified_at` set — the
        transition candidates for `_transition_stale_memories`. Targeted (not a
        full scan): UNVERIFIED memories, the default majority, are skipped."""
        rows = self._conn.execute(
            "SELECT id, verification_state, verified_at, review_after FROM meta "
            "WHERE verification_state IN ('verified', 'stale') "
            "AND verified_at IS NOT NULL AND review_after IS NOT NULL"
        ).fetchall()
        return [
            {
                "id": r["id"],
                "verification_state": r["verification_state"],
                "verified_at": r["verified_at"],
                "review_after": r["review_after"],
            }
            for r in rows
        ]

    def update_path(self, id_: str, path: str) -> bool:
        """Update only ``meta.path`` while preserving vector and FTS rows."""
        with self._tx() as cx:
            cur = cx.execute("UPDATE meta SET path = ? WHERE id = ?", (path, id_))
        return cur.rowcount > 0

    def upsert_text_only(
        self,
        *,
        id_: str,
        path: str,
        title: str,
        type_: str,
        tags: list[str],
        created: str,
        updated: str,
        body_hash: str,
        extra: dict[str, Any] | None = None,
        body_text: str = "",
        topic_key: str | None = None,
        normalized_hash: str | None = None,
        namespace: str | None = None,
        normalized_title: str | None = None,
        normalized_content_hash: str | None = None,
        valid_at: str | None = None,
        invalid_at: str | None = None,
    ) -> None:
        """Write metadata + FTS row without a vector embedding.

        This keeps CRUD and BM25 search usable on fresh installs or while
        models are downloading. A later `memo reindex` fills the missing vector.
        `valid_at`/`invalid_at` mirror the `upsert()` semantics: written on
        INSERT, preserved (not clobbered) on the ON CONFLICT re-save.
        """
        namespace, topic_key, normalized_title, normalized_content_hash = (
            self._derive_identity_values(
                path=path,
                title=title,
                type_=type_,
                tags=tags,
                body_text=body_text,
                topic_key=topic_key,
                namespace=namespace,
                normalized_title=normalized_title,
                normalized_content_hash=normalized_content_hash,
            )
        )
        # Lock spans sqlite commit + tantivy write — see upsert().
        with self._tantivy_write_lock:
            with self._tx() as cx:
                if self._has_identity_cols:
                    cx.execute(
                        "INSERT INTO meta (id, path, title, type, tags, created, updated, "
                        "body_hash, extra_json, topic_key, normalized_hash, namespace, "
                        "normalized_title, normalized_content_hash, valid_at, invalid_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(id) DO UPDATE SET "
                        "path=excluded.path, title=excluded.title, type=excluded.type, "
                        "tags=excluded.tags, updated=excluded.updated, body_hash=excluded.body_hash, "
                        "deleted_at=NULL, extra_json=excluded.extra_json, "
                        "topic_key=excluded.topic_key, normalized_hash=excluded.normalized_hash, "
                        "namespace=excluded.namespace, normalized_title=excluded.normalized_title, "
                        "normalized_content_hash=excluded.normalized_content_hash",
                        (
                            id_,
                            path,
                            title,
                            type_,
                            json.dumps(tags),
                            created,
                            updated,
                            body_hash,
                            json.dumps(extra, default=str) if extra is not None else None,
                            topic_key,
                            normalized_hash,
                            namespace,
                            normalized_title,
                            normalized_content_hash,
                            valid_at,
                            invalid_at,
                        ),
                    )
                elif self._has_pattern_cols:
                    cx.execute(
                        "INSERT INTO meta (id, path, title, type, tags, created, updated, body_hash, extra_json, topic_key, normalized_hash, valid_at, invalid_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(id) DO UPDATE SET "
                        "path=excluded.path, title=excluded.title, type=excluded.type, "
                        "tags=excluded.tags, updated=excluded.updated, body_hash=excluded.body_hash, "
                        "deleted_at=NULL, extra_json=excluded.extra_json, topic_key=excluded.topic_key, normalized_hash=excluded.normalized_hash",
                        (
                            id_,
                            path,
                            title,
                            type_,
                            json.dumps(tags),
                            created,
                            updated,
                            body_hash,
                            json.dumps(extra, default=str) if extra is not None else None,
                            topic_key,
                            normalized_hash,
                            valid_at,
                            invalid_at,
                        ),
                    )
                else:
                    cx.execute(
                        "INSERT INTO meta (id, path, title, type, tags, created, updated, body_hash, extra_json, valid_at, invalid_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(id) DO UPDATE SET "
                        "path=excluded.path, title=excluded.title, type=excluded.type, "
                        "tags=excluded.tags, updated=excluded.updated, body_hash=excluded.body_hash, "
                        "deleted_at=NULL, extra_json=excluded.extra_json",
                        (
                            id_,
                            path,
                            title,
                            type_,
                            json.dumps(tags),
                            created,
                            updated,
                            body_hash,
                            json.dumps(extra, default=str) if extra is not None else None,
                            valid_at,
                            invalid_at,
                        ),
                    )
                cx.execute("DELETE FROM fts WHERE id = ?", (id_,))
                cx.execute(
                    "INSERT INTO fts (id, title, tags, body) VALUES (?, ?, ?, ?)",
                    (id_, title, " ".join(tags), body_text),
                )
                # Drop any stale vector — this method writes text-only, so a
                # pre-existing vec row for the same id would make semantic search
                # see the old body. Callers that later embed will re-insert via
                # upsert() with the correct vector.
                cx.execute("DELETE FROM vec WHERE id = ?", (id_,))
                # Seed signal rows so prune/eviction queries don't need COALESCE.
                cx.execute(
                    "INSERT OR IGNORE INTO access (id, access_count, last_accessed) VALUES (?, 0, ?)",
                    (id_, updated),
                )
                cx.execute(
                    "INSERT OR IGNORE INTO memory_health (id, confidence, roi_score, updated_at) "
                    "VALUES (?, 1.0, 1.0, datetime('now'))",
                    (id_,),
                )
            tantivy = self._get_tantivy()
            if tantivy is not None:
                try:
                    tantivy.delete_document(id_)
                    tantivy.add_document(id_, title, " ".join(tags), body_text)
                    tantivy.commit()
                except Exception as exc:
                    _log.warning("tantivy upsert_text_only failed: %s", exc)
                    self._mark_tantivy_unhealthy()

    def has_vector(self, id_: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM vec WHERE id = ? LIMIT 1", (id_,)).fetchone()
        return row is not None

    def get_embedding_blob(self, id_: str) -> bytes | None:
        """Return the raw embedding blob for ``id_``, or None if missing.

        The blob is 4 B/dim float32 by default, 1 B/dim int8 under
        MEMO_VEC_QUANTIZE=int8 — decode it with :meth:`unpack_embedding`, not a
        hardcoded float32 unpack.
        """
        row = self._conn.execute("SELECT embedding FROM vec WHERE id = ?", (id_,)).fetchone()
        return row["embedding"] if row else None

    def unpack_embedding(self, blob: bytes) -> list[float]:
        """Decode a raw `vec` blob to ``list[float]``, dtype-aware.

        Under int8 the stored blob is 1 B/dim signed int8; dequantize (÷127)
        back to the ~unit-norm float range so callers that historically decoded
        float32 (recall vec-cosine boost, delete rollback) keep working.
        """
        import struct

        if self._quant_int8:
            return [x / 127.0 for x in struct.unpack(f"{len(blob)}b", blob)]
        return list(struct.unpack(f"<{len(blob) // 4}f", blob))

    def export_embed_rows(self, *, limit: int = 0) -> list[dict[str, Any]]:
        """Rows whose embeddings are safe to export to the cross-machine sync
        repo (see `sync_embed_cache`): durable-tier memories plus chunk rows of
        a durable parent — never bulk-ingested reference rows, because the
        vault is not part of the sync corpus and its embeddings must not leak
        into it. `title` + `body` are exactly what `_compose_for_embed` was
        given at index time.

        ``limit`` > 0 caps the export to the N most-recently-updated durable
        parents (their chunks ride along) — the shard-size bound: a 2560-dim
        vector is ~13.7KB in base64, so an uncapped mature corpus would put
        tens of MB in the sync repo. Regularly-syncing peers still converge to
        full coverage because the receiving `repo_embedding_cache` persists;
        only a fresh bootstrap of pre-window rows falls back to re-embedding.
        """
        from memo.tiers import DURABLE_TYPES

        durable = sorted(DURABLE_TYPES)
        ph = ",".join("?" for _ in durable)
        rows = self._conn.execute(
            "WITH durable_parents AS ("
            "  SELECT id FROM meta "
            f"  WHERE deleted_at IS NULL AND type IN ({ph}) "
            "  ORDER BY updated DESC, id LIMIT ?"
            ") "
            "SELECT m.id, m.title, f.body AS body "
            "FROM meta m JOIN fts f ON f.id = m.id "
            "WHERE m.deleted_at IS NULL AND ("
            "  m.id IN (SELECT id FROM durable_parents) OR "
            "  json_extract(m.extra_json, '$.parent_id') IN (SELECT id FROM durable_parents))",
            (*durable, limit if limit > 0 else -1),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_fts_body(self, id_: str) -> str:
        """Return the FTS body text for ``id_``, or empty string."""
        row = self._conn.execute("SELECT body FROM fts WHERE id = ?", (id_,)).fetchone()
        return str(row["body"]) if row else ""

    def get_dedup_keys(self, id_: str) -> tuple[str | None, str | None]:
        """Return ``(topic_key, normalized_hash)`` for ``id_`` — the dedup keys.

        These live ONLY in the sqlite ``meta`` index (not in the ``.md``
        frontmatter), so ``get()`` omits them. A delete-rollback that restores
        a row from ``get()`` would drop them, letting a later same-topic upsert
        create a duplicate instead of updating. Delete pre-fetches them here so
        the rollback restores faithfully. Older stores without the pattern
        columns return ``(None, None)``.
        """
        if not self._has_pattern_cols:
            return (None, None)
        row = self._conn.execute(
            "SELECT topic_key, normalized_hash FROM meta WHERE id = ?", (id_,)
        ).fetchone()
        if not row:
            return (None, None)
        return (row["topic_key"], row["normalized_hash"])

    def get_identity_keys(self, id_: str) -> dict[str, str | None]:
        if not self._has_identity_cols:
            return {
                "namespace": None,
                "topic_key": None,
                "normalized_title": None,
                "normalized_content_hash": None,
            }
        row = self._conn.execute(
            "SELECT namespace, topic_key, normalized_title, normalized_content_hash "
            "FROM meta WHERE id = ?",
            (id_,),
        ).fetchone()
        if row is None:
            return {}
        return dict(row)

    def find_active_by_topic_identity(self, namespace: str, topic_key: str) -> list[dict[str, Any]]:
        if not self._has_identity_cols or not namespace or not topic_key:
            return []
        rows = self._conn.execute(
            "SELECT id, path, created, updated, title, type, tags, topic_key, "
            "normalized_hash, namespace, normalized_title, normalized_content_hash "
            "FROM meta WHERE namespace = ? AND topic_key = ? "
            "AND (deleted_at IS NULL OR deleted_at = '') ORDER BY created, id",
            (namespace, canonical_topic_key(topic_key)),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def find_active_by_exact_identity(
        self,
        namespace: str,
        type_: str,
        normalized_title: str,
        normalized_content_hash: str,
    ) -> list[dict[str, Any]]:
        if not self._has_identity_cols or not namespace:
            return []
        rows = self._conn.execute(
            "SELECT id, path, created, updated, title, type, tags, topic_key, "
            "normalized_hash, namespace, normalized_title, normalized_content_hash "
            "FROM meta WHERE namespace = ? AND type = ? AND normalized_title = ? "
            "AND normalized_content_hash = ? "
            "AND (deleted_at IS NULL OR deleted_at = '') ORDER BY created, id",
            (namespace, type_, normalized_title, normalized_content_hash),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def corroborate_identity(self, id_: str, *, seen_at: str) -> dict[str, int | str]:
        """Atomically strengthen one canonical record without rewriting it."""
        with self._tx() as cx:
            cur = cx.execute(
                "UPDATE meta SET duplicate_count = COALESCE(duplicate_count, 0) + 1, "
                "last_seen_at = ? WHERE id = ? "
                "AND (deleted_at IS NULL OR deleted_at = '')",
                (seen_at, id_),
            )
            if cur.rowcount == 0:
                return {"support_count": 0, "duplicate_count": 0, "last_seen_at": seen_at}
            cx.execute(
                "INSERT OR IGNORE INTO memory_health"
                "(id, confidence, roi_score, updated_at, support_count) "
                "VALUES (?, 1.0, 1.0, ?, 0)",
                (id_, seen_at),
            )
            cx.execute(
                "UPDATE memory_health SET support_count = support_count + 1 WHERE id = ?",
                (id_,),
            )
            row = cx.execute(
                "SELECT m.duplicate_count, m.last_seen_at, h.support_count "
                "FROM meta m JOIN memory_health h ON h.id = m.id WHERE m.id = ?",
                (id_,),
            ).fetchone()
        return {
            "support_count": int(row["support_count"]),
            "duplicate_count": int(row["duplicate_count"]),
            "last_seen_at": str(row["last_seen_at"]),
        }

    def attach_topic_identity(
        self, id_: str, *, namespace: str, topic_key: str, seen_at: str
    ) -> bool:
        """Attach the first explicit topic without touching vector/FTS content."""
        with self._tx() as cx:
            cur = cx.execute(
                "UPDATE meta SET namespace = ?, topic_key = ?, last_seen_at = ?, "
                "revision_count = COALESCE(revision_count, 1) + 1 "
                "WHERE id = ? AND topic_key IS NULL "
                "AND (deleted_at IS NULL OR deleted_at = '')",
                (namespace, canonical_topic_key(topic_key), seen_at, id_),
            )
        return cur.rowcount > 0

    def note_identity_revision(self, id_: str, *, seen_at: str) -> None:
        with self._tx() as cx:
            cx.execute(
                "UPDATE meta SET revision_count = COALESCE(revision_count, 1) + 1, "
                "last_seen_at = ? WHERE id = ?",
                (seen_at, id_),
            )

    def revise_identity(self, id_: str, *, seen_at: str) -> dict[str, int | str]:
        """Atomically record a same-topic revision and its supporting evidence."""
        with self._tx() as cx:
            cur = cx.execute(
                "UPDATE meta SET revision_count = COALESCE(revision_count, 1) + 1, "
                "duplicate_count = COALESCE(duplicate_count, 0) + 1, last_seen_at = ? "
                "WHERE id = ? AND (deleted_at IS NULL OR deleted_at = '')",
                (seen_at, id_),
            )
            if cur.rowcount == 0:
                return {"support_count": 0, "duplicate_count": 0, "last_seen_at": seen_at}
            cx.execute(
                "INSERT OR IGNORE INTO memory_health"
                "(id, confidence, roi_score, updated_at, support_count) "
                "VALUES (?, 1.0, 1.0, ?, 0)",
                (id_, seen_at),
            )
            cx.execute(
                "UPDATE memory_health SET support_count = support_count + 1 WHERE id = ?",
                (id_,),
            )
            row = cx.execute(
                "SELECT m.duplicate_count, m.last_seen_at, h.support_count "
                "FROM meta m JOIN memory_health h ON h.id = m.id WHERE m.id = ?",
                (id_,),
            ).fetchone()
        return {
            "support_count": int(row["support_count"]),
            "duplicate_count": int(row["duplicate_count"]),
            "last_seen_at": str(row["last_seen_at"]),
        }

    def reconcile_identity_constraint(self, *, force: bool = False) -> str:
        """Enable the active topic uniqueness index only for a clean corpus."""
        if not self._has_identity_cols:
            return "unavailable"
        if not force:
            cached = self._conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'identity_topic_unique'"
            ).fetchone()
            index_exists = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' "
                "AND name='idx_meta_active_topic_unique'"
            ).fetchone()
            if cached is not None:
                cached_status = str(cached["value"])
                if (cached_status == "enabled" and index_exists is not None) or (
                    cached_status == "blocked" and index_exists is None
                ):
                    return cached_status
        conflict = self._conn.execute(
            "SELECT 1 FROM meta WHERE namespace IS NOT NULL AND topic_key IS NOT NULL "
            "AND (deleted_at IS NULL OR deleted_at = '') "
            "GROUP BY namespace, topic_key HAVING COUNT(*) > 1 LIMIT 1"
        ).fetchone()
        status = "blocked" if conflict is not None else "enabled"
        with self._tx() as cx:
            if conflict is not None:
                cx.execute("DROP INDEX IF EXISTS idx_meta_active_topic_unique")
            else:
                cx.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_meta_active_topic_unique "
                    "ON meta(namespace, topic_key) WHERE namespace IS NOT NULL "
                    "AND topic_key IS NOT NULL AND (deleted_at IS NULL OR deleted_at = '')"
                )
            cx.execute(
                "INSERT INTO schema_meta(key, value) VALUES "
                "('identity_topic_unique', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (status,),
            )
        return status

    def identity_diagnostics(self) -> dict[str, Any]:
        if not self._has_identity_cols:
            return {
                "ok": False,
                "identity_constraint": "unavailable",
                "multiple_project_tag_rows": 0,
                "topic_collision_groups": 0,
                "exact_duplicate_groups": 0,
                "legacy_identity_rows": self.count(),
            }
        capability = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'identity_topic_unique'"
        ).fetchone()
        topic_groups = self._conn.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM meta WHERE namespace IS NOT NULL "
            "AND topic_key IS NOT NULL AND (deleted_at IS NULL OR deleted_at = '') "
            "GROUP BY namespace, topic_key HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        exact_groups = self._conn.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM meta WHERE namespace IS NOT NULL "
            "AND normalized_title IS NOT NULL AND normalized_content_hash IS NOT NULL "
            "AND (deleted_at IS NULL OR deleted_at = '') GROUP BY namespace, type, "
            "normalized_title, normalized_content_hash HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        ambiguous = self._conn.execute(
            "SELECT COUNT(*) FROM meta WHERE namespace IS NULL "
            "AND (deleted_at IS NULL OR deleted_at = '')"
        ).fetchone()[0]
        legacy = self._conn.execute(
            "SELECT COUNT(*) FROM meta WHERE namespace IS NULL OR normalized_title IS NULL "
            "OR normalized_content_hash IS NULL"
        ).fetchone()[0]
        status = str(capability["value"]) if capability else "unavailable"
        return {
            "ok": not any((topic_groups, exact_groups, ambiguous, legacy)) and status == "enabled",
            "identity_constraint": status,
            "multiple_project_tag_rows": int(ambiguous),
            "topic_collision_groups": int(topic_groups),
            "exact_duplicate_groups": int(exact_groups),
            "legacy_identity_rows": int(legacy),
        }

    def get_fts_bodies(self, ids: list[str]) -> dict[str, str]:
        """Batch-fetch FTS body text for many ids in one query.

        Used to feed the rerank candidate pool without one disk-read +
        frontmatter parse per hit. Returns {id: body} for ids present in fts.
        """
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT id, body FROM fts WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        return {r["id"]: str(r["body"]) for r in rows}

    def find_by_topic_key(self, topic_key: str) -> dict[str, str] | None:
        """Legacy global lookup. New writes use composite topic identity."""
        if not topic_key:
            return None
        try:
            row = self._conn.execute(
                "SELECT id, path, created FROM meta "
                "WHERE topic_key = ? AND (deleted_at IS NULL OR deleted_at = '') "
                "LIMIT 1",
                (canonical_topic_key(topic_key),),
            ).fetchone()
        except sqlite3.OperationalError as exc:
            # Older stores may not have pattern columns yet.
            if "no such column" not in str(exc):
                raise
            return None
        return dict(row) if row else None

    def get_fts_body_by_path(self, path: str) -> str:
        """Return indexed FTS body text for a memory path, or an empty string."""
        try:
            row = self._conn.execute(
                "SELECT fts.body AS body FROM fts JOIN meta ON meta.id = fts.id "
                "WHERE meta.path = ? AND (meta.deleted_at IS NULL OR meta.deleted_at = '') "
                "LIMIT 1",
                (path,),
            ).fetchone()
        except sqlite3.OperationalError as exc:
            if "no such column" not in str(exc):
                raise
            row = self._conn.execute(
                "SELECT fts.body AS body FROM fts JOIN meta ON meta.id = fts.id "
                "WHERE meta.path = ? LIMIT 1",
                (path,),
            ).fetchone()
        return str(row["body"]) if row and row["body"] else ""

    def update_meta(
        self,
        *,
        id_: str,
        title: str,
        type_: str,
        tags: list[str],
        updated: str,
        extra: dict[str, Any] | None = None,
        namespace: str | None = None,
        normalized_title: str | None = None,
        normalized_content_hash: str | None = None,
        dedup_keys: tuple[str | None, str | None] | None = None,
    ) -> bool:
        """Patch metadata fields without touching the embedding. Used by
        `Memory.update()` when only title/type/tags/extra changed and
        `body_hash` is unchanged — saves an embedder forward pass.
        Returns True if a row was updated."""
        # Lock spans sqlite commit + tantivy write — see upsert().
        with self._tantivy_write_lock:
            with self._tx() as cx:
                identity_row = cx.execute(
                    "SELECT path, topic_key, normalized_hash FROM meta WHERE id = ?", (id_,)
                ).fetchone()
                existing_body = cx.execute("SELECT body FROM fts WHERE id = ?", (id_,)).fetchone()
                if identity_row is not None:
                    current_topic_key = identity_row["topic_key"]
                    current_normalized_hash = identity_row["normalized_hash"]
                    if dedup_keys is not None:
                        current_topic_key, current_normalized_hash = dedup_keys
                    namespace, current_topic_key, normalized_title, normalized_content_hash = (
                        self._derive_identity_values(
                            path=str(identity_row["path"]),
                            title=title,
                            type_=type_,
                            tags=tags,
                            body_text=str(existing_body["body"] if existing_body else ""),
                            topic_key=current_topic_key,
                            namespace=namespace,
                            normalized_title=normalized_title,
                            normalized_content_hash=normalized_content_hash,
                        )
                    )
                if self._has_identity_cols:
                    cur = cx.execute(
                        "UPDATE meta SET title = ?, type = ?, tags = ?, updated = ?, "
                        "extra_json = ?, namespace = ?, normalized_title = ?, "
                        "normalized_content_hash = ?, topic_key = ?, normalized_hash = ? "
                        "WHERE id = ?",
                        (
                            title,
                            type_,
                            json.dumps(tags),
                            updated,
                            json.dumps(extra, default=str) if extra is not None else None,
                            namespace,
                            normalized_title,
                            normalized_content_hash,
                            current_topic_key,
                            current_normalized_hash,
                            id_,
                        ),
                    )
                elif self._has_pattern_cols:
                    cur = cx.execute(
                        "UPDATE meta SET title = ?, type = ?, tags = ?, updated = ?, "
                        "extra_json = ?, topic_key = ?, normalized_hash = ? WHERE id = ?",
                        (
                            title,
                            type_,
                            json.dumps(tags),
                            updated,
                            json.dumps(extra, default=str) if extra is not None else None,
                            current_topic_key,
                            current_normalized_hash,
                            id_,
                        ),
                    )
                else:
                    cur = cx.execute(
                        "UPDATE meta SET title = ?, type = ?, tags = ?, updated = ?, "
                        "extra_json = ? WHERE id = ?",
                        (
                            title,
                            type_,
                            json.dumps(tags),
                            updated,
                            json.dumps(extra, default=str) if extra is not None else None,
                            id_,
                        ),
                    )
                # Sync FTS title + tags only when the meta row was actually updated.
                # Unconditional delete+insert would create a ghost FTS row for a
                # non-existent id_ (rowcount 0 means no meta row matched).
                body_text = ""
                if cur.rowcount > 0:
                    existing = cx.execute(
                        "SELECT body FROM fts WHERE id = ?",
                        (id_,),
                    ).fetchone()
                    body_text = existing["body"] if existing else ""
                    cx.execute("DELETE FROM fts WHERE id = ?", (id_,))
                    cx.execute(
                        "INSERT INTO fts (id, title, tags, body) VALUES (?, ?, ?, ?)",
                        (id_, title, " ".join(tags), body_text),
                    )
                # Sync vec.type — vec0 has no UPDATE, so delete + reinsert
                existing_emb = cx.execute(
                    "SELECT embedding FROM vec WHERE id = ?", (id_,)
                ).fetchone()
                if existing_emb is not None:
                    cx.execute("DELETE FROM vec WHERE id = ?", (id_,))
                    cx.execute(
                        f"INSERT INTO vec (id, embedding, type) VALUES (?, {self._vec_bind_stored()}, ?)",
                        (id_, existing_emb["embedding"], type_),
                    )
                was_updated = cur.rowcount > 0
            # Dual-write to tantivy (preserve existing body, just update title/tags).
            if was_updated:
                tantivy = self._get_tantivy()
                if tantivy is not None:
                    try:
                        tantivy.delete_document(id_)
                        tantivy.add_document(id_, title, " ".join(tags), body_text)
                        tantivy.commit()
                    except Exception as exc:
                        _log.warning("tantivy update_meta failed: %s", exc)
                        self._mark_tantivy_unhealthy()
        return was_updated

    def get(self, id_: str) -> dict[str, Any] | None:
        # Check for soft delete column
        try:
            row = self._conn.execute(
                f"SELECT {META_SELECT_COLUMNS} FROM meta WHERE id = ? AND deleted_at IS NULL",
                (id_,),
            ).fetchone()
        except sqlite3.OperationalError as e:
            if "no such column" not in str(e):
                raise
            # Fallback for old DBs without deleted_at column
            row = self._conn.execute(
                f"SELECT {META_SELECT_COLUMNS} FROM meta WHERE id = ?",
                (id_,),
            ).fetchone()
        return _row_to_dict(row) if row else None

    def get_batch(self, ids: list[str]) -> list[dict[str, Any]]:
        """Fetch multiple memories by ID in a single query."""
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        # Check for soft delete column
        try:
            rows = self._conn.execute(
                f"SELECT {META_SELECT_COLUMNS} "
                f"FROM meta WHERE id IN ({placeholders}) AND deleted_at IS NULL",
                ids,
            ).fetchall()
        except sqlite3.OperationalError as e:
            if "no such column" not in str(e):
                raise
            # Fallback for old DBs
            rows = self._conn.execute(
                f"SELECT {META_SELECT_COLUMNS} FROM meta WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def find_by_prefix(self, prefix: str, limit: int = 10) -> list[str]:
        """Return ids whose hex starts with `prefix`. Used by the CLI/MCP
        to let callers reference memories by the first ~7 chars (git-style)
        instead of pasting a 32-char UUID4."""
        if not prefix:
            return []
        try:
            rows = self._conn.execute(
                "SELECT id FROM meta WHERE id LIKE ? || '%' AND deleted_at IS NULL "
                "ORDER BY id LIMIT ?",
                (prefix, limit),
            ).fetchall()
        except sqlite3.OperationalError as e:
            if "no such column" not in str(e):
                raise
            # Fallback for old DBs without deleted_at column
            rows = self._conn.execute(
                "SELECT id FROM meta WHERE id LIKE ? || '%' ORDER BY id LIMIT ?",
                (prefix, limit),
            ).fetchall()
        return [r["id"] for r in rows]

    def get_by_path(self, path: str, *, include_deleted: bool = False) -> dict[str, Any] | None:
        """Row for `path`, or None. `include_deleted=True` also matches
        soft-deleted rows — needed by callers reclaiming a path, because a
        tombstone still occupies the UNIQUE(path) index and blocks INSERTs.
        """
        deleted_filter = "" if include_deleted else " AND (deleted_at IS NULL OR deleted_at = '')"
        try:
            row = self._conn.execute(
                f"SELECT {META_SELECT_COLUMNS} FROM meta WHERE path = ?{deleted_filter}",
                (path,),
            ).fetchone()
        except sqlite3.OperationalError as exc:
            if "no such column" not in str(exc):
                raise
            # Fallback for old DBs without deleted_at column
            row = self._conn.execute(
                f"SELECT {META_SELECT_COLUMNS} FROM meta WHERE path = ?",
                (path,),
            ).fetchone()
        return _row_to_dict(row) if row else None

    def get_by_path_ci(self, path: str) -> dict[str, Any] | None:
        """Like `get_by_path`, but case-insensitive on the path.

        The vault sits on a case-insensitive filesystem, so `notes/Foo.md`
        and `Notes/Foo.md` are the same file. Ingest uses this to find an
        existing row regardless of the casing a re-walk produced, so it
        reuses that row's id instead of minting a duplicate. Returns the
        oldest match if several casings somehow coexist.
        """
        try:
            row = self._conn.execute(
                f"SELECT {META_SELECT_COLUMNS} "
                "FROM meta WHERE path = ? COLLATE NOCASE AND (deleted_at IS NULL OR deleted_at = '') ORDER BY created LIMIT 1",
                (path,),
            ).fetchone()
        except sqlite3.OperationalError as exc:
            if "no such column" not in str(exc):
                raise
            # Fallback for old DBs without deleted_at column
            row = self._conn.execute(
                f"SELECT {META_SELECT_COLUMNS} "
                "FROM meta WHERE path = ? COLLATE NOCASE ORDER BY created LIMIT 1",
                (path,),
            ).fetchone()
        return _row_to_dict(row) if row else None

    def has_searchable_body(self, id_: str) -> bool:
        """Return whether the FTS projection retains a non-empty body.

        Vault re-ingest uses this alongside ``body_hash`` so an unchanged
        source can heal a missing or legacy-null FTS projection instead of
        incorrectly taking the idempotent fast path forever.
        """
        row = self._conn.execute(
            "SELECT body FROM fts WHERE id = ? LIMIT 1",
            (id_,),
        ).fetchone()
        return row is not None and isinstance(row["body"], str) and bool(row["body"].strip())

    def vault_ingest_rows(self, label: str) -> list[dict[str, Any]]:
        """Rows produced by `memo ingest` under a given vault `label`.

        Used by `ingest --prune` to find stale chunks (abs_path gone). Filters
        on `source LIKE 'vault-ingest%'` so curated memories (source NULL) are
        never returned — they're managed by `memo reindex` / `doctor --gc`.
        Returns the minimal fields the prune needs: id, path, abs_path,
        parent_path, chunk_seq.
        """
        rows = self._conn.execute(
            "SELECT id, path, "
            "json_extract(extra_json, '$.abs_path') AS abs_path, "
            "json_extract(extra_json, '$.parent_path') AS parent_path, "
            "json_extract(extra_json, '$.chunk_seq') AS chunk_seq "
            "FROM meta "
            "WHERE json_extract(extra_json, '$.vault') = ? "
            "AND json_extract(extra_json, '$.source') LIKE 'vault-ingest%'"
            + self._deleted_filter_sql(),
            (label,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "path": r["path"],
                "abs_path": r["abs_path"],
                "parent_path": r["parent_path"],
                "chunk_seq": r["chunk_seq"],
            }
            for r in rows
        ]

    def file_rows(self, store_path: str) -> list[dict[str, Any]]:
        """All rows belonging to one ingested file: the single-chunk row
        (path == store_path) plus any multi-chunk rows (parent_path ==
        store_path). Used by per-file reconciliation to drop stale chunks
        (e.g. a multi-chunk note edited down to fewer chunks)."""
        rows = self._conn.execute(
            "SELECT id, path FROM meta "
            "WHERE (path = ? "
            "OR json_extract(extra_json, '$.parent_path') = ?)" + self._deleted_filter_sql(),
            (store_path, store_path),
        ).fetchall()
        return [{"id": r["id"], "path": r["path"]} for r in rows]

    def _deleted_filter_sql(self) -> str:
        """`" AND (deleted_at IS NULL OR deleted_at = '')"` when meta has the
        soft-delete column, else "" (pre-migration DBs). Read surfaces append
        this so soft-deleted tombstones never resurface as results."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(meta)").fetchall()}
        return " AND (deleted_at IS NULL OR deleted_at = '')" if "deleted_at" in cols else ""

    def chunks_by_parent(self, parent_path: str, limit: int = 3) -> list[dict[str, Any]]:
        """Newest chunks of one ingested note, by title (descending).

        Dated transcript chunk titles end in `— YYYY-MM-DD`, which sorts
        lexicographically by date, so `ORDER BY title DESC` surfaces the most
        recent day first. Used by recency asks to guarantee the latest chunk of
        a retrieved transcript enters the candidate pool even when it scores too
        low semantically to be retrieved on its own.
        """
        rows = self._conn.execute(
            f"SELECT {META_SELECT_COLUMNS} "
            "FROM meta "
            "WHERE json_extract(extra_json, '$.parent_path') = ? "
            f"{self._deleted_filter_sql()} "
            "ORDER BY title DESC LIMIT ?",
            (parent_path, limit),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def chunks_by_parent_id(self, parent_id: str) -> list[dict[str, Any]]:
        """All chunks whose extra_json contains the given parent_id.
        Used by reindex to prune stale chunks without scanning every row.

        Deliberately INCLUDES soft-deleted rows: the live callers
        (delete_ops / maintain_ops) cascade hard-deletes over stale chunks,
        and a tombstoned chunk must still be reachable for that cleanup."""
        rows = self._conn.execute(
            f"SELECT {META_SELECT_COLUMNS} "
            "FROM meta "
            "WHERE json_extract(extra_json, '$.parent_id') = ?",
            (parent_id,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def chunks_adjacent(
        self, parent_path: str, seq: int, *, before: int = 2, after: int = 2
    ) -> list[dict[str, Any]]:
        """Seq-window of sibling chunks of one ingested note: chunk_seq in
        [seq-before, seq+after], anchor row included, ordered by chunk_seq ASC.
        Cheap SQL over extra_json — no embedder, no MLX."""
        rows = self._conn.execute(
            f"SELECT {META_SELECT_COLUMNS} "
            "FROM meta "
            "WHERE json_extract(extra_json, '$.parent_path') = ? "
            "AND CAST(coalesce(json_extract(extra_json, '$.chunk_seq'), "
            "json_extract(extra_json, '$.chunk_index')) AS INTEGER) BETWEEN ? AND ? "
            f"{self._deleted_filter_sql()} "
            "ORDER BY CAST(coalesce(json_extract(extra_json, '$.chunk_seq'), "
            "json_extract(extra_json, '$.chunk_index')) AS INTEGER) ASC",
            (parent_path, seq - before, seq + after),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def records_around_created(
        self,
        created: str,
        *,
        before: int = 2,
        after: int = 2,
        exclude_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Chronological neighbourhood by `created`: up to `before` strictly
        earlier rows + `after` strictly later rows (ISO strings compare
        chronologically). Returned oldest -> newest, anchor excluded."""
        ex_sql = ""
        ex_params: list[Any] = []
        if exclude_types:
            _ph = ",".join("?" for _ in exclude_types)
            ex_sql = f" AND type NOT IN ({_ph})"
            ex_params = sorted(exclude_types)
        cols = META_SELECT_COLUMNS
        del_sql = self._deleted_filter_sql()
        older = self._conn.execute(
            f"SELECT {cols} FROM meta "
            f"WHERE coalesce(julianday(created), -1e300) < julianday(?)"
            f"{ex_sql}{del_sql} ORDER BY julianday(created) DESC LIMIT ?",
            (created, *ex_params, before),
        ).fetchall()
        newer = self._conn.execute(
            f"SELECT {cols} FROM meta "
            f"WHERE coalesce(julianday(created), -1e300) > julianday(?)"
            f"{ex_sql}{del_sql} ORDER BY julianday(created) ASC LIMIT ?",
            (created, *ex_params, after),
        ).fetchall()
        return [_row_to_dict(r) for r in [*reversed(older), *newer]]

    def list_by_tag(self, tag: str, limit: int = 500) -> list[dict[str, Any]]:
        """Rows whose JSON tags array contains `tag` exactly (tags are stored as
        json.dumps(list), so the quoted token is an exact-tag match)."""
        rows = self._conn.execute(
            f"SELECT {META_SELECT_COLUMNS} "
            f"FROM meta WHERE tags LIKE ?{self._deleted_filter_sql()} "
            "ORDER BY julianday(updated) DESC LIMIT ?",
            (f'%"{tag}"%', limit),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def list_recent(
        self,
        limit: int = 20,
        type_: str | None = None,
        exclude_types: set[str] | None = None,
        updated_since: str | None = None,
    ) -> list[dict[str, Any]]:
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(meta)").fetchall()}
        has_deleted = "deleted_at" in cols
        sql = f"SELECT {META_SELECT_COLUMNS} FROM meta "
        clauses: list[str] = []
        params: list[Any] = []
        if has_deleted:
            clauses.append("deleted_at IS NULL")
        if type_:
            clauses.append("type = ?")
            params.append(type_)
        if exclude_types:
            placeholders = ",".join("?" for _ in exclude_types)
            clauses.append(f"type NOT IN ({placeholders})")
            params.extend(sorted(exclude_types))
        if updated_since:
            # Compare UTC instants so mixed offsets don't drop rows. Wrapping
            # updated in julianday() forces a full table scan since idx_meta_updated
            # is defined on bare updated, not julianday(updated). For large
            # corpora, consider adding: CREATE INDEX idx_meta_updated_jd ON meta(julianday(updated))
            clauses.append("coalesce(julianday(updated), -1e300) >= julianday(?)")
            params.append(updated_since)
        if clauses:
            sql += "WHERE " + " AND ".join(clauses) + " "
        sql += "ORDER BY julianday(updated) DESC LIMIT ?"
        rows = self._conn.execute(sql, (*params, limit)).fetchall()
        return [_row_to_dict(r) for r in rows]

    def _index_has_invalid(self) -> bool:
        """Whether the corpus holds any row with a closed validity interval.

        Backed by the ``idx_meta_invalid_at`` partial index (only invalid rows
        are in it), so this is O(1)-ish even on a large all-valid corpus —
        cheap enough to gate the validity over-fetch on the hot recall path.
        """
        row = self._conn.execute(
            "SELECT EXISTS(SELECT 1 FROM meta WHERE invalid_at IS NOT NULL)"
        ).fetchone()
        return bool(row[0])

    def search(
        self,
        embedding: list[float],
        limit: int = 10,
        type_: str | None = None,
        exclude_types: set[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        exclude_tags: set[str] | None = None,
        include_invalid: bool = False,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        """Top-k by cosine. Returns metadata dicts with a `score` field
        added (1 - distance, so higher = more similar).

        `exclude_types` drops rows whose `type` is in the set (pushed into
        SQL, not post-filtered, so the candidate pool isn't wasted on rows
        the caller will throw away — e.g. the recall hook excluding the
        bulk `reference` tier).

        `include_invalid=False` (default) drops rows whose world-validity
        interval is closed as of now (contradiction-superseded facts stay
        in-index but out of normal recall). `as_of=T` overrides the now-gate
        with a valid-time predicate (rows valid at T); `include_invalid=True`
        bypasses the gate entirely."""
        if len(embedding) != self.dims:
            raise ValueError(
                f"Query embedding dim mismatch: got {len(embedding)}, expected {self.dims}",
            )
        # `type` is a vec0 METADATA column, so type filters are pushed INTO
        # the kNN — the candidate pool is already filtered before the JOINed
        # meta row is read. sqlite-vec supports chained `AND type != ?`
        # predicates, so multi-value exclude_types becomes N individual push
        # predicates rather than a post-join `NOT IN` that required 5x
        # over-fetch to fill the top-k after off-type rows were discarded.
        push_clauses: list[str] = []
        push_params: list[Any] = []
        if type_:
            push_clauses.append("vec.type = ?")
            push_params.append(type_)
        if exclude_types:
            for ex_type in sorted(exclude_types):
                push_clauses.append("vec.type != ?")
                push_params.append(ex_type)
        # Date windows compare instants, not ISO text: stored/imported rows may
        # carry non-UTC offsets. Fetch a wider kNN pool and filter in Python.
        from_dt = _parse_filter_ts(date_from)
        to_dt = _parse_filter_ts(date_to)
        has_date_filter = from_dt is not None or to_dt is not None
        tag_clauses: list[str] = []
        tag_params: list[Any] = []
        for tag in sorted(exclude_tags or ()):
            # tags column stores json.dumps(list[str]); the quoted token is an
            # exact-tag match (mirrors reference-tier exclusion, but tags can't
            # be pushed into vec0 so this filters on the joined meta row).
            tag_clauses.append("meta.tags NOT LIKE ?")
            tag_params.append(f'%"{tag}"%')
        # The validity gate (default now-gate or `as_of`) drops rows AFTER the
        # kNN, so — like the date/tag post-filters above — a plain vec.k = limit
        # can under-fill when the nearest neighbours are the ones filtered out
        # (finding-7). Widen the pool, but only when the gate can actually
        # remove rows: an `as_of` query (may drop future-dated rows) or a corpus
        # that holds any invalid row. The partial-index EXISTS keeps the common
        # all-valid recall path free.
        validity_can_drop = (not include_invalid) and (
            as_of is not None or self._index_has_invalid()
        )
        k_fetch = limit * 4 if (has_date_filter or tag_clauses or validity_can_drop) else limit

        # Check for deleted_at column and build filter
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(meta)").fetchall()}
        has_deleted = "deleted_at" in cols

        deleted_filter = "meta.deleted_at IS NULL" if has_deleted else "1=1"

        sql = (
            "SELECT vec.id AS id, vec.distance AS distance, "
            "       meta.path, meta.title, meta.type, meta.tags, "
            "       meta.created, meta.updated, meta.body_hash, meta.extra_json, "
            "       meta.verification_state, meta.verified_at, "
            "       meta.valid_at, meta.invalid_at "
            f"FROM vec JOIN meta ON vec.id = meta.id AND {deleted_filter} "
            f"WHERE vec.embedding MATCH {self._vec_bind_new()} AND vec.k = ? "
        )
        params: list[Any] = [serialize_float32(embedding), k_fetch]
        for clause in push_clauses:
            sql += f"AND {clause} "
        params.extend(push_params)
        for clause in tag_clauses:
            sql += f"AND {clause} "
        params.extend(tag_params)
        valid_sql, valid_params = _validity_filter("meta.", include_invalid, as_of)
        sql += valid_sql + " "
        params.extend(valid_params)
        sql += "ORDER BY distance ASC LIMIT ?"
        params.append(k_fetch if has_date_filter else limit)
        rows = self._conn.execute(sql, params).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = _row_to_dict(r)
            if has_date_filter:
                updated_dt = _parse_filter_ts(str(d.get("updated") or ""))
                if updated_dt is None:
                    continue
                if from_dt is not None and updated_dt < from_dt:
                    continue
                if to_dt is not None and updated_dt > to_dt:
                    continue
            d["score"] = max(0.0, 1.0 - float(r["distance"]))
            out.append(d)
            if len(out) >= limit:
                break
        return out

    def clear_memory_index(self) -> int:
        """Truncate the markdown-DERIVABLE memory tables (`meta`, `vec`, `fts`)
        so they can be fully replayed from the `.md` source of truth.

        Deliberately does NOT touch the user-signal tables — `access`,
        `memory_health`, `source_feedback`/`source_feedback_vec` — because those
        are PRIMARY data not present in markdown. They key on the stable
        memory `id`, so they re-join after the replay; rows whose memory no
        longer exists on disk become harmless orphans (cleanable by `gc`).
        Repo corpus tables are untouched (separate ingest surface).

        Returns the number of `meta` rows cleared. Use only for a full
        `reindex(rebuild=True)` — never per-row (that path is `delete()`).
        """
        # Lock spans sqlite commit + tantivy clear — see upsert(). Otherwise a
        # concurrent writer could commit sqlite between the two and have its
        # tantivy doc wiped by the clear (present in sqlite, missing in tantivy).
        with self._tantivy_write_lock:
            with self._tx() as cx:
                n = cx.execute("SELECT COUNT(*) FROM meta").fetchone()[0]
                cx.execute("DELETE FROM meta")
                cx.execute("DELETE FROM vec")
                cx.execute("DELETE FROM fts")
            tantivy = self._get_tantivy()
            if tantivy is not None:
                try:
                    # rebuild([]) clears the index while holding tantivy._lock
                    # for the whole delete+commit (direct _writer access didn't).
                    tantivy.rebuild([])
                except Exception as exc:
                    _log.warning("tantivy clear failed during rebuild: %s", exc)
                    self._mark_tantivy_unhealthy()
            elif self.tantivy_index_dir.is_dir():
                # Tantivy unavailable but stale index directory exists — nuke it
                # so _maybe_rebuild_tantivy will do a full rebuild when tantivy
                # becomes available again, rather than serving ghost documents.
                import shutil

                try:
                    shutil.rmtree(self.tantivy_index_dir)
                except Exception as exc:
                    _log.warning("failed to remove stale tantivy index: %s", exc)
        return int(n)

    def delete(self, id_: str) -> bool:
        # Check if soft delete column exists (session pattern)
        from ..flags import flag_bool

        _use_soft = flag_bool("MEMO_SOFT_DELETE")
        has_soft_delete = False
        if _use_soft:
            try:
                col_check = self._conn.execute("PRAGMA table_info(meta)").fetchall()
                columns = {row["name"] for row in col_check}
                has_soft_delete = "deleted_at" in columns
            except Exception:
                _log.debug("soft-delete column check failed")

        if has_soft_delete and _use_soft:
            # Soft delete pattern: mark deleted_at + remove from vec/fts indexes
            import datetime

            now = datetime.datetime.now(datetime.UTC).isoformat()
            # Lock spans sqlite commit + tantivy write — see upsert().
            with self._tantivy_write_lock:
                with self._tx() as cx:
                    cur = cx.execute(
                        "UPDATE meta SET deleted_at = ? WHERE id = ?",
                        (now, id_),
                    )
                    existed = cur.rowcount > 0
                    cx.execute("DELETE FROM vec WHERE id = ?", (id_,))
                    cx.execute("DELETE FROM fts WHERE id = ?", (id_,))
                if existed:
                    tantivy = self._get_tantivy()
                    if tantivy is not None:
                        try:
                            tantivy.delete_document(id_)
                            tantivy.commit()
                        except Exception as exc:
                            _log.warning("tantivy soft-delete failed: %s", exc)
                            self._mark_tantivy_unhealthy()
            return existed

        # Hard delete fallback (for old DBs without deleted_at)
        # Lock spans sqlite commit + tantivy write — see upsert().
        with self._tantivy_write_lock:
            with self._tx() as cx:
                cur = cx.execute("DELETE FROM meta WHERE id = ?", (id_,))
                existed = cur.rowcount > 0
                cx.execute("DELETE FROM vec WHERE id = ?", (id_,))
                cx.execute("DELETE FROM fts WHERE id = ?", (id_,))
                cx.execute("DELETE FROM access WHERE id = ?", (id_,))
                cx.execute("DELETE FROM memory_health WHERE id = ?", (id_,))
                cx.execute("DELETE FROM source_feedback_vec WHERE source_id = ?", (id_,))
                cx.execute("DELETE FROM source_feedback WHERE source_id = ?", (id_,))
            if existed:
                tantivy = self._get_tantivy()
                if tantivy is not None:
                    try:
                        tantivy.delete_document(id_)
                        tantivy.commit()
                    except Exception as exc:
                        _log.warning("tantivy delete failed: %s", exc)
                        self._mark_tantivy_unhealthy()
        return existed

    @safe_operation(
        fallback=False,
        log_level=logging.WARNING,
        error_message="tantivy delete failed",
    )
    def _delete_tantivy_document(self, id_: str) -> bool:
        """Best-effort removal from the optional full-text sidecar."""

        tantivy = self._get_tantivy()
        if tantivy is not None:
            tantivy.delete_document(id_)
            tantivy.commit()
        return True

    def hard_delete(self, id_: str) -> bool:
        """Permanently delete a memory bypassing soft-delete."""
        # Lock spans sqlite commit + tantivy write — see upsert().
        with self._tantivy_write_lock:
            with self._tx() as cx:
                cur = cx.execute("DELETE FROM meta WHERE id = ?", (id_,))
                existed = cur.rowcount > 0
                cx.execute("DELETE FROM vec WHERE id = ?", (id_,))
                cx.execute("DELETE FROM fts WHERE id = ?", (id_,))
                cx.execute("DELETE FROM access WHERE id = ?", (id_,))
                cx.execute("DELETE FROM memory_health WHERE id = ?", (id_,))
                cx.execute("DELETE FROM source_feedback_vec WHERE source_id = ?", (id_,))
                cx.execute("DELETE FROM source_feedback WHERE source_id = ?", (id_,))
            if existed and not self._delete_tantivy_document(id_):
                self._mark_tantivy_unhealthy()
        return existed

    def hard_delete_if_soft_deleted_before(self, id_: str, *, before: str) -> bool:
        """Atomically vacuum an id only while its tombstone remains eligible.

        Selection and deletion are intentionally separate calls so maintenance
        can report per-id failures. The conditional DELETE rechecks
        ``deleted_at`` inside the same transaction and writer lock as all
        attached index/signal deletion, preventing a stale candidate list from
        deleting a record restored by reindex in the meantime.
        """

        with self._tantivy_write_lock:
            with self._tx() as cx:
                cur = cx.execute(
                    "DELETE FROM meta WHERE id = ? AND deleted_at IS NOT NULL "
                    "AND coalesce(julianday(deleted_at), -1e300) < julianday(?)",
                    (id_, before),
                )
                existed = cur.rowcount > 0
                if existed:
                    cx.execute("DELETE FROM vec WHERE id = ?", (id_,))
                    cx.execute("DELETE FROM fts WHERE id = ?", (id_,))
                    cx.execute("DELETE FROM access WHERE id = ?", (id_,))
                    cx.execute("DELETE FROM memory_health WHERE id = ?", (id_,))
                    cx.execute("DELETE FROM source_feedback_vec WHERE source_id = ?", (id_,))
                    cx.execute("DELETE FROM source_feedback WHERE source_id = ?", (id_,))
            if existed and not self._delete_tantivy_document(id_):
                self._mark_tantivy_unhealthy()
        return existed

    def list_soft_deleted(self, before: str | None = None) -> list[str]:
        """List IDs of soft-deleted records, optionally filtered by age."""
        query = "SELECT id FROM meta WHERE deleted_at IS NOT NULL"
        params: list[Any] = []
        if before is not None:
            query += " AND coalesce(julianday(deleted_at), -1e300) < julianday(?)"
            params.append(before)
        rows = self._conn.execute(query, params).fetchall()
        return [r["id"] for r in rows]

    def bulk_update_type(self, ids: list[str], new_type: str) -> int:
        """Reclassify the `type` of many memories in one transaction.

        Only the `meta.type` column changes — the vec embedding and fts
        index are untouched (bodies are unchanged), so this is cheap and
        needs no re-embed. Used by `memo retier`.
        """
        if not ids:
            return 0
        with self._tx() as cx:
            cx.executemany(
                "UPDATE meta SET type = ? WHERE id = ?",
                [(new_type, i) for i in ids],
            )
            # Batch delete + insert for vec.type: read embeddings, then
            # issue two executemany calls instead of 2×N individual stmts.
            placeholders = ",".join("?" for _ in ids)
            vec_rows = {
                r["id"]: r["embedding"]
                for r in cx.execute(
                    f"SELECT id, embedding FROM vec WHERE id IN ({placeholders})", ids
                ).fetchall()
            }
            to_update = [(id_,) for id_ in ids if id_ in vec_rows]
            if to_update:
                cx.executemany("DELETE FROM vec WHERE id = ?", to_update)
                cx.executemany(
                    f"INSERT INTO vec (id, embedding, type) VALUES (?, {self._vec_bind_stored()}, ?)",
                    [(id_, vec_rows[id_], new_type) for (id_,) in to_update],
                )
        return len(ids)

    def secret_store_insert(
        self,
        *,
        id: str,
        name: str,
        kind: str,
        encrypted_blob: bytes,
        nonce: bytes,
        created_at: str,
        detection_method: str | None = None,
        confidence: float | None = None,
    ) -> None:
        """Insert a secret into secret_store."""
        with self._tx() as cursor:
            cursor.execute(
                """
                INSERT INTO secret_store
                (id, name, kind, encrypted_blob, nonce, created_at, detection_method, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (id, name, kind, encrypted_blob, nonce, created_at, detection_method, confidence),
            )

    def secret_store_get(self, name: str) -> dict[str, Any] | None:
        """Fetch a secret by name."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT id, kind, encrypted_blob, nonce, accessed_count FROM secret_store WHERE name = ?",
            (name,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "name": name,
            "kind": row[1],
            "encrypted_blob": row[2],
            "nonce": row[3],
            "accessed_count": row[4],
        }

    def secret_store_list(self, kind: str | None = None) -> list[dict[str, Any]]:
        """List secrets (metadata only)."""
        cursor = self._conn.cursor()
        if kind:
            cursor.execute(
                "SELECT id, name, kind, accessed_count FROM secret_store WHERE kind = ? ORDER BY name",
                (kind,),
            )
        else:
            cursor.execute("SELECT id, name, kind, accessed_count FROM secret_store ORDER BY name")
        rows = cursor.fetchall()
        return [
            {
                "id": row[0],
                "name": row[1],
                "kind": row[2],
                "accessed_count": row[3],
            }
            for row in rows
        ]

    def secret_store_delete(self, name: str) -> bool:
        """Delete a secret by name. Returns True if deleted, False if not found."""
        with self._tx() as cx:
            cursor = cx.execute("DELETE FROM secret_store WHERE name = ?", (name,))
            return cursor.rowcount > 0

    def secret_store_update_encrypted(
        self,
        name: str,
        encrypted_blob: bytes,
        nonce: bytes,
    ) -> None:
        """Atomically rotate one secret's ciphertext and nonce."""
        with self._tx() as cx:
            cursor = cx.execute(
                "UPDATE secret_store SET encrypted_blob = ?, nonce = ? WHERE name = ?",
                (encrypted_blob, nonce, name),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Secret not found during key rotation: {name}")

    def secret_store_increment_access(self, name: str) -> None:
        """Increment access count and update accessed_at timestamp."""
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with self._tx() as cx:
            cx.execute(
                """
                UPDATE secret_store
                SET accessed_count = accessed_count + 1, accessed_at = ?
                WHERE name = ?
                """,
                (now, name),
            )
