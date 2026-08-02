"""Index-health diagnostic for the derived sqlite index.

memo's canonical store is the Markdown corpus; ``memvec.db`` (``meta`` +
sqlite-vec ``vec`` + FTS5 ``fts``, plus derived HyPE sidecars and the
``repo_embedding_cache``) is a **rebuildable index** over it. Over time the
two can drift — a hand-edit in Obsidian, an interrupted delete, a model swap.

``check_index_health`` is a pure, MLX-free SQL diagnostic: it DETECTS those
divergences and, only under ``repair=True``, removes genuinely *derived*
orphans (a vector or index row whose canonical memory is gone). It NEVER
touches a ``.md`` file and never deletes a memory whose ``.md`` still exists —
the source of truth is always safe.

Best-effort by construction: no sub-check may raise. A failed check records its
error in ``errors`` and the diagnostic continues (dream-pass style).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

# Cap every sample list so a badly-diverged store can't return a huge payload.
_SAMPLE_CAP = 10

# Checks that are purely informational — their non-zero count is expected on a
# healthy store (the embed cache holds rows) and must NOT flip status to "issues".
_INFORMATIONAL = frozenset({"stale_caches"})


def _summary(items: list[Any]) -> dict[str, Any]:
    """Uniform ``{count, sample}`` view over a list of offenders."""
    return {"count": len(items), "sample": items[:_SAMPLE_CAP]}


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _resolve_md(cfg: Any, rel_path: str) -> Path | None:
    """Resolve a store ``path`` to an existing canonical ``.md`` file, or None.

    Mirrors ``Memory._resolve_existing``: try ``memory_dir`` (current layout),
    then ``vault_path`` (legacy). A path with a ``#`` fragment (``...#chunk-N``)
    is a derived chunk with no standalone file. Store rows are untrusted, so an
    absolute path or one escaping the root is rejected rather than followed.
    """
    if not rel_path or "#" in rel_path:
        return None
    parts = Path(rel_path).parts
    if Path(rel_path).is_absolute() or ".." in parts:
        return None
    candidate = cfg.memory_dir / rel_path
    if candidate.is_file():
        return candidate
    vault = getattr(cfg, "vault_path", None)
    if vault is not None:
        legacy = Path(vault) / rel_path
        if legacy.is_file():
            return legacy
    return None


def _chunk_parent_id(extra_json: str | None, path: str) -> str | None:
    """Best-effort parent id for a chunk row: ``extra.parent_id`` if present,
    else None (the caller falls back to matching the base path)."""
    if not extra_json:
        return None
    try:
        import json

        extra = json.loads(extra_json)
    except (ValueError, TypeError):
        return None
    parent = extra.get("parent_id") if isinstance(extra, dict) else None
    return str(parent) if parent else None


def check_index_health(cfg: Any, mem: Any, *, repair: bool = False) -> dict[str, Any]:
    """Diagnose (and optionally repair) divergence in the derived sqlite index.

    Returns ``{"status", "checks", "repaired", "errors"}`` where ``status`` is
    ``"ok"`` (clean), ``"issues"`` (divergence found or a sub-check errored) or
    ``"error"`` (the store could not be opened at all). ``repair=True`` only
    ever deletes derived orphans (vectors / chunk index rows / HyPE-attempt
    rows whose canonical memory is gone); it never modifies a ``.md`` file.
    """
    checks: dict[str, dict[str, Any]] = {}
    repaired: dict[str, int] = {}
    errors: list[str] = []

    try:
        store = mem.store
        conn = store.connection
    except Exception as exc:  # store unavailable — nothing to diagnose
        return {
            "status": "error",
            "checks": {},
            "repaired": {},
            "errors": [f"store unavailable: {exc}"],
        }

    # -- 1. FTS bodies NULL/empty (or missing) while the memory row exists ----
    try:
        rows = conn.execute(
            "SELECT m.id AS id, m.path AS path FROM meta m "
            "LEFT JOIN fts f ON f.id = m.id "
            "WHERE f.id IS NULL OR f.body IS NULL OR TRIM(f.body) = ''"
        ).fetchall()
        checks["fts_missing_body"] = _summary(
            [{"id": str(r["id"]), "path": str(r["path"])} for r in rows]
        )
    except Exception as exc:
        errors.append(f"fts_missing_body: {exc}")

    # -- 2. Durable memories with a missing canonical Markdown file ------------
    #     Detect only: a "missing" file may just be an unmounted vault, so we
    #     never delete the index row (it may be the last searchable copy).
    try:
        from memo.tiers import DURABLE_TYPES

        durable = sorted(DURABLE_TYPES)
        placeholders = ",".join("?" for _ in durable)
        rows = conn.execute(
            f"SELECT id, path FROM meta WHERE type IN ({placeholders})",  # noqa: S608
            durable,
        ).fetchall()
        offenders: list[dict[str, Any]] = [
            {"id": str(r["id"]), "path": str(r["path"])}
            for r in rows
            if "#" not in str(r["path"]) and _resolve_md(cfg, str(r["path"])) is None
        ]
        checks["missing_markdown"] = _summary(offenders)
    except Exception as exc:
        errors.append(f"missing_markdown: {exc}")

    # -- 3. Orphan chunk rows whose parent memory is gone ---------------------
    try:
        live = conn.execute("SELECT id, path FROM meta").fetchall()
        live_ids = {str(r["id"]) for r in live}
        live_paths = {str(r["path"]) for r in live}
        chunk_rows = conn.execute(
            "SELECT id, path, extra_json FROM meta WHERE path LIKE '%#chunk-%'"
        ).fetchall()
        offenders = []
        for r in chunk_rows:
            path = str(r["path"])
            parent = _chunk_parent_id(r["extra_json"], path)
            if parent is not None:
                gone = parent not in live_ids
            else:
                gone = path.split("#", 1)[0] not in live_paths
            if gone:
                offenders.append({"id": str(r["id"]), "path": path})
        checks["orphan_chunks"] = _summary(offenders)
        if repair and offenders:
            with store._tx() as cx:
                for o in offenders:
                    cx.execute("DELETE FROM meta WHERE id = ?", (o["id"],))
                    cx.execute("DELETE FROM vec WHERE id = ?", (o["id"],))
                    cx.execute("DELETE FROM fts WHERE id = ?", (o["id"],))
            repaired["orphan_chunks"] = len(offenders)
    except Exception as exc:
        errors.append(f"orphan_chunks: {exc}")

    # -- 4. Orphan vectors (vec rows with no matching meta row) ----------------
    try:
        ids = [
            str(r["id"])
            for r in conn.execute(
                "SELECT id FROM vec WHERE id NOT IN (SELECT id FROM meta)"
            ).fetchall()
        ]
        checks["orphan_vectors"] = _summary(ids)
        if repair and ids:
            with store._tx() as cx:
                for vid in ids:
                    cx.execute("DELETE FROM vec WHERE id = ?", (vid,))
            repaired["orphan_vectors"] = len(ids)
    except Exception as exc:
        errors.append(f"orphan_vectors: {exc}")

    # -- 5. Memories with a meta row but no vector ----------------------------
    #     Rows explicitly stamped `_memo_embed_pending` are a tracked transient
    #     state (model still downloading), not a divergence — excluded.
    try:
        rows = conn.execute(
            "SELECT id, path FROM meta WHERE id NOT IN (SELECT id FROM vec) "
            "AND COALESCE(json_extract(extra_json, '$._memo_embed_pending'), 0) = 0"
        ).fetchall()
        checks["missing_vector"] = _summary(
            [{"id": str(r["id"]), "path": str(r["path"])} for r in rows]
        )
    except Exception as exc:
        errors.append(f"missing_vector: {exc}")

    # -- 6. Vectors whose stored dimension != cfg.embedder_dims ---------------
    #     vec0 enforces a uniform column width, so this is a table-level fact.
    try:
        stored_dims = store._vec_table_dims("vec")
        expected = int(cfg.embedder_dims)
        if stored_dims is not None and stored_dims != expected:
            affected = conn.execute("SELECT COUNT(*) AS n FROM vec").fetchone()
            checks["wrong_dims"] = {
                "count": int(affected["n"]) if affected else 0,
                "sample": [{"stored_dims": stored_dims, "expected_dims": expected}],
            }
        else:
            checks["wrong_dims"] = _summary([])
    except Exception as exc:
        errors.append(f"wrong_dims: {exc}")

    # -- 7. Stored embedder-model tag inconsistent with the configured model --
    try:
        stored_model: str | None = None
        if _table_exists(conn, "schema_meta"):
            row = conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'embedder_model'"
            ).fetchone()
            stored_model = str(row["value"]) if row and row["value"] else None
        # Compare against the STORE's configured model (it carries the same
        # ``@revision`` suffix the vectors were stamped with), not the bare
        # ``cfg.embedder_model`` — otherwise the revision suffix false-positives.
        configured = str(getattr(store, "embedder_model", "") or "").strip()
        mismatch = bool(
            stored_model
            and configured
            and "stub" not in configured.lower()
            and stored_model != configured
        )
        if mismatch:
            affected = conn.execute("SELECT COUNT(*) AS n FROM meta").fetchone()
            checks["model_mismatch"] = {
                "count": int(affected["n"]) if affected else 0,
                "sample": [{"stored_model": stored_model, "configured_model": configured}],
            }
        else:
            checks["model_mismatch"] = _summary([])
    except Exception as exc:
        errors.append(f"model_mismatch: {exc}")

    # -- 8. MD/SQLite divergence (disk body_hash != indexed body_hash) --------
    #     Reproduces the exact index-time hash (sanitize -> sha256_short) so an
    #     unmodified file never false-positives; a hand-edit in Obsidian does.
    try:
        import frontmatter

        from memo.redact import sanitize_memory_input
        from memo.tiers import DURABLE_TYPES
        from memo.util import sha256_short

        # Store is a foundation layer: read the flag via a relative deferred
        # import (matches queries.py/store.py — avoids the module-level memo.flags
        # dependency the architecture-boundary guard forbids).
        from ..flags import flag_bool

        entropy = flag_bool("MEMO_REDACT_ENTROPY")
        durable = sorted(DURABLE_TYPES)
        placeholders = ",".join("?" for _ in durable)
        rows = conn.execute(
            f"SELECT id, path, body_hash FROM meta WHERE type IN ({placeholders})",  # noqa: S608
            durable,
        ).fetchall()
        offenders = []
        for r in rows:
            path = str(r["path"])
            if "#" in path:
                continue
            md_path = _resolve_md(cfg, path)
            if md_path is None:
                continue  # missing file is reported by check #2, not here
            try:
                text = md_path.read_text(encoding="utf-8")
                body = frontmatter.loads(text).content or ""
            except (OSError, UnicodeDecodeError, ValueError):
                continue
            clean = sanitize_memory_input(
                content=body, entropy=entropy, allow_empty_content=True
            ).content
            indexed_hash = str(r["body_hash"]) if r["body_hash"] is not None else ""
            if sha256_short(clean) != indexed_hash:
                offenders.append({"id": str(r["id"]), "path": path})
        checks["md_divergence"] = _summary(offenders)
    except Exception as exc:
        errors.append(f"md_divergence: {exc}")

    # -- 9. Duplicate HyPE attempts (should be exactly one per memory) --------
    try:
        if _table_exists(conn, "hype_attempts"):
            rows = conn.execute(
                "SELECT memory_id, COUNT(*) AS n FROM hype_attempts "
                "GROUP BY memory_id HAVING COUNT(*) > 1"
            ).fetchall()
            offenders = [
                {"memory_id": str(r["memory_id"]), "count": int(r["n"])} for r in rows
            ]
            checks["duplicate_hype_attempts"] = _summary(offenders)
            if repair and offenders:
                removed = 0
                with store._tx() as cx:
                    for o in offenders:
                        cur = cx.execute(
                            "DELETE FROM hype_attempts WHERE memory_id = ? AND rowid NOT IN "
                            "(SELECT MIN(rowid) FROM hype_attempts WHERE memory_id = ?)",
                            (o["memory_id"], o["memory_id"]),
                        )
                        removed += cur.rowcount
                if removed:
                    repaired["duplicate_hype_attempts"] = removed
        else:
            checks["duplicate_hype_attempts"] = _summary([])
    except Exception as exc:
        errors.append(f"duplicate_hype_attempts: {exc}")

    # -- 10. Orphan HyPE attempts (memory_id no longer in meta) ---------------
    try:
        if _table_exists(conn, "hype_attempts"):
            ids = [
                str(r["memory_id"])
                for r in conn.execute(
                    "SELECT memory_id FROM hype_attempts "
                    "WHERE memory_id NOT IN (SELECT id FROM meta)"
                ).fetchall()
            ]
            checks["orphan_hype_attempts"] = _summary(ids)
            if repair and ids:
                with store._tx() as cx:
                    for mid in ids:
                        cx.execute("DELETE FROM hype_attempts WHERE memory_id = ?", (mid,))
                repaired["orphan_hype_attempts"] = len(ids)
        else:
            checks["orphan_hype_attempts"] = _summary([])
    except Exception as exc:
        errors.append(f"orphan_hype_attempts: {exc}")

    # -- 11. Stale caches (informational: embed-cache size vs live corpus) ----
    try:
        if _table_exists(conn, "repo_embedding_cache"):
            cache_rows = conn.execute(
                "SELECT COUNT(*) AS n FROM repo_embedding_cache"
            ).fetchone()
            live_mem = conn.execute("SELECT COUNT(*) AS n FROM meta").fetchone()
            by_model = conn.execute(
                "SELECT model, dims, COUNT(*) AS n FROM repo_embedding_cache "
                "GROUP BY model, dims"
            ).fetchall()
            sample: list[Any] = [
                {
                    "cache_rows": int(cache_rows["n"]) if cache_rows else 0,
                    "live_memories": int(live_mem["n"]) if live_mem else 0,
                }
            ]
            sample.extend(
                {"model": str(r["model"]), "dims": int(r["dims"]), "rows": int(r["n"])}
                for r in by_model
            )
            checks["stale_caches"] = {
                "count": int(cache_rows["n"]) if cache_rows else 0,
                "sample": sample[:_SAMPLE_CAP],
            }
        else:
            checks["stale_caches"] = _summary([])
    except Exception as exc:
        errors.append(f"stale_caches: {exc}")

    problem_count = sum(
        v["count"] for name, v in checks.items() if name not in _INFORMATIONAL
    )
    status = "issues" if problem_count > 0 or errors else "ok"

    return {"status": status, "checks": checks, "repaired": repaired, "errors": errors}


__all__ = ["check_index_health"]
