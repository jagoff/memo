"""Maintenance + provenance operations for `Memory`.

`_MaintainOpsMixin` holds the corpus-wide maintenance surface (reindex, lint,
gc, entity extraction), and provenance lookups.
Replay resolution moved to replay_ops.py; consolidation + synthesis to
consolidate_ops.py.
"""

from __future__ import annotations

import builtins
import contextlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import frontmatter

from memo.dream_chronicle import CHRONICLE_BUCKET
from memo.embedder import assert_valid_embedding
from memo.errors import StorageError
from memo.fact_extraction import fact_edges_from_metadata, upsert_declared_fact_edges
from memo.flags import flag_bool
from memo.identity import (
    canonical_topic_key,
    namespace_for_index,
    normalized_content_hash,
    normalized_title,
)
from memo.lifecycle import FORGET_AFTER_KEY, FORGET_REASON_KEY
from memo.memory._base import _MemoryBase
from memo.memory.record import (
    _EXTRACT_ENTITIES_SYSTEM_PROMPT,
    _VALID_TYPES,
    _derive_title,
    _extract_provenance,
    _log,
    _normalise_tags,
    _now_iso,
    chat_with_timeout,
    is_canonical_memory_id,
    strip_llm_output,
)
from memo.project import LIFECYCLE_ARCHIVE_DIRS
from memo.prompt_overrides import resolve_prompt
from memo.redact import sanitize_memory_input
from memo.tiers import VerificationState
from memo.util import sha256_full as _sha256_full
from memo.util import sha256_short as _sha256_short

# Chronicle diaries carry no id: frontmatter by design (dream_chronicle) —
# skip them like lifecycle archives instead of warning per file.
_REINDEX_SKIP_DIRS: frozenset[str] = LIFECYCLE_ARCHIVE_DIRS | {CHRONICLE_BUCKET}


def _purge_legacy_secret_index(
    store: Any,
    md_path: Path,
    memory_root: Path,
    md_id: str,
    meta: dict[str, Any],
) -> bool:
    """Drop a legacy credential marker from the derived search index."""
    rel = md_path.relative_to(memory_root)
    if meta.get("type") != "secret" and rel.parts[:1] != ("secrets",):
        return False
    if store.get(md_id) is not None:
        store.hard_delete(md_id)
    return True


def _path_has_symlink_component(memory_root: Path, relative_parts: tuple[str, ...]) -> bool:
    """Return whether any component in a canonical Markdown path is a symlink."""
    current = memory_root
    for part in relative_parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


class _MaintainOpsMixin(_MemoryBase):
    # -- provenance ---------------------------------------------------------

    def provenance(self, id_: str) -> dict[str, Any] | None:
        """Return the full provenance trail for a memory.

        Combines the current state (provenance subset of `meta.extra_json`)
        with the per-op history (each save/update event carrying its own
        provenance snapshot in `delta_json`). Returns `None` if the id is
        unknown.

        Shape:

            {
              "id": "<full id>",
              "current": {trace_id, actor_id, route_reason, ...},
              "events": [
                {"ts", "op", "title", "type", "provenance": {...}},
                ...
              ]
            }
        """
        resolved = self.resolve_id(id_)
        if resolved is None:
            return None
        rec = self.store.get(resolved)
        if rec is None:
            return None
        current = _extract_provenance(rec.get("extra") or {})
        events: list[dict[str, Any]] = []
        for raw in self.history.list_recent(limit=10_000, record_id=resolved):
            entry: dict[str, Any] = {
                "ts": raw.get("ts"),
                "op": raw.get("op"),
                "title": raw.get("title"),
                "type": raw.get("type"),
            }
            delta = raw.get("delta") or {}
            if isinstance(delta, dict) and "_provenance" in delta:
                prov = delta["_provenance"]
                # save op stores `{...keys...}`; update op stores
                # `[old_dict, new_dict]` (delta-pair convention). Surface
                # the post-state in both cases.
                if isinstance(prov, list) and len(prov) == 2:
                    entry["provenance"] = prov[1] or {}
                elif isinstance(prov, dict):
                    entry["provenance"] = prov
            events.append(entry)
        events.reverse()  # oldest first
        return {"id": resolved, "current": current, "events": events}

    # -- reindex / gc -------------------------------------------------------

    def _embed_cached(self, text: str, *, ctx: str) -> list[float]:
        """Embed `text`, reusing a content-addressed cache to avoid redundant
        forward passes on `force`/`rebuild` reindexes of unchanged content.

        Keyed on `(embedder_model, embedder_dims, sha256(text))` — identical
        content under the same model always maps to the same vector, so a hit
        is always correct. A model/dims swap changes the key, forcing a fresh
        embed (the right behaviour). The cache survives `clear_memory_index()`
        (it lives in `repo_embedding_cache`, untouched by the rebuild), so a
        full rebuild after the cache is warm issues zero embedder calls.
        """
        model = self.store.embedder_model
        dims = self.store.dims
        input_hash = _sha256_full(text)
        cached = self.store.get_repo_embedding_cache(
            model=model, dims=dims, input_hashes=[input_hash]
        )
        hit = cached.get(input_hash)
        if hit is not None:
            return hit
        emb = self.embedder.embed([text])[0]
        assert_valid_embedding(emb, dims, context=ctx)
        import contextlib

        with contextlib.suppress(Exception):
            self.store.upsert_repo_embedding_cache(
                model=model,
                dims=dims,
                embeddings=[(input_hash, list(emb))],
                created_at=_now_iso(),
            )
        return emb

    def reindex(self, *, force: bool = False, rebuild: bool = False) -> dict[str, int]:
        """Reindex against a stable Markdown snapshot shared with all CRUD."""
        with self._data_dir_write_lock():
            return self._reindex_locked(force=force, rebuild=rebuild)

    def _reindex_locked(self, *, force: bool = False, rebuild: bool = False) -> dict[str, int]:
        """Scan the memory dir, re-embed entries whose on-disk body
        diverged from `body_hash`. Picks up edits the user made in
        Obsidian directly. Also indexes any `.md` with a valid `id` in
        frontmatter that the store doesn't know about (e.g. restored
        from a backup or copied from another machine).

        With `force=True`, re-embeds EVERY indexed entry regardless of
        body_hash match. Use after an embedder model swap, after a
        change to `_compose_for_embed`, or to refresh the index after
        a corruption/incident.

        With `rebuild=True`, first preflights the full markdown corpus and
        atomically replaces the markdown-derivable tables (`meta`/`vec`/`fts`)
        — the markdown-is-truth reset. User-signal tables (`access`,
        `memory_health`, `source_feedback*`) are preserved and re-join on the
        stable `id`, so a rebuild never destroys feedback/telemetry. Implies
        `force`. Embedding reuse (see `_embed_cached`) keeps it cheap once the
        content cache is warm.

        When `MEMO_CHUNK_INGEST=1`, long notes (> chunker DEFAULT_TARGET_CHARS)
        are additionally split into heading-aware chunk records. Each chunk is
        stored with type='reference', extra.parent_id pointing to the parent
        memory id, and a path of `<rel>#chunk-<n>`. The parent record itself
        is always indexed whole-note for semantic coherence. Default off
        (MEMO_CHUNK_INGEST=0) preserves existing whole-note behaviour.

        Returns counts including checked, reindexed, added, skipped, and facts.
        """
        memory_root = self.cfg.memory_dir
        checked = reindexed = added = skipped = facts = 0
        if not memory_root.is_dir():
            return {"checked": 0, "reindexed": 0, "added": 0, "skipped": 0, "facts": 0}

        chunk_ingest = flag_bool("MEMO_CHUNK_INGEST")
        rebuild_rows: list[dict[str, Any]] | None = [] if rebuild else None
        rebuild_fact_edges: list[dict[str, Any]] = []
        pending_marker_paths: list[tuple[Path, str]] = []
        canonical_parent_count = 0
        canonical_paths_by_id: dict[str, Path] = {}

        if rebuild:
            # Safety: never truncate a populated index against an empty disk.
            # If data_dir lost its .md (deleted dir, half-broken clone) a rebuild
            # would clear the derivable tables and replay nothing — wiping the only
            # surviving copy. Refuse so the markdown can be restored first.
            if next(memory_root.rglob("*.md"), None) is None:
                indexed = self.store.count()
                if indexed > 0:
                    raise StorageError(
                        f"reindex --rebuild refused: data_dir {memory_root} has 0 .md "
                        f"but the index holds {indexed} memories — rebuilding would wipe "
                        "them. Restore the .md first (`memo sync bootstrap <url>`, or "
                        "`git -C <repo> restore .`), or run `memo reindex` (no --rebuild)."
                    )
            # Parse and embed everything before touching the current index.
            # The final SQLite replacement is one transaction, so any row
            # failure rolls back to the previous searchable state.
            force = True

        for md_path in sorted(memory_root.rglob("*.md")):
            checked += 1
            relative_parts = md_path.relative_to(memory_root).parts
            if relative_parts[:1] and relative_parts[0] in _REINDEX_SKIP_DIRS:
                # Current lifecycle archives live under ``inactive``; older
                # vaults used ``archived``.  Both may retain a canonical id so
                # a human can recover a note by moving it back out, but neither
                # may be re-absorbed automatically on reindex/sync.  A project
                # bucket can never collide here — project_bucket() remaps those
                # two slugs to ``_inactive``/``_archived`` (see project.py).
                # ``_chronicle`` diaries are not memories at all.
                skipped += 1
                continue
            if _path_has_symlink_component(memory_root, relative_parts):
                message = f"reindex: refusing symlinked canonical path {md_path}"
                if rebuild_rows is not None:
                    raise StorageError(message)
                _log.warning(message)
                skipped += 1
                continue
            try:
                source_text = md_path.read_text(encoding="utf-8")
                post = frontmatter.loads(source_text)
            except Exception as exc:
                if rebuild_rows is not None:
                    raise StorageError(
                        f"reindex rebuild preflight failed for {md_path}: {exc}"
                    ) from exc
                _log.warning("reindex: skipping %s (parse error): %s", md_path.name, exc)
                skipped += 1
                continue
            meta: dict[str, Any] = post.metadata
            md_id = meta.get("id")
            rel_path = md_path.relative_to(self.cfg.memory_dir)
            is_secret = meta.get("type") == "secret" or rel_path.parts[:1] == ("secrets",)
            if is_secret:
                if (
                    rebuild_rows is None
                    and isinstance(md_id, str)
                    and _purge_legacy_secret_index(
                        self.store,
                        md_path,
                        self.cfg.memory_dir,
                        md_id,
                        meta,
                    )
                ):
                    self._purge_version_history(md_id)
                skipped += 1
                continue
            if not is_canonical_memory_id(md_id):
                _log.warning("reindex: skipping %s (invalid memory id)", md_path.name)
                skipped += 1
                continue
            md_id = str(md_id)
            duplicate_path = canonical_paths_by_id.get(md_id)
            if duplicate_path is not None:
                message = (
                    f"reindex: duplicate canonical id {md_id} in {duplicate_path} and {md_path}"
                )
                if rebuild_rows is not None:
                    raise StorageError(message)
                _log.warning(message)
                skipped += 1
                continue
            canonical_paths_by_id[md_id] = md_path
            canonical_parent_count += 1
            body = post.content or ""
            existing = None if rebuild_rows is not None else self.store.get(md_id)
            prior = existing if existing is not None else self.store.get(md_id)
            topic_key = meta.get("topic_key")
            normalized_hash = meta.get("normalized_hash")
            # Path relative to memory_dir — paths in the store no longer
            # carry the legacy `<vault>/<memory_subdir>/...` prefix.
            rel = str(rel_path)

            title = (meta.get("title") or _derive_title(body) or "untitled").strip()
            type_ = meta.get("type") or "note"
            if type_ not in _VALID_TYPES:
                _log.warning(
                    "reindex: invalid type %r in %s, coercing to 'note'", type_, md_path.name
                )
                type_ = "note"
            raw_tags = meta.get("tags") or []
            # A scalar YAML string (`tags: python`) must not be char-split by
            # list() — treat it as a single (optionally comma-separated) value.
            if isinstance(raw_tags, str):
                raw_tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
            tags = _normalise_tags(list(raw_tags))
            # YAML frontmatter may store tags as a comma-separated string
            # instead of a list (hand-edited files). Normalize to list.
            if tags and isinstance(tags[0], str) and "," in tags[0]:
                tags = _normalise_tags([t.strip() for t in tags[0].split(",") if t.strip()])
            created = meta.get("created") or _now_iso()
            updated = meta.get("updated") or created
            extra = meta.get("extra") or {}
            if not isinstance(extra, dict):
                extra = {}
            # Obsidian-friendly: accept `forget_after` / `forget_reason` as
            # TOP-LEVEL frontmatter keys (what a user naturally types in their
            # editor), folding them into the extra bag the lifecycle layer
            # reads. The nested `extra:` form still works and takes precedence.
            for _fk in (FORGET_AFTER_KEY, FORGET_REASON_KEY):
                if _fk in meta and _fk not in extra:
                    extra = {**extra, _fk: meta[_fk]}

            # Markdown remains untouched, but every derived representation is
            # sanitized before hashing, embedding, FTS, or metadata indexing.
            sanitized = sanitize_memory_input(
                content=body,
                title=title,
                tags=tags,
                topic_key=str(topic_key) if topic_key is not None else None,
                normalized_hash=(str(normalized_hash) if normalized_hash is not None else None),
                extra=extra,
                entropy=flag_bool("MEMO_REDACT_ENTROPY"),
                allow_empty_content=True,
            )
            body = sanitized.content
            title = sanitized.title or "untitled"
            tags = _normalise_tags(sanitized.tags)
            topic_key = sanitized.topic_key
            normalized_hash = sanitized.normalized_hash
            extra = sanitized.extra
            new_hash = _sha256_short(body)
            identity_namespace = namespace_for_index(tags, path=rel)
            identity_topic_key = canonical_topic_key(topic_key)
            identity_title = normalized_title(title)
            identity_content_hash = normalized_content_hash(body)

            # Extract verified_at timestamp (can be None)
            verified_at = meta.get("verified_at")
            if verified_at is not None and not isinstance(verified_at, int):
                try:
                    verified_at = int(verified_at)
                except (ValueError, TypeError):
                    verified_at = None
            raw_verification_state = meta.get(
                "verification_state",
                (prior or {}).get("verification_state", VerificationState.UNVERIFIED.value),
            )
            try:
                verification_state = VerificationState(str(raw_verification_state)).value
            except ValueError:
                verification_state = VerificationState.UNVERIFIED.value
            if "verified_at" not in meta and prior is not None:
                verified_at = prior.get("verified_at")
            # A memory marked VERIFIED without an explicit verified_at enters the
            # decay clock now, so _transition_stale_memories can age it and the
            # recall penalty can distinguish fresh vs old verifications.
            if verification_state == VerificationState.VERIFIED.value and verified_at is None:
                verified_at = int(time.time())
            review_after = meta.get("review_after")
            if "review_after" not in meta and prior is not None:
                review_after = prior.get("review_after")

            # World-validity interval (bi-temporal). `valid_at` is always written
            # to frontmatter (defaulted to `created` on save); `invalid_at` is
            # written ONLY when non-None (open intervals omit it, like
            # `verified_at`). So an ABSENT key means "leave the current index
            # value as-is" — never "clear it" — mirroring the verified_at
            # fallback above: only a value actually present in the markdown folds.
            valid_at = meta.get("valid_at")
            if "valid_at" not in meta and prior is not None:
                valid_at = prior.get("valid_at")
            invalid_at = meta.get("invalid_at")
            if "invalid_at" not in meta and prior is not None:
                invalid_at = prior.get("invalid_at")

            if existing is None:
                # Path-collision guard: an .md may have its frontmatter id
                # regenerated (manual edit, restore-from-backup, or a stale
                # row pointing at a file whose id was rewritten) while the
                # vault-relative path stays the same. The store's
                # UNIQUE(meta.path) constraint blocks a plain INSERT, so the
                # store transfers ownership to the new id atomically after the
                # replacement embedding has succeeded.
                # include_deleted also finds a soft-deleted tombstone, which
                # still holds the path in the UNIQUE index (a soft delete
                # would leave it there and the INSERT would fail again). Do NOT
                # hard-delete here: an embed/upsert failure must leave the
                # previous searchable row intact.
                stale = (
                    self.store.get_by_path(rel, include_deleted=True)
                    if rebuild_rows is None
                    else None
                )
                if stale is not None:
                    _log.warning(
                        "reindex: path %r reused with new id (%s → %s); replacing stale row",
                        rel,
                        stale["id"][:8],
                        md_id[:8],
                    )
                had_embed_pending = False
                if isinstance(extra, dict):
                    extra = dict(extra)
                    had_embed_pending = extra.pop("_memo_embed_pending", None) is not None
                try:
                    emb = self._embed_cached(
                        self._compose_for_embed(title, body), ctx=f"reindex add {md_id[:8]}"
                    )
                    row = dict(
                        id_=md_id,
                        path=rel,
                        title=title,
                        type_=type_,
                        tags=tags,
                        created=created,
                        updated=updated,
                        body_hash=new_hash,
                        embedding=emb,
                        extra=extra if extra else None,
                        body_text=body,
                        topic_key=topic_key,
                        normalized_hash=normalized_hash,
                        namespace=identity_namespace,
                        normalized_title=identity_title,
                        normalized_content_hash=identity_content_hash,
                        verification_state=verification_state,
                        verified_at=verified_at,
                        review_after=review_after,
                        valid_at=valid_at,
                        invalid_at=invalid_at,
                    )
                    if rebuild_rows is not None:
                        rebuild_rows.append(row)
                    else:
                        if stale is not None:
                            self.store.upsert_replacing_path_owner(
                                stale_id=str(stale["id"]),
                                **row,
                            )
                        else:
                            verification = {
                                "verification_state": row.pop("verification_state"),
                                "verified_at": row.pop("verified_at"),
                                "review_after": row.pop("review_after"),
                            }
                            self.store.upsert(**row)
                            self.store.update_review_state(id_=md_id, **verification)
                except Exception as exc:
                    if rebuild_rows is not None:
                        raise StorageError(
                            f"reindex rebuild preflight failed for {md_path}: {exc}"
                        ) from exc
                    _log.warning("reindex: skipping %s (embed failed): %s", md_path.name, exc)
                    skipped += 1
                    continue
                if had_embed_pending:
                    pending_marker_paths.append((md_path, source_text))
                added += 1
                if chunk_ingest:
                    added += self._reindex_emit_chunks(
                        parent_id=md_id,
                        parent_rel=rel,
                        title=title,
                        body=body,
                        tags=tags,
                        created=created,
                        updated=updated,
                        valid_at=valid_at,
                        invalid_at=invalid_at,
                        force=force,
                        rebuild_rows=rebuild_rows,
                    )
                elif rebuild_rows is None:
                    self._prune_chunks(md_id)
                declared_edges = fact_edges_from_metadata(
                    record_id=md_id,
                    title=title,
                    type_=type_,
                    created=created,
                    updated=updated,
                    extra=extra if isinstance(extra, dict) else None,
                    top_level=meta.get("fact_edges"),
                )
                facts += len(declared_edges)
                if rebuild_rows is not None:
                    rebuild_fact_edges.extend(declared_edges)
                else:
                    self.fact_edges.delete_for_source(md_id)
                    for edge in declared_edges:
                        self.fact_edges.upsert_fact(**edge)
                continue
            missing_vector = not self.store.has_vector(md_id)
            path_changed = existing["path"] != rel
            body_changed = existing["body_hash"] != new_hash
            title_changed = title != existing["title"]
            if force or body_changed or title_changed or missing_vector:
                had_embed_pending = False
                if isinstance(extra, dict):
                    extra = dict(extra)
                    had_embed_pending = extra.pop("_memo_embed_pending", None) is not None
                try:
                    emb = self._embed_cached(
                        self._compose_for_embed(title, body), ctx=f"reindex update {md_id[:8]}"
                    )
                    self.store.upsert(
                        id_=md_id,
                        path=rel,
                        title=title,
                        type_=type_,
                        tags=tags,
                        created=existing["created"],
                        updated=_now_iso() if body_changed or title_changed else updated,
                        body_hash=new_hash,
                        embedding=emb,
                        extra=extra if extra else None,
                        body_text=body,
                        topic_key=topic_key,
                        normalized_hash=normalized_hash,
                        namespace=identity_namespace,
                        normalized_title=identity_title,
                        normalized_content_hash=identity_content_hash,
                    )
                    self.store.update_review_state(
                        id_=md_id,
                        review_after=review_after,
                        verification_state=verification_state,
                        verified_at=verified_at,
                    )
                    self.store.update_validity(
                        id_=md_id,
                        valid_at=valid_at,
                        invalid_at=invalid_at,
                    )
                except Exception as exc:
                    _log.warning("reindex: skipping %s (re-embed failed): %s", md_path.name, exc)
                    skipped += 1
                    continue
                if had_embed_pending:
                    pending_marker_paths.append((md_path, source_text))
                reindexed += 1
            else:
                if path_changed:
                    self.store.update_path(md_id, rel)
                    reindexed += 1
                # Metadata-only frontmatter change (tags/type/extra changed,
                # body unchanged) — update meta without re-embedding.
                meta_changed = (
                    type_ != existing["type"]
                    or tags != existing["tags"]
                    or extra != (existing.get("extra") or {})
                    or identity_namespace != existing.get("namespace")
                    or identity_topic_key != existing.get("topic_key")
                    or identity_title != existing.get("normalized_title")
                    or identity_content_hash != existing.get("normalized_content_hash")
                    or normalized_hash != existing.get("normalized_hash")
                    or review_after != existing.get("review_after")
                )
                if meta_changed:
                    self.store.update_meta(
                        id_=md_id,
                        title=title,
                        type_=type_,
                        tags=tags,
                        updated=_now_iso(),
                        extra=extra if extra else None,
                        namespace=identity_namespace,
                        normalized_title=identity_title,
                        normalized_content_hash=identity_content_hash,
                        dedup_keys=(identity_topic_key, normalized_hash),
                    )
                self.store.update_review_state(
                    id_=md_id,
                    review_after=review_after,
                    verification_state=verification_state,
                    verified_at=verified_at,
                )
                self.store.update_validity(
                    id_=md_id,
                    valid_at=valid_at,
                    invalid_at=invalid_at,
                )
            if chunk_ingest:
                reindexed += self._reindex_emit_chunks(
                    parent_id=md_id,
                    parent_rel=rel,
                    title=title,
                    body=body,
                    tags=tags,
                    created=created,
                    updated=updated,
                    valid_at=valid_at,
                    invalid_at=invalid_at,
                    force=force,
                    rebuild_rows=None,
                )
            else:
                self._prune_chunks(md_id)
            self.fact_edges.delete_for_source(md_id)
            facts += upsert_declared_fact_edges(
                self.fact_edges,
                record_id=md_id,
                title=title,
                type_=type_,
                created=created,
                updated=updated,
                extra=extra if isinstance(extra, dict) else None,
                top_level=meta.get("fact_edges"),
            )
        if rebuild_rows is not None:
            if canonical_parent_count == 0:
                indexed = self.store.count()
                if indexed > 0:
                    raise StorageError(
                        f"reindex --rebuild refused: data_dir {memory_root} has no valid "
                        f"canonical memories but the index holds {indexed} memories — "
                        "rebuilding would wipe them. Restore canonical Markdown first."
                    )
            try:
                cleared = self.store.replace_memory_index(rebuild_rows)
            except Exception as exc:
                raise StorageError(
                    f"reindex rebuild atomic index replace failed; previous index preserved: {exc}"
                ) from exc
            _log.info(
                "reindex(rebuild): atomically replaced %d derivable rows with %d rows",
                cleared,
                len(rebuild_rows),
            )
            # A model/profile migration invalidates source-feedback vectors in
            # the same transaction as the main index replacement. Rehydrate
            # them from their preserved query_text rows now. This is
            # best-effort: the canonical memory rebuild already committed and
            # feedback boosting may safely remain disabled until a later run.
            try:
                feedback_reembedded = self.store.rebuild_feedback_vecs(self.embedder.embed_query)
                if feedback_reembedded:
                    _log.info(
                        "reindex(rebuild): re-embedded %d source-feedback vectors",
                        feedback_reembedded,
                    )
            except Exception as exc:
                _log.warning(
                    "reindex(rebuild): source-feedback vector refresh deferred: %s",
                    exc,
                )
            cleared_facts = self.fact_edges.clear()
            for edge in rebuild_fact_edges:
                self.fact_edges.upsert_fact(**edge)
            _log.info(
                "reindex(rebuild): replaced %d temporal fact edges with %d edges",
                cleared_facts,
                len(rebuild_fact_edges),
            )

        for pending_path, expected_source_text in pending_marker_paths:
            try:
                with self._data_dir_write_lock():
                    if pending_path.is_symlink():
                        continue
                    current_text = pending_path.read_text(encoding="utf-8")
                    if current_text != expected_source_text:
                        continue
                    pending_post = frontmatter.loads(current_text)
                    raw_extra = pending_post.metadata.get("extra")
                    pending_extra = dict(raw_extra) if isinstance(raw_extra, dict) else {}
                    pending_extra.pop("_memo_embed_pending", None)
                    pending_post.metadata["extra"] = pending_extra
                    pending_rel_path = pending_path.relative_to(self.cfg.memory_dir).as_posix()
                    self._atomic_write_text(pending_rel_path, frontmatter.dumps(pending_post))
            except Exception as exc:
                _log.debug(
                    "reindex: could not clear _memo_embed_pending from %s: %s",
                    pending_path.name,
                    exc,
                )

        # Successful reindex: every meta.path now uses the current
        # memory_dir-relative layout, so future startups can skip the
        # legacy-path probe in `_maybe_warn_legacy_paths`.
        self.store.set_user_version(1)
        counts = {
            "checked": checked,
            "reindexed": reindexed,
            "added": added,
            "skipped": skipped,
            "facts": facts,
        }
        if reindexed or added:
            try:
                self.operational.receipt(
                    "reindex",
                    subject_uri="memo://maintenance/reindex",
                    metadata={
                        "checked": checked,
                        "reindexed": reindexed,
                        "added": added,
                        "skipped": skipped,
                        "facts": facts,
                        "force": force,
                    },
                )
            except Exception as exc:
                # The markdown/index rebuild is already authoritative and
                # committed. A pre-activation runtime has no operational
                # authority, so receipt emission must fail closed without
                # turning a successful, idempotent bootstrap into a partial
                # onboarding failure.
                _log.warning("native reindex receipt failed: %s", exc)
        # Rebuild crossref edges from disk (flag-gated) — markdown is truth,
        # so hand-edited '- relation [[target]]' lines win on reindex.
        from memo.flags import flag_bool as _flag_bool

        if _flag_bool("MEMO_CROSSREF_INDEX"):
            try:
                self.crossref.reset()
                for rec in self.list(limit=100_000):  # type: ignore[attr-defined]
                    if rec.body:
                        self.crossref.index_source(rec.id, rec.body)
            except Exception as exc:
                _log.debug("reindex: crossref rebuild skipped — %s", exc)

        return counts

    def _reindex_emit_chunks(
        self,
        *,
        parent_id: str,
        parent_rel: str,
        title: str,
        body: str,
        tags: builtins.list[str],
        created: str,
        updated: str,
        valid_at: str | None,
        invalid_at: str | None,
        force: bool,
        rebuild_rows: list[dict[str, Any]] | None = None,
    ) -> int:
        """Split `body` into heading-aware chunks and upsert each as a
        reference-tier record with `extra.parent_id` pointing back.

        Only called when `MEMO_CHUNK_INGEST=1`. Returns the number of
        chunk rows that were written (added or updated). Short bodies
        that produce a single chunk are skipped — the parent already
        covers them.
        """
        from memo.chunker import DEFAULT_TARGET_CHARS, chunk_markdown

        chunks = chunk_markdown(body, target_chars=DEFAULT_TARGET_CHARS)
        if len(chunks) <= 1:
            # Body is short or structurally unsplittable — no extra records.
            if rebuild_rows is None:
                self._prune_chunks(parent_id)
            return 0

        now = _now_iso()
        written = 0
        valid_chunk_ids: builtins.list[str] = []

        for chunk in chunks:
            seq = chunk["seq"]
            heading = chunk["heading"]
            chunk_body = chunk["body"]
            chunk_id = f"{parent_id}_chunk_{seq}"
            chunk_path = f"{parent_rel}#chunk-{seq}"
            valid_chunk_ids.append(chunk_id)

            chunk_title = (
                f"{title} § {heading}" if heading else f"{title} (§{seq + 1}/{len(chunks)})"
            )
            chunk_extra: dict[str, Any] = {
                "parent_id": parent_id,
                "chunk_index": seq,
                "chunk_seq": seq,
                "chunk_count": len(chunks),
                "chunk_heading": heading,
                "parent_path": parent_rel,
            }
            chunk_hash = _sha256_short(chunk_body)
            existing_chunk = None if rebuild_rows is not None else self.store.get(chunk_id)
            if (
                existing_chunk
                and existing_chunk["body_hash"] == chunk_hash
                and existing_chunk["path"] == chunk_path
                and existing_chunk["title"] == chunk_title
                and existing_chunk["tags"] == tags
                and (existing_chunk.get("extra") or {}) == chunk_extra
                and existing_chunk.get("valid_at") == valid_at
                and existing_chunk.get("invalid_at") == invalid_at
                and not force
            ):
                # Content and all parent-derived metadata are unchanged.
                continue

            emb = self._embed_cached(
                self._compose_for_embed(chunk_title, chunk_body),
                ctx=f"chunk {parent_id[:8]}#{seq}",
            )
            row = dict(
                id_=chunk_id,
                path=chunk_path,
                title=chunk_title,
                type_="reference",
                tags=tags,
                created=existing_chunk["created"] if existing_chunk else created,
                updated=now,
                body_hash=chunk_hash,
                embedding=emb,
                extra=chunk_extra,
                body_text=chunk_body,
                valid_at=valid_at,
                invalid_at=invalid_at,
            )
            if rebuild_rows is not None:
                rebuild_rows.append(row)
            else:
                self.store.upsert(**row)
                self.store.update_validity(
                    id_=chunk_id,
                    valid_at=valid_at,
                    invalid_at=invalid_at,
                )
            written += 1

        # Prune stale chunks from a previous reindex that had more chunks
        # (e.g. note was shortened or restructured into fewer sections).
        if rebuild_rows is None:
            self._prune_chunks(parent_id, valid_chunk_ids=valid_chunk_ids)

        return written

    def maybe_emit_chunks(
        self,
        *,
        parent_id: str,
        parent_rel: str,
        title: str,
        body: str,
        tags: builtins.list[str],
        created: str,
        updated: str,
        valid_at: str | None,
        invalid_at: str | None,
    ) -> int:
        """Best-effort chunk emission for one just-written memory.

        save()/update() call this so long documents get section-level
        retrieval immediately instead of waiting for the next manual
        reindex. Flag off → no-op. `_reindex_emit_chunks` handles every
        layout case (≤1 chunk prunes stale rows; unchanged sections are
        body_hash cache hits). Never raises: chunks are derived data and
        the next reindex heals — a write must not fail because of them.
        """
        if not flag_bool("MEMO_CHUNK_INGEST"):
            return 0
        try:
            return self._reindex_emit_chunks(
                parent_id=parent_id,
                parent_rel=parent_rel,
                title=title,
                body=body,
                tags=tags,
                created=created,
                updated=updated,
                valid_at=valid_at,
                invalid_at=invalid_at,
                force=False,
            )
        except Exception:
            _log.warning(
                "chunk emission failed for %s (next reindex heals)",
                parent_id[:8],
                exc_info=True,
            )
            return 0

    def _purge_version_history(self, id_: str) -> None:
        """Best-effort purge of derived version rows for a hard-deleted id.

        Mirrors delete_ops._delete_locked: version rows live in versions.db, not
        markdown, so a direct store.hard_delete (bypassing Memory.delete) would
        otherwise orphan them. Guarded on versions.db existing so a maintain pass
        never eagerly creates an empty db.
        """
        if (self.cfg.state_dir / "versions.db").is_file():
            with contextlib.suppress(Exception):
                self.versioning.version_store.delete_versions(id_)

    def _prune_chunks(
        self,
        parent_id: str,
        *,
        valid_chunk_ids: builtins.list[str] | None = None,
    ) -> None:
        """Hard-delete derived chunks not present in the parent's current layout."""
        keep = set(valid_chunk_ids or [])
        for row in self.store.chunks_by_parent_id(parent_id):
            if row["id"] in keep:
                continue
            self.store.hard_delete(row["id"])
            self._purge_version_history(row["id"])
            _log.debug("reindex: pruned stale chunk %s (parent %s)", row["id"][:12], parent_id[:8])

    def lint(self) -> dict[str, builtins.list[dict[str, Any]]]:
        """Surface memories with quality issues.

        Categories:
        - `legacy_extra`: has `extra` keys from mem-vault migration
          (`agent_id`, `last_used`, `usage_count`, `user_id`, `description`).
          These don't affect retrieval but bloat the frontmatter — worth
          a manual cleanup pass.
        - `few_tags`: <3 tags. The CLAUDE.md convention is ≥3 (project +
          domain + technique). Few tags hurt discovery via `memo top <tag>`.
        - `body_skinny`: body shorter than 100 chars. May still be useful
          for one-liner facts but worth checking if the user meant to
          write more.
        - `untitled`: title is literally "untitled" or matches the slug.

        Returns a dict of category → list of {id, title, reason} dicts.
        Pure read; never modifies the store.
        """
        legacy_keys = frozenset(
            {
                "agent_id",
                "last_used",
                "usage_count",
                "user_id",
                "description",
            }
        )
        out: dict[str, list[dict[str, Any]]] = {
            "legacy_extra": [],
            "few_tags": [],
            "body_skinny": [],
            "untitled": [],
        }
        for r in self.store.list_recent(limit=100_000):
            entry = {"id": r["id"], "title": r["title"]}
            extra = r.get("extra") or {}
            if any(k in extra for k in legacy_keys):
                out["legacy_extra"].append(
                    {
                        **entry,
                        "reason": "mem-vault legacy fields in extra: "
                        + ", ".join(sorted(set(extra) & legacy_keys)),
                    },
                )
            if len(r.get("tags") or []) < 3:
                out["few_tags"].append(
                    {**entry, "reason": f"only {len(r.get('tags') or [])} tag(s)"},
                )
            body = self._read_body(r["path"]) or ""
            if len(body.strip()) < 100:
                out["body_skinny"].append(
                    {**entry, "reason": f"body {len(body.strip())} chars"},
                )
            t = (r["title"] or "").strip().lower()
            if t == "untitled" or not t:
                out["untitled"].append({**entry, "reason": "title missing or 'untitled'"})
        return out

    # -- knowledge graph ----------------------------------------------------

    def extract_entities(
        self,
        *,
        ids: builtins.list[str] | None = None,
        all_: bool = False,
        skip_already_indexed: bool = True,
        max_batch: int | None = None,
    ) -> dict[str, int]:
        """Extract named entities from memories and write to the graph.

        Modes:
        - `ids=[...]`: process exactly the listed memory ids.
        - `all_=True`: process every memory in the store.

        With `skip_already_indexed=True` (default), memories that
        already have entries in `entity_memory` are skipped — useful
        for incremental runs after adding new memories. Pass False to
        force re-extraction (e.g. after improving the prompt).

        Returns counts: `{processed, entities_extracted, links_written, skipped, errors}`.
        Cost: ~0.5-1s per memory with the configured helper model.
        """
        if not all_ and not ids:
            raise ValueError("pass either ids=[...] or all_=True")

        if all_:
            target = [r["id"] for r in self.store.list_recent(limit=100_000)]
        else:
            target = list(ids or [])

        # Pre-filter already-indexed unless --force.
        if skip_already_indexed:
            target = [
                tid
                for tid in target
                if not self.graph.memory_extraction_provenance(tid).intersection(
                    {"explicit", "llm"}
                )
            ]

        if max_batch is not None:
            target = target[:max_batch]

        counts = {
            "processed": 0,
            "entities_extracted": 0,
            "links_written": 0,
            "skipped": 0,
            "errors": 0,
        }

        if not target:
            return counts

        chat = self._ensure_chat()

        for tid in target:
            r = self.store.get(tid)
            if r is None:
                counts["skipped"] += 1
                continue
            body = self._read_body(r["path"])
            if not body.strip():
                counts["skipped"] += 1
                continue
            # Build prompt: title + body excerpt. Cap to ~3000 chars to
            # keep the helper LLM cheap; entities tend to live in the
            # opening paragraphs.
            user_msg = (
                f"Title: {r['title']}\n"
                f"Tags: {', '.join(r['tags']) if r['tags'] else '—'}\n\n"
                f"{body[:3000]}"
            )
            try:
                out = chat_with_timeout(
                    chat,
                    timeout=30,
                    model=self.cfg.helper_model,
                    messages=[
                        {
                            "role": "system",
                            "content": resolve_prompt(
                                "extract_entities",
                                _EXTRACT_ENTITIES_SYSTEM_PROMPT,
                                self.cfg.state_dir,
                            ),
                        },
                        {"role": "user", "content": user_msg},
                    ],
                    options={"temperature": 0.0, "max_tokens": 384, "thinking": False},
                )
                if out is None:
                    _log.warning("extract_entities: LLM timeout for %s", tid[:8])
                    counts["errors"] += 1
                    continue
                text = ((out.get("message") or {}).get("content") or "").strip()
            except Exception as exc:
                _log.warning("extract_entities: LLM call failed for %s: %s", tid[:8], exc)
                counts["errors"] += 1
                continue
            text = strip_llm_output(text)
            try:
                data = json.loads(text) if text else {}
            except (ValueError, TypeError):
                counts["errors"] += 1
                continue
            ents = data.get("entities") if isinstance(data, dict) else None
            if not isinstance(ents, list):
                ents = []
            # Filter to dicts with both name + type fields.
            ents = [
                {"name": e.get("name"), "type": e.get("type")}
                for e in ents
                if isinstance(e, dict) and e.get("name") and e.get("type")
            ]
            n = self.graph.record_extraction(
                memory_id=tid,
                memory_date=r["created"][:10] if r.get("created") else _now_iso()[:10],
                entities=ents,
                extracted_at=_now_iso(),
                extractor="llm",
                extractor_version="helper-v1",
                confidence=0.85,
            )
            counts["processed"] += 1
            counts["entities_extracted"] += len(ents)
            counts["links_written"] += n
        return counts

    def _gc_store_orphans(self, *, fix: bool) -> builtins.list[str]:
        """Find store rows whose canonical source cannot be verified."""
        orphan_store: list[str] = []
        for row in self.store.list_recent(limit=100_000):
            extra = row.get("extra") or {}
            parent_id = extra.get("parent_id") if isinstance(extra, dict) else None
            ingest_abs = extra.get("abs_path") if isinstance(extra, dict) else None
            try:
                if parent_id:
                    parent = self.store.get(str(parent_id))
                    path_exists = (
                        parent is not None and self._resolve_existing(parent["path"]).is_file()
                    )
                elif ingest_abs:
                    # Ingested reference rows store label-prefixed paths that
                    # never resolve under memory_dir/vault — check the recorded
                    # source file instead of mass-deleting every labeled row.
                    path_exists = Path(str(ingest_abs)).is_file()
                else:
                    path_exists = self._resolve_existing(row["path"]).is_file()
                    if not path_exists and row.get("type") == "reference":
                        # Legacy ingest rows without abs_path provenance:
                        # existence can't be verified here — never mass-delete.
                        continue
            except StorageError:
                path_exists = False
            if not path_exists:
                orphan_store.append(row["id"])
                if fix:
                    if parent_id:
                        self.store.hard_delete(row["id"])
                        self._purge_version_history(row["id"])
                    else:
                        self.store.delete(row["id"])
        return orphan_store

    def _gc_disk_orphans(self) -> builtins.list[str]:
        """Find canonical Markdown ids absent from the derived store."""
        orphan_disk: list[str] = []
        if not self.cfg.memory_dir.is_dir():
            return orphan_disk
        for md_path in self.cfg.memory_dir.rglob("*.md"):
            _parts = md_path.relative_to(self.cfg.memory_dir).parts
            if _parts[:1] and _parts[0] in LIFECYCLE_ARCHIVE_DIRS:
                # Archived memories are intentionally out of the index — not
                # orphans, and "reindex to absorb" would resurrect them.  The
                # second name is the legacy vault convention.  Project buckets
                # never collide (project_bucket() remaps those two slugs).
                continue
            try:
                post = frontmatter.loads(md_path.read_text(encoding="utf-8"))
            except Exception as exc:
                _log.debug("gc: skipping %s (parse error): %s", md_path.name, exc)
                continue
            md_id = post.get("id")
            if isinstance(md_id, str) and md_id and self.store.get(md_id) is None:
                orphan_disk.append(str(md_path.relative_to(self.cfg.memory_dir)))
        return orphan_disk

    def _gc_stale_syntheses(self, *, fix: bool) -> builtins.list[str]:
        """Find synthesis records whose declared source has disappeared."""
        stale_synthesis: list[str] = []
        synth_rows = self.store._conn.execute(
            "SELECT meta.id, meta.path FROM meta WHERE meta.type = 'synthesis'",
        ).fetchall()
        for row in synth_rows:
            path = row["path"]
            if not path:
                continue
            source_path = self._resolve_existing(path)
            if not source_path.is_file():
                continue  # already caught by the orphan-store walk
            try:
                post = frontmatter.loads(source_path.read_text(encoding="utf-8"))
                extra: dict = post.get("extra") or {}  # type: ignore[assignment]
                source_ids = extra.get("synthesis_sources") or []
                for source_id in source_ids:
                    if self.store.get(source_id) is None:
                        stale_synthesis.append(row["id"])
                        if fix:
                            self.lifecycle.archive_memory(row["id"])
                        break
            except Exception as exc:
                _log.debug(
                    "gc: stale-synthesis check failed for %s: %s",
                    row["id"][:8],
                    exc,
                )
        return stale_synthesis

    def gc(self, *, fix: bool = False) -> dict[str, builtins.list[str]]:
        """Find orphans between the store and the memory dir.

        - `orphan_store`: store rows whose `.md` is missing on disk.
        - `orphan_disk`: `.md` files with an `id` frontmatter that the
          store doesn't know about. (Untagged `.md` files — no `id` —
          are ignored: they're user-authored content, not memories.)

        With `fix=True`, deletes orphan store rows. `.md` files are
        never deleted automatically — that's destructive and the user
        should review them first. Use `memo reindex` to absorb
        orphan disk files into the store.
        """
        return {
            "orphan_store": self._gc_store_orphans(fix=fix),
            "orphan_disk": self._gc_disk_orphans(),
            "stale_synthesis": self._gc_stale_syntheses(fix=fix),
        }

    def _transition_stale_memories(self, *, dry_run: bool = False) -> int:
        """Mark VERIFIED records STALE when their own review date passes.

        Loads only the decayable candidates from the store (a targeted query,
        not a full-corpus scan — most memories are UNVERIFIED and skipped) and
        persists each change via `update_verification` (meta only, no re-embed).
        Returns the count transitioned (or would-transition, on `dry_run`).
        Wired into `memo maintain` behind MEMO_VERIFICATION_STATE_TRACKING.
        """
        now_dt = datetime.now(UTC)
        transitioned = 0

        for row in self.store.verification_candidates():
            verified_at = row.get("verified_at")
            if not verified_at:
                continue
            state = row.get("verification_state")
            mem_id = row["id"]
            try:
                review_after = datetime.fromisoformat(
                    str(row.get("review_after") or "").replace("Z", "+00:00")
                )
                if review_after.tzinfo is None:
                    review_after = review_after.replace(tzinfo=UTC)
            except ValueError:
                review_after = None

            if (
                state == VerificationState.VERIFIED.value
                and review_after is not None
                and review_after <= now_dt
            ):
                transitioned += 1
                if not dry_run:
                    self._set_review_metadata(
                        mem_id,
                        review_after=str(row["review_after"]),
                        verification_state=VerificationState.STALE,
                        verified_at=int(verified_at),
                    )

        return transitioned
