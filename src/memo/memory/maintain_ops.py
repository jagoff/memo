"""Maintenance + provenance operations for `Memory`.

`_MaintainOpsMixin` holds the corpus-wide maintenance surface (reindex, lint,
gc, entity extraction), provenance lookups, and the synapse freeze-write guard.
Replay resolution moved to replay_ops.py; consolidation + synthesis to
consolidate_ops.py.
"""

from __future__ import annotations

import builtins
import json
import time
from typing import Any

import frontmatter

from memo.embedder import assert_valid_embedding
from memo.errors import StorageError
from memo.fact_extraction import upsert_declared_fact_edges
from memo.flags import flag_bool
from memo.lifecycle import FORGET_AFTER_KEY, FORGET_REASON_KEY
from memo.memory._base import _MemoryBase
from memo.memory.record import (
    _EXTRACT_ENTITIES_SYSTEM_PROMPT,
    _VALID_TYPES,
    WriteRefused,
    _build_freeze_query,
    _derive_title,
    _extract_provenance,
    _log,
    _normalise_tags,
    _now_iso,
    chat_with_timeout,
    strip_llm_output,
)
from memo.prompt_overrides import resolve_prompt
from memo.tiers import VerificationState
from memo.util import sha256_full as _sha256_full
from memo.util import sha256_short as _sha256_short


class _MaintainOpsMixin(_MemoryBase):
    def _enforce_synapse_freeze(
        self,
        *,
        title: str | None,
        content: str,
        tags: builtins.list[str] | None,
        trace_id: str,
    ) -> None:
        """Query synapse for blocking RealityConflicts; raise on hit.

        Derives a query from the most signal-dense fields available
        (title, first non-empty tags, first content line). Best-effort:
        if synapse is not on PATH, returns without raising — the
        opt-in nature already implies "best information available".
        """
        # Deferred import: keeps memo's hard deps free of synapse.
        from memo import synapse_client

        if not synapse_client.is_available():
            return
        query = _build_freeze_query(title=title, content=content, tags=tags)
        if not query:
            return
        # Fail-closed only when the env knob is explicitly set: an unreachable
        # synapse then refuses the write instead of silently disarming the
        # freeze gate. Without the knob (or when freeze was enabled only via the
        # per-save kwarg) we stay permissive — synapse outages mustn't block
        # memo's standalone writes. A missing binary is handled above and is
        # always permissive regardless of this flag.
        from memo.flags import flag_bool

        fail_closed = flag_bool("MEMO_RESPECT_SYNAPSE_FREEZE")
        try:
            conflicts = synapse_client.list_conflicts(
                query,
                trace_id=trace_id,
                strict=fail_closed,
            )
        except synapse_client.SynapseUnavailable as exc:
            if fail_closed:
                raise WriteRefused(
                    {
                        "conflict_id": "synapse-unreachable",
                        "summary": (
                            f"Synapse freeze-check could not complete ({exc}); "
                            f"refusing under MEMO_RESPECT_SYNAPSE_FREEZE=1"
                        ),
                        "freeze_write": True,
                        "lifecycle_state": "unknown",
                        "severity": "unknown",
                        "synapse_unreachable": True,
                    }
                ) from exc
            _log.debug("synapse freeze-check unavailable (permissive): %s", exc)
            return
        except Exception as exc:  # pragma: no cover - subprocess noise
            _log.debug("synapse freeze-check failed: %s", exc)
            return
        blocked, conflict = synapse_client.has_blocking_freeze(conflicts)
        if blocked and conflict is not None:
            raise WriteRefused(conflict)

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
              "current": {synapse_trace_id, synapse_route_reason, ...},
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
        model = self.cfg.embedder_model
        dims = self.cfg.embedder_dims
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
        """Scan the memory dir, re-embed entries whose on-disk body
        diverged from `body_hash`. Picks up edits the user made in
        Obsidian directly. Also indexes any `.md` with a valid `id` in
        frontmatter that the store doesn't know about (e.g. restored
        from a backup or copied from another machine).

        With `force=True`, re-embeds EVERY indexed entry regardless of
        body_hash match. Use after an embedder model swap, after a
        change to `_compose_for_embed`, or to refresh the index after
        a corruption/incident.

        With `rebuild=True`, first TRUNCATES the markdown-derivable tables
        (`meta`/`vec`/`fts`) and replays the whole index from disk — the
        markdown-is-truth reset. User-signal tables (`access`,
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
            # Wipe only the derivable tables; signal tables survive and re-join
            # on id. Every file then takes the `existing is None` add path.
            cleared = self.store.clear_memory_index()
            cleared_facts = self.fact_edges.clear()
            _log.info("reindex(rebuild): cleared %d derivable rows, replaying from disk", cleared)
            _log.info("reindex(rebuild): cleared %d temporal fact edges", cleared_facts)
            force = True

        for md_path in sorted(memory_root.rglob("*.md")):
            checked += 1
            try:
                post = frontmatter.loads(md_path.read_text(encoding="utf-8"))
            except Exception as exc:
                _log.warning("reindex: skipping %s (parse error): %s", md_path.name, exc)
                skipped += 1
                continue
            meta: dict[str, Any] = post.metadata
            md_id = meta.get("id")
            if not md_id or not isinstance(md_id, str):
                skipped += 1
                continue
            body = post.content or ""
            new_hash = _sha256_short(body)
            existing = self.store.get(md_id)
            # Path relative to memory_dir — paths in the store no longer
            # carry the legacy `<vault>/<memory_subdir>/...` prefix.
            rel = str(md_path.relative_to(self.cfg.memory_dir))

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
            # Obsidian-friendly: accept `forget_after` / `forget_reason` as
            # TOP-LEVEL frontmatter keys (what a user naturally types in their
            # editor), folding them into the extra bag the lifecycle layer
            # reads. The nested `extra:` form still works and takes precedence.
            for _fk in (FORGET_AFTER_KEY, FORGET_REASON_KEY):
                if _fk in meta and _fk not in extra:
                    extra = {**extra, _fk: meta[_fk]}

            # Extract verified_at timestamp (can be None)
            verified_at = meta.get("verified_at")
            if verified_at is not None and not isinstance(verified_at, int):
                try:
                    verified_at = int(verified_at)
                except (ValueError, TypeError):
                    verified_at = None

            if existing is None:
                # Path-collision guard: an .md may have its frontmatter id
                # regenerated (manual edit, restore-from-backup, or a stale
                # row pointing at a file whose id was rewritten) while the
                # vault-relative path stays the same. The store's
                # UNIQUE(meta.path) constraint blocks a plain INSERT, so we
                # drop the orphan row before re-adding under the new id.
                # include_deleted + hard_delete: a soft-deleted tombstone
                # still holds the path in the UNIQUE index (a soft delete
                # would leave it there and the INSERT would fail again) —
                # the disk file reclaims the path, so the tombstone is purged.
                stale = self.store.get_by_path(rel, include_deleted=True)
                if stale is not None:
                    _log.warning(
                        "reindex: path %r reused with new id (%s → %s); replacing stale row",
                        rel,
                        stale["id"][:8],
                        md_id[:8],
                    )
                    self.store.hard_delete(stale["id"])
                had_embed_pending = False
                if isinstance(extra, dict):
                    extra = dict(extra)
                    had_embed_pending = extra.pop("_memo_embed_pending", None) is not None
                try:
                    emb = self._embed_cached(
                        self._compose_for_embed(title, body), ctx=f"reindex add {md_id[:8]}"
                    )
                    self.store.upsert(
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
                    )
                except Exception as exc:
                    _log.warning("reindex: skipping %s (embed failed): %s", md_path.name, exc)
                    skipped += 1
                    continue
                if had_embed_pending:
                    try:
                        _post = frontmatter.loads(md_path.read_text(encoding="utf-8"))
                        _raw_extra = _post.metadata.get("extra")
                        _post_extra = dict(_raw_extra) if isinstance(_raw_extra, dict) else {}
                        _post_extra.pop("_memo_embed_pending", None)
                        _post.metadata["extra"] = _post_extra
                        md_path.write_text(frontmatter.dumps(_post), encoding="utf-8")
                    except Exception as _pend_exc:
                        _log.debug(
                            "reindex: could not clear _memo_embed_pending from %s: %s",
                            md_path.name,
                            _pend_exc,
                        )
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
                        force=force,
                    )
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
                continue
            missing_vector = not self.store.has_vector(md_id)
            if force or existing["body_hash"] != new_hash or missing_vector:
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
                        updated=_now_iso() if existing["body_hash"] != new_hash else updated,
                        body_hash=new_hash,
                        embedding=emb,
                        extra=extra if extra else None,
                        body_text=body,
                    )
                except Exception as exc:
                    _log.warning("reindex: skipping %s (re-embed failed): %s", md_path.name, exc)
                    skipped += 1
                    continue
                if had_embed_pending:
                    try:
                        _post = frontmatter.loads(md_path.read_text(encoding="utf-8"))
                        _raw_extra = _post.metadata.get("extra")
                        _post_extra = dict(_raw_extra) if isinstance(_raw_extra, dict) else {}
                        _post_extra.pop("_memo_embed_pending", None)
                        _post.metadata["extra"] = _post_extra
                        md_path.write_text(frontmatter.dumps(_post), encoding="utf-8")
                    except Exception as _pend_exc:
                        _log.debug(
                            "reindex: could not clear _memo_embed_pending from %s: %s",
                            md_path.name,
                            _pend_exc,
                        )
                reindexed += 1
                if chunk_ingest:
                    reindexed += self._reindex_emit_chunks(
                        parent_id=md_id,
                        parent_rel=rel,
                        title=title,
                        body=body,
                        tags=tags,
                        created=created,
                        updated=updated,
                        force=force,
                    )
            else:
                # Metadata-only frontmatter change (tags/type/extra changed,
                # body unchanged) — update meta without re-embedding.
                meta_changed = (
                    title != existing["title"]
                    or type_ != existing["type"]
                    or tags != existing["tags"]
                    or extra != (existing.get("extra") or {})
                )
                if meta_changed:
                    self.store.update_meta(
                        id_=md_id,
                        title=title,
                        type_=type_,
                        tags=tags,
                        updated=_now_iso(),
                        extra=extra if extra else None,
                    )
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
            from memo.receipts import emit_receipt

            emit_receipt(
                "reindex",
                text=(
                    f"Memo reindex: checked={checked} reindexed={reindexed} "
                    f"added={added} skipped={skipped} force={force}"
                ),
                meta={
                    "checked": checked,
                    "reindexed": reindexed,
                    "added": added,
                    "skipped": skipped,
                    "facts": facts,
                    "force": force,
                },
            )
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
        force: bool,
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

            chunk_hash = _sha256_short(chunk_body)
            existing_chunk = self.store.get(chunk_id)
            if existing_chunk and existing_chunk["body_hash"] == chunk_hash and not force:
                # Unchanged since last reindex — skip.
                continue

            chunk_title = (
                f"{title} § {heading}" if heading else f"{title} (§{seq + 1}/{len(chunks)})"
            )
            chunk_extra: dict[str, Any] = {
                "parent_id": parent_id,
                "chunk_index": seq,
                "chunk_count": len(chunks),
                "chunk_heading": heading,
                "parent_path": parent_rel,
            }
            emb = self._embed_cached(
                self._compose_for_embed(chunk_title, chunk_body),
                ctx=f"chunk {parent_id[:8]}#{seq}",
            )
            self.store.upsert(
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
            )
            written += 1

        # Prune stale chunks from a previous reindex that had more chunks
        # (e.g. note was shortened or restructured into fewer sections).
        for row in self.store.chunks_by_parent_id(parent_id):
            if row["id"] not in valid_chunk_ids:
                self.store.delete(row["id"])
                _log.debug(
                    "reindex: pruned stale chunk %s (parent %s)", row["id"][:12], parent_id[:8]
                )

        return written

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
            target = [tid for tid in target if not self.graph.memory_entities(tid)]

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
            )
            counts["processed"] += 1
            counts["entities_extracted"] += len(ents)
            counts["links_written"] += n
        return counts

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
        orphan_store: list[str] = []
        orphan_disk: list[str] = []

        # Store-side: walk meta, check file existence (with legacy fallback).
        for r in self.store.list_recent(limit=100_000):
            if not self._resolve_existing(r["path"]).is_file():
                orphan_store.append(r["id"])
                if fix:
                    self.store.delete(r["id"])

        # Disk-side: walk memory dir, check ids in store.
        if self.cfg.memory_dir.is_dir():
            for md_path in self.cfg.memory_dir.rglob("*.md"):
                try:
                    post = frontmatter.loads(md_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    _log.debug("gc: skipping %s (parse error): %s", md_path.name, exc)
                    continue
                md_id = post.get("id")
                if not md_id or not isinstance(md_id, str):
                    continue
                if self.store.get(md_id) is None:
                    orphan_disk.append(str(md_path.relative_to(self.cfg.memory_dir)))

        # Stale synthesis: type=synthesis memories where ≥1 source no longer
        # exists in the store. The synthesis is derived knowledge — if its
        # sources are gone the insight is unverifiable and should be archived.
        stale_synthesis: list[str] = []
        synth_rows = self.store._conn.execute(
            "SELECT meta.id, meta.path FROM meta WHERE meta.type = 'synthesis'",
        ).fetchall()
        for sr in synth_rows:
            p = sr["path"]
            if not p:
                continue
            sp = self._resolve_existing(p)
            if not sp.is_file():
                continue  # already caught above by orphan_store walk
            try:
                post = frontmatter.loads(sp.read_text(encoding="utf-8"))
                _extra: dict = post.get("extra") or {}  # type: ignore[assignment]
                source_ids = _extra.get("synthesis_sources") or []
                if not source_ids:
                    continue
                for sid in source_ids:
                    if self.store.get(sid) is None:
                        stale_synthesis.append(sr["id"])
                        if fix:
                            self.lifecycle.archive_memory(sr["id"])
                        break
            except Exception as exc:
                _log.debug("gc: stale-synthesis check failed for %s: %s", sr["id"][:8], exc)

        return {
            "orphan_store": orphan_store,
            "orphan_disk": orphan_disk,
            "stale_synthesis": stale_synthesis,
        }

    def _transition_stale_memories(
        self, *, stale_age_days: int = 30, unverify_age_days: int = 60
    ) -> int:
        """Auto-transition verified memories to stale, and stale to unverified.

        VERIFIED memories older than stale_age_days transition to STALE.
        STALE memories older than unverify_age_days transition to UNVERIFIED.
        Updates memory_map and the store, but does NOT re-embed.

        Returns the count of transitioned memories.
        """
        now = int(time.time())
        transitioned = 0

        # Iterate through memory_map (populated by maintain pipeline)
        if not hasattr(self, "memory_map"):
            return 0

        for mem_id, mem in list(self.memory_map.items()):
            if not mem.verified_at:
                continue

            days_old = (now - mem.verified_at) / 86400.0

            if mem.verification_state == VerificationState.VERIFIED and days_old > stale_age_days:
                # Transition VERIFIED → STALE
                from dataclasses import replace as dataclass_replace

                mem_updated = dataclass_replace(
                    mem,
                    verification_state=VerificationState.STALE,
                )
                # Update store meta without re-embedding
                self.store.update_meta(
                    id_=mem_id,
                    title=mem_updated.title,
                    type_=mem_updated.type,
                    tags=mem_updated.tags,
                    updated=mem_updated.updated,
                    extra=dict(mem_updated.extra) if mem_updated.extra else {},
                )
                # Update verification_state and verified_at directly in store
                self.store._conn.execute(
                    "UPDATE meta SET verification_state = ?, verified_at = ? WHERE id = ?",
                    (mem_updated.verification_state.value, mem_updated.verified_at, mem_id),
                )
                self.memory_map[mem_id] = mem_updated
                transitioned += 1
            elif mem.verification_state == VerificationState.STALE and days_old > unverify_age_days:
                # Transition STALE → UNVERIFIED
                from dataclasses import replace as dataclass_replace

                mem_updated = dataclass_replace(
                    mem,
                    verification_state=VerificationState.UNVERIFIED,
                    verified_at=None,
                )
                # Update store meta without re-embedding
                self.store.update_meta(
                    id_=mem_id,
                    title=mem_updated.title,
                    type_=mem_updated.type,
                    tags=mem_updated.tags,
                    updated=mem_updated.updated,
                    extra=dict(mem_updated.extra) if mem_updated.extra else {},
                )
                # Update verification_state and verified_at directly in store
                self.store._conn.execute(
                    "UPDATE meta SET verification_state = ?, verified_at = ? WHERE id = ?",
                    (mem_updated.verification_state.value, mem_updated.verified_at, mem_id),
                )
                self.memory_map[mem_id] = mem_updated
                transitioned += 1

        return transitioned
