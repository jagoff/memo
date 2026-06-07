"""Maintenance + provenance operations for `Memory`.

`_MaintainOpsMixin` holds the corpus-wide maintenance surface (reindex, lint,
gc, entity extraction), provenance lookups, and the synapse freeze-write guard.
Replay resolution moved to replay_ops.py; consolidation + synthesis to
consolidate_ops.py.
"""

from __future__ import annotations

import builtins
import json
from typing import Any

import frontmatter

from memo.embedder import assert_valid_embedding
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
        try:
            conflicts = synapse_client.list_conflicts(
                query, trace_id=trace_id,
            )
        except Exception as exc:  # pragma: no cover - subprocess noise
            _log.debug("synapse freeze-check failed: %s", exc)
            return
        blocked, conflict = synapse_client.has_blocking_freeze(conflicts)
        if blocked and conflict is not None:
            raise WriteRefused(conflict)

    # -- provenance ---------------------------------------------------------


    def provenance(self, id_: str) -> dict[str, Any] | None:
        """Return the full provenance trail for a memoria.

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


    def reindex(self, *, force: bool = False) -> dict[str, int]:
        """Scan the memory dir, re-embed entries whose on-disk body
        diverged from `body_hash`. Picks up edits the user made in
        Obsidian directly. Also indexes any `.md` with a valid `id` in
        frontmatter that the store doesn't know about (e.g. restored
        from a backup or copied from another machine).

        With `force=True`, re-embeds EVERY indexed entry regardless of
        body_hash match. Use after an embedder model swap, after a
        change to `_compose_for_embed`, or to refresh the index after
        a corruption/incident.

        Returns counts: `{"checked", "reindexed", "added", "skipped"}`.
        """
        memory_root = self.cfg.memory_dir
        checked = reindexed = added = skipped = 0
        if not memory_root.is_dir():
            return {"checked": 0, "reindexed": 0, "added": 0, "skipped": 0}

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
                _log.warning("reindex: invalid type %r in %s, coercing to 'note'", type_, md_path.name)
                type_ = "note"
            tags = _normalise_tags(list(meta.get("tags") or []))
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

            if existing is None:
                # Path-collision guard: an .md may have its frontmatter id
                # regenerated (manual edit, restore-from-backup, or a stale
                # row pointing at a file whose id was rewritten) while the
                # vault-relative path stays the same. The store's
                # UNIQUE(meta.path) constraint blocks a plain INSERT, so we
                # drop the orphan row before re-adding under the new id.
                stale = self.store.get_by_path(rel)
                if stale is not None:
                    _log.warning(
                        "reindex: path %r reused with new id (%s → %s); "
                        "replacing stale row",
                        rel, stale["id"][:8], md_id[:8],
                    )
                    self.store.delete(stale["id"])
                emb = self.embedder.embed([self._compose_for_embed(title, body)])[0]
                assert_valid_embedding(emb, self.cfg.embedder_dims, context=f"reindex add {md_id[:8]}")
                self.store.upsert(
                    id_=md_id, path=rel, title=title, type_=type_, tags=tags,
                    created=created, updated=updated, body_hash=new_hash,
                    embedding=emb, extra=extra if extra else None,
                    body_text=body,
                )
                added += 1
                continue
            missing_vector = not self.store.has_vector(md_id)
            if force or existing["body_hash"] != new_hash or missing_vector:
                if isinstance(extra, dict):
                    extra = dict(extra)
                    extra.pop("_memo_embed_pending", None)
                emb = self.embedder.embed([self._compose_for_embed(title, body)])[0]
                assert_valid_embedding(emb, self.cfg.embedder_dims, context=f"reindex update {md_id[:8]}")
                self.store.upsert(
                    id_=md_id, path=rel, title=title, type_=type_, tags=tags,
                    created=existing["created"], updated=_now_iso(),
                    body_hash=new_hash, embedding=emb,
                    extra=extra if extra else None,
                    body_text=body,
                )
                reindexed += 1
        # Successful reindex: every meta.path now uses the current
        # memory_dir-relative layout, so future startups can skip the
        # legacy-path probe in `_maybe_warn_legacy_paths`.
        self.store.set_user_version(1)
        counts = {"checked": checked, "reindexed": reindexed, "added": added, "skipped": skipped}
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
                    "force": force,
                },
            )
        return counts


    def lint(self) -> dict[str, builtins.list[dict[str, Any]]]:
        """Surface memorias with quality issues.

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
        legacy_keys = frozenset({
            "agent_id", "last_used", "usage_count", "user_id", "description",
        })
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
                    {**entry, "reason": "mem-vault legacy fields in extra: "
                                        + ", ".join(sorted(set(extra) & legacy_keys))},
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
        self, *, ids: builtins.list[str] | None = None, all_: bool = False,
        skip_already_indexed: bool = True,
        max_batch: int | None = None,
    ) -> dict[str, int]:
        """Extract named entities from memorias and write to the graph.

        Modes:
        - `ids=[...]`: process exactly the listed memoria ids.
        - `all_=True`: process every memoria in the store.

        With `skip_already_indexed=True` (default), memorias that
        already have entries in `entity_memoria` are skipped — useful
        for incremental runs after adding new memorias. Pass False to
        force re-extraction (e.g. after improving the prompt).

        Returns counts: `{processed, entities_extracted, links_written, skipped, errors}`.
        Cost: ~0.5-1s per memoria with Qwen2.5-3B. 223 memorias ≈ 2-4 min.
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
                tid for tid in target
                if not self.graph.memoria_entities(tid)
            ]

        if max_batch is not None:
            target = target[:max_batch]

        counts = {"processed": 0, "entities_extracted": 0,
                  "links_written": 0, "skipped": 0, "errors": 0}

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
                    chat, timeout=30,
                    model=self.cfg.helper_model,
                    messages=[
                        {"role": "system", "content": _EXTRACT_ENTITIES_SYSTEM_PROMPT},
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
                memoria_id=tid,
                memoria_date=r["created"][:10] if r.get("created") else _now_iso()[:10],
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
                            self.lifecycle.archive_memoria(sr["id"])
                        break
            except Exception as exc:
                _log.debug("gc: stale-synthesis check failed for %s: %s", sr["id"][:8], exc)

        return {
            "orphan_store": orphan_store,
            "orphan_disk": orphan_disk,
            "stale_synthesis": stale_synthesis,
        }
