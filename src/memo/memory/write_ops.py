"""Write-path operations for `Memory` — save / update / forget / delete.

`_WriteOpsMixin` holds the methods that mutate the vault + index (and their
private helpers), moved verbatim from the former `memory.py` god-file. The
real attributes/managers live on the `Memory` facade; this mixin only
declares the typed contract via `_MemoryBase`.
"""

from __future__ import annotations

import builtins
import contextlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import frontmatter

from memo.embedder import assert_valid_embedding
from memo.lifecycle import (
    FORGET_AFTER_KEY,
    FORGET_REASON_KEY,
    IS_FORGOTTEN_KEY,
)
from memo.llm import MLXChat
from memo.memory._base import _MemoryBase
from memo.memory.record import (
    _DERIVE_SYSTEM_PROMPT,
    _VALID_TYPES,
    MemoryRecord,
    _derive_title,
    _extract_provenance,
    _log,
    _normalise_tags,
    _now_iso,
    _slugify,
)
from memo.util import sha256_short as _sha256_short


class _WriteOpsMixin(_MemoryBase):
    # -- save ---------------------------------------------------------------

    def _derive_metadata(self, content: str) -> dict[str, Any]:
        """Use the helper LLM (Qwen2.5-3B-Instruct-4bit) to derive
        {title, type, tags} from raw content. Returns a dict with
        whatever keys the model produced (any can be None on parse
        failure). Caller decides whether to fill missing fields.

        Failure modes are absorbed: a bad LLM response yields an empty
        dict and the caller falls back to heuristics. We never propagate
        an LLM error up to a save() call — the save must succeed even
        if the helper is broken.
        """
        if self._chat is None:
            self._chat = MLXChat()
        try:
            out = self._chat.chat(
                model=self.cfg.helper_model,
                messages=[
                    {"role": "system", "content": _DERIVE_SYSTEM_PROMPT},
                    # Cap input to keep the prompt cheap. The helper only
                    # needs the gist, not the full body.
                    {"role": "user", "content": content[:2000]},
                ],
                options={"temperature": 0.0, "max_tokens": 256},
            )
            text = (out.get("message") or {}).get("content") or ""
        except Exception as exc:
            _log.warning("_derive_metadata LLM call failed: %s", exc)
            return {}
        # Tolerate markdown code fences even though the prompt forbids them.
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
        try:
            data = json.loads(text)
        except Exception as exc:
            _log.warning("_derive_metadata JSON parse failed (%r…): %s", text[:80], exc)
            return {}
        if not isinstance(data, dict):
            return {}
        derived: dict[str, Any] = {}
        t_title = (data.get("title") or "")
        if isinstance(t_title, str) and t_title.strip():
            derived["title"] = t_title.strip()[:80]
        t_type = data.get("type")
        if isinstance(t_type, str) and t_type in _VALID_TYPES:
            derived["type"] = t_type
        t_tags = data.get("tags") or []
        if isinstance(t_tags, list):
            derived["tags"] = _normalise_tags([t for t in t_tags if isinstance(t, str)])
        return derived

    def save(
        self,
        *,
        content: str,
        title: str | None = None,
        type_: str = "note",
        type: str | None = None,
        tags: list[str] | None = None,
        extra: dict[str, Any] | None = None,
        auto_derive: bool = False,
        auto_project: bool = True,
        cwd: str | None = None,
        created: str | None = None,
        defer_embed: bool = False,
        respect_synapse_freeze: bool | None = None,
        skip_memflow_receipt: bool = False,
    ) -> MemoryRecord:
        """Persist a memory to disk + index.

        - `content`: free-form markdown body (no frontmatter; we add it).
        - `title`: optional. If omitted, derived from the first line of
          content (truncated, slug-safe).
        - `type_`: must be in `_VALID_TYPES`. `type` is accepted as a
          compatibility alias. `note` is the default neutral value.
        - `tags`: optional list. Lower-cased + de-duplicated.
        - `extra`: arbitrary JSON-serialisable metadata bag.
        - `created`: optional ISO8601 override for the `created` field
          (frontmatter + index). When None, defaults to NOW. `updated`
          always reflects this write. Use to back-date imported records
          (e.g. historical WhatsApp messages) so temporal queries see the
          original event time, not ingest time.
        - `auto_derive`: when True, calls the helper LLM
          (`Qwen2.5-3B-Instruct-4bit`) to fill any missing field
          (title is None, type_ is "note" with no tags). Adds ~1-2s
          latency on first call (cold model load) plus ~0.5-1s per save.
          Use for callers (eg. another agent) that don't carry context
          to derive metadata themselves.
        - `defer_embed`: when True, write markdown + metadata + BM25
          index only. Semantic search won't see the record until
          `memo reindex` runs with the embedder available.
        - `respect_synapse_freeze`: when True, query synapse's
          `RealityConflict` ledger before commit and raise
          `WriteRefused` if a blocking freeze-write covers this
          memoria's topic. Defaults to the env knob
          `MEMO_RESPECT_SYNAPSE_FREEZE=1` (opt-in). Only fires when
          `extra` carries a `synapse_trace_id` — anonymous saves
          bypass the check.
        """
        if not content or not content.strip():
            raise ValueError("`content` must be non-empty")

        # Auto-attach SYNAPSE_TRACE_ID from env when the caller did not
        # carry an explicit trace_id in `extra`. Lets provenance walks
        # link memo writes back to the synapse session that spawned the
        # subprocess, even for direct `memo save` CLI invocations.
        env_trace = os.environ.get("SYNAPSE_TRACE_ID", "").strip()
        if env_trace and (extra is None or not extra.get("synapse_trace_id")):
            extra = dict(extra or {})
            extra["synapse_trace_id"] = env_trace

        # Freeze-write protocol: opt-in pre-write check against synapse.
        # Only fires when (a) the caller asked (kwarg or env), (b) the
        # save carries provenance (otherwise we have no agent context
        # to reason about), and (c) synapse is on PATH.
        if respect_synapse_freeze is None:
            respect_synapse_freeze = (
                os.environ.get("MEMO_RESPECT_SYNAPSE_FREEZE") == "1"
            )
        if respect_synapse_freeze and extra and extra.get("synapse_trace_id"):
            self._enforce_synapse_freeze(
                title=title, content=content, tags=tags,
                trace_id=str(extra.get("synapse_trace_id") or ""),
            )
        if type is not None:
            if type_ != "note" and type_ != type:
                raise ValueError("Pass either `type_` or `type`, not conflicting values")
            type_ = type
        if type_ not in _VALID_TYPES:
            raise ValueError(
                f"`type_={type_!r}` not in valid set {sorted(_VALID_TYPES)}",
            )

        if auto_derive:
            # Only fire the LLM if at least one field looks "default-y".
            # User-provided values always win.
            wants_title = title is None
            wants_type = type_ == "note"
            wants_tags = not tags
            if wants_title or wants_type or wants_tags:
                derived = self._derive_metadata(content)
                if wants_title and derived.get("title"):
                    title = derived["title"]
                if wants_type and derived.get("type"):
                    type_ = derived["type"]
                if wants_tags and derived.get("tags"):
                    tags = derived["tags"]

        title = (title or _derive_title(content)).strip()
        if not title:
            title = "untitled"

        norm_tags = _normalise_tags(tags or [])

        # Auto-tag with the caller's project (git toplevel basename or
        # MEMO_PROJECT_TAG) so per-repo recall can boost the right
        # memorias. Skipped when the caller already passed any
        # `project:` tag — explicit always wins.
        if auto_project and os.environ.get("MEMO_AUTO_PROJECT_TAG", "1") == "1":
            try:
                from memo.project import current_project_tag, has_project_tag
                if not has_project_tag(norm_tags):
                    pt = current_project_tag(cwd)
                    if pt:
                        norm_tags = _normalise_tags([*norm_tags, pt])
            except Exception as exc:
                _log.warning("auto-project tag failed (cwd=%s): %s", cwd, exc)

        now_iso = _now_iso()
        created_iso = created or now_iso
        # Truncate content for embedding (vec store doesn't truncate;
        # disk file keeps full content). 64KB is the default cap.
        if len(content) > self.cfg.max_content_chars:
            _log.warning(
                "save: content truncated from %d to %d chars (title=%r)",
                len(content), self.cfg.max_content_chars, title,
            )
        content = content[: self.cfg.max_content_chars]

        record_id = uuid.uuid4().hex
        rel_path = self._build_rel_path(title, now_iso)
        body_hash = _sha256_short(content)

        # Write `.md` first — if anything fails after this, the user
        # can recover by re-indexing. Conversely if we write the index
        # first and the disk write fails, the index points to a
        # non-existent file.
        abs_path = self.cfg.memory_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        extra_for_store = dict(extra or {})
        if defer_embed:
            extra_for_store["_memo_embed_pending"] = True

        post = frontmatter.Post(
            content,
            id=record_id,
            title=title,
            type=type_,
            tags=norm_tags,
            created=created_iso,
            updated=now_iso,
        )
        if extra_for_store:
            post["extra"] = extra_for_store
        abs_path.write_text(frontmatter.dumps(post), encoding="utf-8")

        if defer_embed:
            self.store.upsert_text_only(
                id_=record_id,
                path=rel_path,
                title=title,
                type_=type_,
                tags=norm_tags,
                created=created_iso,
                updated=now_iso,
                body_hash=body_hash,
                extra=extra_for_store,
                body_text=content,
            )
            self.history.log_save(
                ts=now_iso, record_id=record_id, title=title, type_=type_,
                provenance=_extract_provenance(extra_for_store),
            )
            deferred_rec = MemoryRecord(
                id=record_id, path=rel_path, title=title, type=type_, tags=norm_tags,
                created=created_iso, updated=now_iso, body=content, extra=extra_for_store,
            )
            self._emit_save_receipt(
                deferred_rec, deferred=True, disabled=skip_memflow_receipt,
            )
            return deferred_rec

        # Embed `title + body`: the title carries the highest-density
        # signal for retrieval ("Astor — Informe TO" is a much better
        # match for a query like "informe terapia ocupacional astor"
        # than the body's clinical paragraphs alone). Prepending also
        # protects the title from head-truncation when the body is
        # long — see embedder.py for the truncation rationale.
        embedding = self.embedder.embed([self._compose_for_embed(title, content)])[0]
        assert_valid_embedding(
            embedding, self.cfg.embedder_dims, context=f"save id={record_id[:8]}",
        )

        self.store.upsert(
            id_=record_id,
            path=rel_path,
            title=title,
            type_=type_,
            tags=norm_tags,
            created=created_iso,
            updated=now_iso,
            body_hash=body_hash,
            embedding=embedding,
            extra=extra,
            body_text=content,
        )

        self.history.log_save(
            ts=now_iso, record_id=record_id, title=title, type_=type_,
            provenance=_extract_provenance(extra),
        )

        rec = MemoryRecord(
            id=record_id, path=rel_path, title=title, type=type_, tags=norm_tags,
            created=created_iso, updated=now_iso, body=content, extra=extra or {},
        )
        self._emit_save_receipt(rec, deferred=False, disabled=skip_memflow_receipt)
        # Cache-tier: write policy (push/dirty) then capacity bound. Both
        # no-op unless MEMO_CACHE_MODE is on. Guarded so a backend hiccup
        # never fails the save itself.
        self._apply_write_policy(rec)
        if self.cache.policy.enabled and self.cache.policy.max_entries > 0:
            try:
                self.cache.evict_if_needed()
            except Exception as exc:
                _log.warning("cache eviction skipped after save: %s", exc)
        return rec

    def _emit_save_receipt(
        self, rec: MemoryRecord, *, deferred: bool, disabled: bool,
    ) -> None:
        """Fire-and-forget memflow receipt for a successful save.

        No-op unless `MEMO_EMIT_RECEIPTS=1`; never raises. `disabled`
        is the synapse-originated opt-out (synapse keeps its own ledger).
        """
        from memo.receipts import emit_receipt

        prov = _extract_provenance(rec.extra or {})
        emit_receipt(
            "save",
            text=f"Memo saved memoria {rec.id[:8]} ({rec.type}): {rec.title}",
            meta={
                "id": rec.id,
                "type": rec.type,
                "tags": ",".join(rec.tags),
                "path": rec.path,
                "deferred": deferred,
                "synapse_trace_id": prov.get("synapse_trace_id", ""),
                "synapse_route_reason": prov.get("synapse_route_reason", ""),
                "synapse_agent_id": prov.get("synapse_agent_id", ""),
            },
            disabled=disabled,
        )
        # M2b: also emit to the unified trinity ledger (best-effort,
        # independent of the memflow receipt path).
        self._emit_ledger("save", rec, prov, deferred=deferred)

    def _emit_ledger(
        self,
        op: str,
        rec: MemoryRecord,
        prov: dict[str, Any] | None = None,
        *,
        deferred: bool = False,
    ) -> None:
        """Fire-and-forget ConsciousnessEvent for the unified ledger (M2)."""
        from memo.consciousness_ledger import emit_event

        emit_event(
            op,
            subject_uri=f"memo://memoria/{rec.id}",
            trace_id=(prov or {}).get("synapse_trace_id", "") or "",
            actor=(prov or {}).get("synapse_agent_id", "") or "memo",
            payload={
                "id": rec.id,
                "type": rec.type,
                "title": rec.title or "",
                "tags": list(rec.tags or []),
                "deferred": deferred,
            },
        )

    def _apply_write_policy(self, rec: MemoryRecord) -> None:
        """Honor MEMO_CACHE_MODE on a fresh save:

        - write_through: push to the backing store now; if the push fails,
          mark the entry dirty so a later flush retries (no lost write).
        - write_back: mark dirty; the push happens on flush / before eviction.
        - read_through / off: nothing (the local store is authoritative).
        Guarded — a backend failure never fails the save.
        """
        policy = self.cache.policy
        if not policy.enabled:
            return
        # A read-through fill already mirrors the backing store — pushing it
        # back (write_through) or marking it dirty (write_back) would be a
        # redundant round-trip, so skip the write policy for those saves.
        if (rec.extra or {}).get("source") == "memo-cache-fill":
            return
        try:
            if policy.write_through:
                backend = self._cache_backend()
                if backend is None or not backend.push(rec):
                    self._mark_dirty(rec.id)
            elif policy.write_back:
                self._mark_dirty(rec.id)
        except Exception as exc:
            _log.warning("cache write policy skipped for %s: %s", rec.id[:8], exc)

    # -- update -------------------------------------------------------------

    def update(
        self,
        id_: str,
        *,
        title: str | None = None,
        type_: str | None = None,
        tags: builtins.list[str] | None = None,
        content: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> MemoryRecord | None:
        """Patch one or more fields on an existing record.

        Only the kwargs you pass are touched; everything else stays as-is.
        Re-embed only if `content` changed (body_hash check). The file
        path stays stable — renaming the slug after the fact would break
        wikilinks the user may have created in their vault.
        """
        resolved = self.resolve_id(id_)
        if resolved is None:
            return None
        id_ = resolved
        r = self.store.get(id_)
        if r is None:
            return None
        if type_ is not None and type_ not in _VALID_TYPES:
            raise ValueError(
                f"`type_={type_!r}` not in valid set {sorted(_VALID_TYPES)}",
            )

        new_title = (title.strip() if title else r["title"]) or r["title"]
        new_type = type_ or r["type"]
        new_tags = _normalise_tags(tags) if tags is not None else r["tags"]
        new_extra = extra if extra is not None else r.get("extra") or {}
        now_iso = _now_iso()

        # Body resolution: provided > on-disk > empty.
        old_body = self._read_body(r["path"])
        new_body = (content if content is not None else old_body)
        new_body = new_body[: self.cfg.max_content_chars]
        new_body_hash = _sha256_short(new_body)
        body_changed = new_body_hash != r["body_hash"]
        title_changed = new_title != r["title"]

        abs_path = self._resolve_existing(r["path"])
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        post = frontmatter.Post(
            new_body,
            id=id_,
            title=new_title,
            type=new_type,
            tags=new_tags,
            created=r["created"],
            updated=now_iso,
        )
        if new_extra:
            post["extra"] = new_extra
        abs_path.write_text(frontmatter.dumps(post), encoding="utf-8")

        # Re-embed when the body OR title changed — both are part of the
        # embed input now (see `_compose_for_embed`). Pure retag/type
        # changes still skip the embedder.
        if body_changed or title_changed:
            embedding = self.embedder.embed([self._compose_for_embed(new_title, new_body)])[0]
            assert_valid_embedding(embedding, self.cfg.embedder_dims, context=f"update id={id_[:8]}")
            self.store.upsert(
                id_=id_, path=r["path"], title=new_title, type_=new_type,
                tags=new_tags, created=r["created"], updated=now_iso,
                body_hash=new_body_hash, embedding=embedding, extra=new_extra,
                body_text=new_body,
            )
        else:
            self.store.update_meta(
                id_=id_, title=new_title, type_=new_type, tags=new_tags,
                updated=now_iso, extra=new_extra,
            )

        # Audit log: build a delta of just the fields that changed.
        delta: dict[str, tuple[Any, Any]] = {}
        if title_changed:
            delta["title"] = (r["title"], new_title)
        if new_type != r["type"]:
            delta["type"] = (r["type"], new_type)
        if new_tags != r["tags"]:
            delta["tags"] = (r["tags"], new_tags)
        if body_changed:
            delta["body_hash"] = (r["body_hash"], new_body_hash)
        # Track provenance churn so a re-route (e.g. Synapse re-issues a
        # different trace_id on the same memoria) shows up in history.
        old_prov = _extract_provenance(r.get("extra") or {})
        new_prov = _extract_provenance(new_extra)
        if old_prov != new_prov:
            delta["_provenance"] = (old_prov, new_prov)
        if delta:
            self.history.log_update(
                ts=now_iso, record_id=id_, title=new_title, type_=new_type,
                delta=delta,
            )

        updated_rec = MemoryRecord(
            id=id_, path=r["path"], title=new_title, type=new_type,
            tags=new_tags, created=r["created"], updated=now_iso,
            body=new_body, extra=new_extra,
        )
        if delta:
            from memo.receipts import emit_receipt

            emit_receipt(
                "update",
                text=f"Memo updated memoria {id_[:8]}: {', '.join(sorted(delta.keys()))}",
                meta={
                    "id": id_,
                    "type": new_type,
                    "title": new_title,
                    "delta_keys": ",".join(sorted(delta.keys())),
                },
            )
            # M2b: also emit to the unified trinity ledger.
            self._emit_ledger("update", updated_rec, new_prov)
        return updated_rec

    # -- forget (soft, reversible) ------------------------------------------

    def forget(self, id_: str, *, reason: str | None = None) -> MemoryRecord | None:
        """Soft-forget a memoria: keep the file + index, but exclude it from
        `search` / recall / `list` by default.

        Distinct from `delete` (which removes file + index) — `forget` is
        reversible via `unforget`. Sets `is_forgotten` (and an optional
        `forget_reason`) in the `extra` bag, merging onto existing metadata so
        provenance and other keys survive. Returns the updated record, or None
        if the id is unknown.
        """
        resolved = self.resolve_id(id_)
        if resolved is None:
            return None
        r = self.store.get(resolved)
        if r is None:
            return None
        merged = dict(r.get("extra") or {})
        merged[IS_FORGOTTEN_KEY] = True
        if reason:
            merged[FORGET_REASON_KEY] = reason
        return self.update(resolved, extra=merged)

    def unforget(self, id_: str) -> MemoryRecord | None:
        """Reverse a `forget`: clear `is_forgotten` so the memoria is searchable
        again. Also clears `forget_after` / `forget_reason` so the next
        maintenance pass doesn't immediately re-forget it. No-op (returns the
        record) if it wasn't forgotten.
        """
        resolved = self.resolve_id(id_)
        if resolved is None:
            return None
        r = self.store.get(resolved)
        if r is None:
            return None
        merged = dict(r.get("extra") or {})
        for key in (IS_FORGOTTEN_KEY, FORGET_AFTER_KEY, FORGET_REASON_KEY):
            merged.pop(key, None)
        return self.update(resolved, extra=merged)

    # -- delete -------------------------------------------------------------

    def delete(self, id_: str) -> bool:
        """Remove from store + disk. Returns True if anything was deleted."""
        resolved = self.resolve_id(id_)
        if resolved is None:
            return False
        id_ = resolved
        r = self.store.get(id_)
        if not r:
            return False
        existed = self.store.delete(id_)
        # File deletion is best-effort; the store is the authoritative delete signal.
        with contextlib.suppress(OSError):
            self._resolve_existing(r["path"]).unlink(missing_ok=True)
        if existed:
            self.history.log_delete(
                ts=_now_iso(), record_id=id_, title=r["title"], type_=r["type"],
            )
            # Drop graph edges for this memoria so entity counts stay
            # honest. Cheap (one DELETE + counter decrement per edge).
            self.graph.drop_for_memoria(id_)
            # Drop dangling contradiction pairs touching this memoria.
            # Only walks if the sidecar was already opened, so callers
            # that never used the radar pay nothing.
            if self._contradict_store is not None:
                with contextlib.suppress(Exception):
                    self._contradict_store.drop_for_memoria(id_)
            from memo.receipts import emit_receipt

            emit_receipt(
                "delete",
                text=f"Memo deleted memoria {id_[:8]} ({r['type']}): {r['title']}",
                meta={
                    "id": id_,
                    "type": r["type"],
                    "title": r["title"],
                    "path": r["path"],
                },
            )
            # M2b: also emit to the unified trinity ledger.
            from memo.consciousness_ledger import emit_event

            emit_event(
                "delete",
                subject_uri=f"memo://memoria/{id_}",
                trace_id=(_extract_provenance(r.get("extra") or {}) or {}).get("synapse_trace_id", ""),
                actor="memo",
                payload={
                    "id": id_,
                    "type": r["type"],
                    "title": r["title"],
                    "path": r["path"],
                },
            )
        return existed

    # -- internals ----------------------------------------------------------

    def _build_rel_path(self, title: str, now_iso: str) -> str:
        date = now_iso.split("T", 1)[0]
        slug = _slugify(title)[:80] or "untitled"
        # POSIX path joins. Path is relative to `cfg.memory_dir`.
        base = f"{date}-{slug}"
        candidate = f"{base}.md"
        # `meta.path` is UNIQUE. Two saves with the same title on the same day
        # (e.g. several WhatsApp chunks from one chat on one date) would collide.
        # Append a numeric suffix until the path is free — checking both the
        # index and the on-disk file so a deferred/unindexed write still counts.
        n = 2
        while (
            self.store.get_by_path(candidate) is not None
            or (self.cfg.memory_dir / candidate).exists()
        ):
            candidate = f"{base}-{n}.md"
            n += 1
        return candidate

    def _resolve_existing(self, rel_path: str) -> Path:
        """Resolve a DB-stored path to an absolute `Path`.

        Tries `memory_dir / rel_path` first (current layout). Falls back
        to `vault_path / rel_path` if `vault_path` is set AND the file
        actually exists there (legacy layout: paths in older DB rows
        carry a `<memory_subdir>/...` prefix relative to `vault_path`).

        Returns the new-layout path even when the file doesn't exist on
        either branch — callers that need to CREATE a file always write
        to the new layout.
        """
        new_path = self.cfg.memory_dir / rel_path
        if new_path.is_file():
            return new_path
        if self.cfg.vault_path is not None:
            legacy = self.cfg.vault_path / rel_path
            if legacy.is_file():
                return legacy
        return new_path

    def _read_body(self, rel_path: str) -> str:
        # Fast path: curated memorias live on disk under memory_dir / vault_path.
        abs_path = self._resolve_existing(rel_path)
        if abs_path.is_file():
            try:
                text = abs_path.read_text(encoding="utf-8")
                post = frontmatter.loads(text)
                return post.content
            except Exception:
                pass
        # Fallback: vault-ingest rows (e.g. `notes/01-Projects/Foo.md`,
        # `work/.../bar.md#chunk-3`) don't resolve to disk via `memory_dir`
        # because the label-prefixed path lives outside `data_dir`. The
        # body was written into the FTS table at ingest time — read it
        # from there so retrieval surfaces real snippets instead of "".
        try:
            row = self.store._conn.execute(
                "SELECT body FROM fts WHERE id = "
                "(SELECT id FROM meta WHERE path = ?)",
                (rel_path,),
            ).fetchone()
            if row and row["body"]:
                return str(row["body"])
        except Exception:
            pass
        return ""


# ── Helpers ──────────────────────────────────────────────────────────────

