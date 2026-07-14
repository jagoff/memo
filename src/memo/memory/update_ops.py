"""Update operations for `Memory` — patch title/type/tags/body on existing records.

Extracted from `_WriteOpsMixin` to keep each file under 800 lines.
`Memory` facade inherits this mixin alongside the others.
"""

from __future__ import annotations

import builtins
import logging
from typing import Any

import frontmatter

from memo.memory._base import _MemoryBase
from memo.memory.record import (
    _VALID_TYPES,
    MemoryRecord,
    _extract_provenance,
    _normalise_tags,
    _now_iso,
)
from memo.util import sha256_short as _sha256_short

_log = logging.getLogger(__name__)


class _UpdateOpsMixin(_MemoryBase):
    # -- update -------------------------------------------------------------

    def update(
        self,
        id_: str,
        *,
        title: str | None = None,
        type_: str | None = None,
        tags: builtins.list[str] | None = None,
        content: str | None = None,
        replace: tuple[str, str] | None = None,
        append: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> MemoryRecord | None:
        """Patch one or more fields on an existing record.

        Only the kwargs you pass are touched; everything else stays as-is.
        Re-embed only if `content` changed (body_hash check). The file
        path stays stable — renaming the slug after the fact would break
        wikilinks the user may have created in their vault.

        `replace=(old, new)` is an exact-string surgical edit — old must occur
        exactly once (ValueError otherwise) so unchanged content stays
        byte-identical; `append` adds a paragraph. Both route through the same
        versioned path as `content`.
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
        if sum(x is not None for x in (content, replace, append)) > 1:
            raise ValueError("update: pass at most one of content=, replace=, append=")

        new_title = (title.strip() if title else r["title"]) or r["title"]
        new_type = type_ or r["type"]
        new_tags = _normalise_tags(tags) if tags is not None else r["tags"]
        new_extra = extra if extra is not None else dict(r.get("extra") or {})
        now_iso = _now_iso()

        # Body resolution: provided > on-disk > empty.
        old_body = self._read_body(r["path"])
        if replace is not None:
            old_str, new_str = replace
            if not old_str:
                raise ValueError("replace: old string must be non-empty")
            occurrences = old_body.count(old_str)
            if occurrences == 0:
                raise ValueError(f"replace: old string not found in body of {id_[:8]}")
            if occurrences > 1:
                raise ValueError(
                    f"replace: old string occurs {occurrences} times in {id_[:8]}; must be unique"
                )
            content = old_body.replace(old_str, new_str, 1)
        elif append is not None:
            content = (
                old_body.rstrip("\n") + "\n\n" + append.strip()
                if old_body.strip()
                else append.strip()
            )
        new_body = content if content is not None else old_body
        new_body = new_body[: self.cfg.max_content_chars]
        new_body_hash = _sha256_short(new_body)
        body_changed = new_body_hash != r["body_hash"]
        title_changed = new_title != r["title"]
        embed_pending = bool(new_extra.get("_memo_embed_pending"))
        # A record saved with ``defer_embed`` has no vector yet.  Metadata-only
        # corrections must remain available in no-model environments instead
        # of trying to load an embedder merely to update the title.  The
        # pending marker keeps the eventual reindex responsible for composing
        # the vector from the corrected title and body.
        embedding_required = body_changed or (title_changed and not embed_pending)

        # Snapshot the prior record BEFORE mutating so `memo version
        # history/diff/rollback` have data. Gated on a real edit so pure
        # extra/provenance bumps (e.g. the cache dirty-bit) don't spam version
        # rows. Best-effort: a versioning failure must never break the update.
        if body_changed or title_changed or new_type != r["type"] or new_tags != r["tags"]:
            try:
                self.versioning.track_update(
                    id_,
                    r["title"],
                    r["type"],
                    r["tags"],
                    old_body,
                    reason="pre-update snapshot",
                )
            except Exception as exc:
                _log.debug("update(%s): version snapshot failed — %s", id_[:8], exc)

        abs_path = self._resolve_existing(r["path"])
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        # Re-embed when the body OR title changed — both are part of the
        # embed input now (see `_compose_for_embed`). Pure retag/type
        # changes still skip the embedder. Embed BEFORE touching the file
        # so a failure doesn't leave file and store diverged.
        if embedding_required:
            embedding = self._embed_cached(
                self._compose_for_embed(new_title, new_body),
                ctx=f"update id={id_[:8]}",
            )

        # Lock around file+store write so concurrent updates to the same id
        # don't lose the first writer's changes (last-writer-wins).
        with self._save_path_lock:
            post = frontmatter.Post(
                new_body,
                id=id_,
                title=new_title,
                type=new_type,
                tags=new_tags,
                created=r["created"],
                updated=now_iso,
            )
            post["extra"] = new_extra or {}
            abs_path.write_text(frontmatter.dumps(post), encoding="utf-8")

            if embedding_required:
                self.store.upsert(
                    id_=id_,
                    path=r["path"],
                    title=new_title,
                    type_=new_type,
                    tags=new_tags,
                    created=r["created"],
                    updated=now_iso,
                    body_hash=new_body_hash,
                    embedding=embedding,
                    extra=new_extra,
                    body_text=new_body,
                )
            else:
                self.store.update_meta(
                    id_=id_,
                    title=new_title,
                    type_=new_type,
                    tags=new_tags,
                    updated=now_iso,
                    extra=new_extra,
                )

        # Crossref edges (flag-gated): body edits can add/remove typed links.
        if body_changed:
            from memo.flags import flag_bool as _flag_bool

            if _flag_bool("MEMO_CROSSREF_INDEX"):
                try:
                    self.crossref.index_source(id_, new_body)
                except Exception as exc:
                    _log.debug("update(%s): crossref index skipped — %s", id_[:8], exc)

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
        # Record the old `updated` so time-machine can restore the exact
        # timestamp when replaying history in reverse (reconstruct() uses
        # old_val). Only add when something else changed — a pure
        # timestamp-bump carries no semantic change worth logging.
        if delta:
            delta["updated"] = (r.get("updated"), now_iso)
        # Track provenance churn so a re-route (e.g. Synapse re-issues a
        # different trace_id on the same memory) shows up in history.
        old_prov = _extract_provenance(r.get("extra") or {})
        new_prov = _extract_provenance(new_extra)
        if old_prov != new_prov:
            delta["_provenance"] = (old_prov, new_prov)
        if delta:
            self.history.log_update(
                ts=now_iso,
                record_id=id_,
                title=new_title,
                type_=new_type,
                delta=delta,
            )

        updated_rec = MemoryRecord(
            id=id_,
            path=r["path"],
            title=new_title,
            type=new_type,
            tags=new_tags,
            created=r["created"],
            updated=now_iso,
            body=new_body,
            extra=new_extra,
        )
        if delta:
            from memo.receipts import emit_receipt

            emit_receipt(
                "update",
                text=f"Memo updated memory {id_[:8]}: {', '.join(sorted(delta.keys()))}",
                meta={
                    "id": id_,
                    "type": new_type,
                    "title": new_title,
                    "delta_keys": ",".join(sorted(delta.keys())),
                },
            )
            # M2b: also emit to the unified trinity ledger.
            self._emit_ledger("update", updated_rec, new_prov)
        self._write_gen += 1
        return updated_rec

    # -- last saved ----------------------------------------------------------

    def last_saved_id(self, *, limit: int = 20) -> str | None:
        """Id of the most recent memory saved on this device that still exists.

        Scans the audit log's `save` events (newest first, this device only —
        synced events from other machines don't count as "just saved here")
        and returns the first record_id not deleted since. Backs the
        "rename what I just saved" flows (`memo rename`, MCP `memo_rename`).
        """
        events = self.history.list_recent(
            limit=limit,
            op="save",
            device_id=self.history.device_id,
        )
        for ev in events:
            rid = ev.get("record_id")
            if rid and self.get(rid) is not None:
                return str(rid)
        return None
