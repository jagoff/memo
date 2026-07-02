"""Forget / unforget / delete operations for `Memory`.

Extracted from `_WriteOpsMixin` to keep each file under 800 lines.
`Memory` facade inherits this mixin alongside the others.
"""

from __future__ import annotations

import contextlib

from memo.lifecycle import (
    FORGET_AFTER_KEY,
    FORGET_REASON_KEY,
    IS_FORGOTTEN_KEY,
)
from memo.memory._base import _MemoryBase
from memo.memory.record import (
    MemoryRecord,
    _extract_provenance,
    _now_iso,
)


class _DeleteOpsMixin(_MemoryBase):
    # -- forget (soft, reversible) ------------------------------------------

    def forget(self, id_: str, *, reason: str | None = None) -> MemoryRecord | None:
        """Soft-forget a memory: keep the file + index, but exclude it from
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
        """Reverse a `forget`: clear `is_forgotten` so the memory is searchable
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
        """Remove from disk + store. Returns True if anything was deleted.

        Authority contract (markdown is the source of truth): the canonical
        `.md` is removed LAST to prevent data loss. If store operations succeed
        but file deletion fails, we rollback the store operations to keep the
        system consistent. This prevents orphaned index rows when the file is
        deleted but subsequent operations fail.
        """
        resolved = self.resolve_id(id_)
        if resolved is None:
            return False
        id_ = resolved
        r = self.store.get(id_)
        if not r:
            return False

        # Pre-fetch embedding + body text for rollback (store.get() omits them).
        # vec0 returns the embedding as a packed little-endian float32 blob —
        # deserialize it back to list[float] so upsert() can re-serialize on
        # restore. (An earlier `isinstance(..., (list, tuple))` check never
        # matched the blob, so the embedding was silently dropped on rollback.)
        stored_embedding: list[float] = []
        stored_body_text: str = ""
        blob = self.store.get_embedding_blob(id_) if self.store.has_vector(id_) else None
        if isinstance(blob, (bytes, bytearray)):
            import struct

            stored_embedding = list(struct.unpack(f"<{len(blob) // 4}f", blob))
        elif isinstance(blob, (list, tuple)):
            stored_embedding = list(blob)
        stored_body_text = self.store.get_fts_body(id_)
        # topic_key + normalized_hash live ONLY in the sqlite index (not in the
        # .md), and store.get() omits them — pre-fetch so the rollback restores
        # them, else a later same-topic save would duplicate instead of update.
        stored_topic_key, stored_normalized_hash = self.store.get_dedup_keys(id_)

        # Step 1: drop the derived index row + edges first (reversible via reindex)
        existed = self.store.delete(id_)
        if not existed:
            self._write_gen += 1
            return False

        # Step 2 (final, authoritative): remove the canonical .md FIRST, before
        # touching any other derived/audit state. Until the delete is known to
        # have succeeded, only the store index (rolled back below) and the .md
        # are mutated — so a failed unlink leaves history, graph edges, and
        # receipts untouched (no spurious 'delete' audit event, no dropped graph
        # edges for a memory that actually survives the failure).
        md_path = self._resolve_existing(r["path"])
        try:
            md_path.unlink(missing_ok=True)
        except OSError as exc:
            # File deletion failed but the store index was already dropped.
            # Roll the index back so the memory is recoverable; the reverse
            # (file gone, store intact) would cause permanent data loss.
            from memo.errors import StorageError

            try:
                self.store.upsert(
                    id_=id_,
                    path=r["path"],
                    title=r["title"],
                    type_=r["type"],
                    tags=r.get("tags") or [],
                    created=r["created"],
                    updated=r["updated"],
                    body_hash=r.get("body_hash") or "",
                    embedding=stored_embedding,
                    extra=r.get("extra"),
                    body_text=stored_body_text,
                    topic_key=stored_topic_key,
                    normalized_hash=stored_normalized_hash,
                )
            except Exception as restore_exc:
                raise StorageError(
                    f"delete partially failed AND rollback failed: {restore_exc}. "
                    "Run 'memo reindex' to recover."
                ) from restore_exc

            raise StorageError(
                f"delete partially failed: store operations succeeded but could not remove "
                f"canonical .md {md_path}: {exc}. Store record restored where possible. "
                "Manual cleanup may be needed."
            ) from exc

        # Delete is now authoritative (index dropped + .md gone). Only now mutate
        # the derived/audit state — these run strictly after the point of no
        # return, so they never need rolling back on a failed delete.
        # Step 3: log history (audit trail of a *completed* delete)
        self.history.log_delete(
            ts=_now_iso(),
            record_id=id_,
            title=r["title"],
            type_=r["type"],
        )

        # Step 4: drop graph edges (derived; rebuildable via reindex)
        self.graph.drop_for_memoria(id_)

        # Step 5: drop contradiction pairs (non-critical, suppress errors)
        if self._contradict_store is not None:
            with contextlib.suppress(Exception):
                self._contradict_store.drop_for_memoria(id_)

        # Step 6: emit receipts/events (non-critical, suppress errors)
        try:
            from memo.receipts import emit_receipt

            emit_receipt(
                "delete",
                text=f"Memo deleted memory {id_[:8]} ({r['type']}): {r['title']}",
                meta={
                    "id": id_,
                    "type": r["type"],
                    "title": r["title"],
                    "path": r["path"],
                },
            )
        except Exception:  # noqa: S110
            pass  # non-critical: file deletion is the authoritative step

        try:
            from memo.consciousness_ledger import emit_event

            emit_event(
                "delete",
                subject_uri=f"memo://memoria/{id_}",
                trace_id=(_extract_provenance(r.get("extra") or {}) or {}).get(
                    "synapse_trace_id", ""
                ),
                actor="memo",
                payload={
                    "id": id_,
                    "type": r["type"],
                    "title": r["title"],
                    "path": r["path"],
                },
            )
        except Exception:  # noqa: S110
            pass  # non-critical: file deletion is the authoritative step

        self._write_gen += 1
        return True
