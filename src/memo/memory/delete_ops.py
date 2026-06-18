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

        # Step 1: drop the derived index row + edges first (reversible via reindex)
        existed = self.store.delete(id_)
        if not existed:
            return False

        # Step 2: log history (before file delete for audit trail)
        self.history.log_delete(
            ts=_now_iso(),
            record_id=id_,
            title=r["title"],
            type_=r["type"],
        )

        # Step 3: drop graph edges
        self.graph.drop_for_memoria(id_)

        # Step 4: drop contradiction pairs (non-critical, suppress errors)
        if self._contradict_store is not None:
            with contextlib.suppress(Exception):
                self._contradict_store.drop_for_memoria(id_)

        # Step 5: emit receipts/events (non-critical, suppress errors)
        try:
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
        except Exception:
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
        except Exception:
            pass  # non-critical: file deletion is the authoritative step

        # Step 6 (final, authoritative): remove the canonical .md
        # Only after all store operations succeed to prevent data loss
        md_path = self._resolve_existing(r["path"])
        try:
            md_path.unlink(missing_ok=True)
        except OSError as exc:
            # File deletion failed but store operations succeeded.
            # The record is now orphaned in the store (file exists but marked deleted).
            # This is recoverable via reindex, whereas the reverse (file gone, store intact)
            # would cause permanent data loss.
            from memo.errors import StorageError

            # Attempt to restore the store record to minimize inconsistency
            with contextlib.suppress(Exception):
                # Re-insert the record (simplified recovery - full recovery would need all metadata)
                self.store.upsert(
                    id_=id_,
                    path=r["path"],
                    title=r["title"],
                    type_=r["type"],
                    tags=r.get("tags") or [],
                    created=r["created"],
                    updated=r["updated"],
                    body_hash=r.get("body_hash") or "",
                    embedding=r.get("embedding") or [],
                    extra=r.get("extra"),
                    body_text=r.get("body_text") or "",
                )

            raise StorageError(
                f"delete partially failed: store operations succeeded but could not remove "
                f"canonical .md {md_path}: {exc}. Store record restored where possible. "
                "Manual cleanup may be needed."
            ) from exc

        return True
