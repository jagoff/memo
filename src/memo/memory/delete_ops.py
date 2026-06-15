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
        `.md` is removed FIRST. If the file exists but we cannot delete it
        (permission/IO error), we abort with `StorageError` and leave the
        index untouched — wiping the index while the truth-bearing file
        survives would desync the two. A *missing* file is fine: reference-tier
        rows (vault-ingest) and already-gone files resolve to a non-existent
        path, so we fall through and drop the orphaned index row.
        """
        resolved = self.resolve_id(id_)
        if resolved is None:
            return False
        id_ = resolved
        r = self.store.get(id_)
        if not r:
            return False
        # Step 1 (authoritative): remove the canonical .md. `missing_ok=True`
        # makes a non-existent file a no-op; only a real OSError aborts.
        md_path = self._resolve_existing(r["path"])
        try:
            md_path.unlink(missing_ok=True)
        except OSError as exc:
            from memo.errors import StorageError

            raise StorageError(
                f"delete refused: could not remove canonical .md {md_path}: {exc}. "
                "Index left intact to stay consistent with the source of truth."
            ) from exc
        # Step 2: the truth is gone — now drop the derived index row + edges.
        existed = self.store.delete(id_)
        if existed:
            self.history.log_delete(
                ts=_now_iso(),
                record_id=id_,
                title=r["title"],
                type_=r["type"],
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
        return bool(existed)
