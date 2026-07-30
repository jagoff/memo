"""Forget / unforget / delete operations for `Memory`.

Extracted from `_WriteOpsMixin` to keep each file under 800 lines.
`Memory` facade inherits this mixin alongside the others.
"""

from __future__ import annotations

import contextlib
from typing import Any

from memo.contracts import ActorIdentity
from memo.flags import flag_bool
from memo.lifecycle import (
    FORGET_AFTER_KEY,
    FORGET_REASON_KEY,
    IS_FORGOTTEN_KEY,
)
from memo.memory._base import _MemoryBase
from memo.memory.record import (
    MemoryRecord,
    _extract_provenance,
    _log,
    _now_iso,
    is_derived_chunk_id,
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

    def delete(self, id_: str, *, actor: ActorIdentity | None = None) -> bool:
        """Serialize delete with save/update filesystem-index transitions."""
        if is_derived_chunk_id(id_):
            raise ValueError("derived chunk records are read-only; update the parent memory")
        with self._data_dir_write_lock():
            return self._delete_locked(id_, actor=actor)

    def _delete_locked(
        self,
        id_: str,
        *,
        actor: ActorIdentity | None = None,
    ) -> bool:
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
        canonical_body = self._read_body(str(r["path"]))
        decision = self.write_policy.preflight(
            title=str(r["title"]),
            content=canonical_body,
            tags=list(r.get("tags") or ()),
            extra=dict(r.get("extra") or {}),
            actor=actor
            or ActorIdentity(
                actor_id="memo-delete",
                actor_kind="agent",
            ),
        )
        self.write_policy.enforce(decision)

        # Validate the untrusted store path before mutating any derived state.
        # Otherwise a traversal row could both escape the vault and leave the
        # index deleted before path validation had a chance to fail.
        md_path = self._resolve_existing(r["path"])

        # Pre-fetch embedding + body text for rollback (store.get() omits them).
        # vec0 returns the embedding as a packed little-endian float32 blob —
        # deserialize it back to list[float] so upsert() can re-serialize on
        # restore. (An earlier `isinstance(..., (list, tuple))` check never
        # matched the blob, so the embedding was silently dropped on rollback.)
        stored_embedding: list[float] = []
        stored_body_text: str = ""
        had_vector = self.store.has_vector(id_)
        blob = self.store.get_embedding_blob(id_) if had_vector else None
        if isinstance(blob, (bytes, bytearray)):
            stored_embedding = self.store.unpack_embedding(bytes(blob))
        elif isinstance(blob, (list, tuple)):
            stored_embedding = list(blob)
        stored_body_text = self.store.get_fts_body(id_)
        # topic_key + normalized_hash live ONLY in the sqlite index (not in the
        # .md), and store.get() omits them — pre-fetch so the rollback restores
        # them, else a later same-topic save would duplicate instead of update.
        stored_topic_key, stored_normalized_hash = self.store.get_dedup_keys(id_)

        # On the HARD-delete path, store.delete() also wipes the user-signal
        # tables (access counts, memory_health, source_feedback) — PRIMARY data
        # absent from the .md that the upsert rollback below does NOT restore.
        # Snapshot this id's signal rows first so a failed unlink can put them
        # back. Soft-delete keeps the row (and its signal), so it needs none.
        signal_backup: dict[str, list[dict[str, Any]]] | None = None
        if not flag_bool("MEMO_SOFT_DELETE"):
            with contextlib.suppress(Exception):
                dumped = self.store.dump_signal()
                signal_backup = {
                    "access": [r for r in dumped.get("access", []) if r.get("id") == id_],
                    "memory_health": [
                        r for r in dumped.get("memory_health", []) if r.get("id") == id_
                    ],
                    "source_feedback": [
                        r for r in dumped.get("source_feedback", []) if r.get("source_id") == id_
                    ],
                }

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
        try:
            md_path.unlink(missing_ok=True)
        except OSError as exc:
            # File deletion failed but the store index was already dropped.
            # Roll the index back so the memory is recoverable; the reverse
            # (file gone, store intact) would cause permanent data loss.
            from memo.errors import StorageError

            try:
                restore_kwargs = {
                    "id_": id_,
                    "path": r["path"],
                    "title": r["title"],
                    "type_": r["type"],
                    "tags": r.get("tags") or [],
                    "created": r["created"],
                    "updated": r["updated"],
                    "body_hash": r.get("body_hash") or "",
                    "extra": r.get("extra"),
                    "body_text": stored_body_text,
                    "topic_key": stored_topic_key,
                    "normalized_hash": stored_normalized_hash,
                }
                # Restore the wiped signal rows BEFORE re-seeding via upsert:
                # upsert's `INSERT OR IGNORE` would otherwise plant default
                # access/health rows that block the real values (merge_signal
                # is newer-wins on health). Best-effort — the meta-row restore
                # below is the load-bearing anti-data-loss step.
                if signal_backup and any(signal_backup.values()):
                    with contextlib.suppress(Exception):
                        self.store.merge_signal(signal_backup)
                if had_vector:
                    self.store.upsert(embedding=stored_embedding, **restore_kwargs)
                else:
                    self.store.upsert_text_only(**restore_kwargs)
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
        # Chunk rows have no canonical files of their own; once their parent is
        # authoritatively gone they must be hard-deleted as derived state.
        try:
            for chunk in self.store.chunks_by_parent_id(id_):
                self.store.hard_delete(chunk["id"])
        except Exception as exc:
            _log.warning("delete(%s): derived chunk cleanup failed — %s", id_[:8], exc)

        # Step 3: log history (audit trail of a *completed* delete). Runs after
        # the point of no return like Steps 4-7, so a history-sidecar failure
        # must not make the already-completed delete report failure.
        with contextlib.suppress(Exception):
            self.history.log_delete(
                ts=_now_iso(),
                record_id=id_,
                title=r["title"],
                type_=r["type"],
                tags=list(r.get("tags") or ()),
                body=canonical_body,
            )

        # Step 4: drop graph edges (derived; rebuildable via reindex). Runs
        # after the point of no return, so a graph-sidecar failure (e.g. a
        # locked graph.db) must not make the already-completed delete report
        # failure — a leftover edge is recoverable via `memo reindex`.
        try:
            self.graph.drop_for_memoria(id_)
        except Exception as exc:
            _log.warning("delete(%s): graph edge cleanup failed — %s", id_[:8], exc)

        # Canonical relation rows are audit signals: retain them and mark their
        # missing endpoint instead of deleting the judgment history.
        with contextlib.suppress(Exception):
            self.store.orphan_relations_for(id_)

        # Step 5: drop contradiction pairs (non-critical, suppress errors)
        if self._contradict_store is not None:
            with contextlib.suppress(Exception):
                self._contradict_store.drop_for_memoria(id_)

        # Step 5b: GC operational conflicts that reference this memory so a
        # detected contradiction never orphans into a permanent freeze_write
        # block once one of its subject memories is gone (non-critical).
        with contextlib.suppress(Exception):
            self.operational.gc_conflicts_for_memory(id_)

        # Step 5c: drop temporal fact edges sourced from this memory. Unlike the
        # graph edges above, these are NOT recovered by a later `memo reindex`:
        # incremental reindex only iterates .md files still on disk, and this
        # memory's file is already gone — so orphaned fact rows would keep
        # surfacing in the SessionStart briefing and the temporal CLI/MCP reads
        # (which query fact_edges directly). Purge them here (non-critical).
        with contextlib.suppress(Exception):
            self.fact_edges.delete_for_source(id_)

        # Step 6: append a native receipt. A journal failure cannot reverse an
        # authoritative completed delete.
        with contextlib.suppress(Exception):
            provenance = _extract_provenance(r.get("extra") or {})
            self.operational.receipt(
                "delete",
                subject_uri=f"memo://memoria/{id_}",
                trace_id=str(provenance.get("trace_id") or ""),
                actor_id=str(provenance.get("actor_id") or "memo"),
                metadata={
                    "id": id_,
                    "type": r["type"],
                    "title": r["title"],
                    "path": r["path"],
                },
            )

        # Drop crossref rows for the deleted memory (flag-gated, best-effort).
        from memo.flags import flag_bool as _flag_bool

        if _flag_bool("MEMO_CROSSREF_INDEX"):
            try:
                self.crossref.remove_memoria(id_)
            except Exception as exc:
                _log.debug("delete(%s): crossref cleanup skipped — %s", id_[:8], exc)

        # Step 7: purge version history (derived; not in markdown, so it would
        # otherwise grow unbounded in versions.db on every hard delete). Only on
        # hard delete — soft delete keeps the row and its history. Guard on the
        # db existing so a delete never eagerly creates an empty versions.db.
        if not flag_bool("MEMO_SOFT_DELETE") and (self.cfg.state_dir / "versions.db").is_file():
            with contextlib.suppress(Exception):
                self.versioning.version_store.delete_versions(id_)

        self._write_gen += 1
        return True
