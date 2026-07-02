"""Partial-failure / blindaje tests for the save + delete paths.

These pin the "markdown is the source of truth; sqlite is a rebuildable
index" contract at its two dangerous seams:

  * save()  — the canonical `.md` is written BEFORE the (slow, fallible)
    embed + index. If indexing fails, the memory must survive on disk,
    stamped `_memo_embed_pending`, and be recoverable via `memo reindex`.
    It must never be lost just because the embedder or store hiccuped.

  * delete() — the derived index is dropped FIRST, the canonical `.md`
    LAST. If the final unlink fails, the store is rolled back and a
    `StorageError` is raised, so the memory is never left half-deleted
    (index gone, file present) in a way that silently loses it.

All failures are forced with monkeypatch (embedder raising, store raising,
`Path.unlink` raising) — no MLX, no real I/O faults. The stub embedder is
4-dim (see the `mem_with_stub` fixture in conftest.py).
"""

from __future__ import annotations

import pytest

from memo.errors import StorageError
from memo.memory import Memory


def _working_embed(_inputs):
    # 4-dim, L2-normalised: norm = sqrt(4 * 0.5^2) = 1.0. Matches the
    # `mem_with_stub` fixture's embedder_dims=4 so assert_valid_embedding passes.
    return [[0.5, 0.5, 0.5, 0.5] for _ in _inputs]


def _explode_embed(_inputs):
    raise RuntimeError("embedder down (injected)")


def _wrong_dim_embed(_inputs):
    # 7-dim against a 4-dim config → assert_valid_embedding raises ValueError.
    return [[1.0] * 7 for _ in _inputs]


# ── save(): index failure keeps the .md and stays recoverable ────────────────


def test_save_index_failure_stamps_pending_and_recovers_via_reindex(
    mem_with_stub: Memory, monkeypatch
):
    """Embed fails after the .md is on disk → the memory is stamped
    `_memo_embed_pending`, stays BM25-searchable via the text-only index,
    and a later `memo reindex` (with a working embedder) fills the vector
    and clears the marker. The save never raises."""
    monkeypatch.setattr(mem_with_stub.embedder, "embed", _explode_embed)

    rec = mem_with_stub.save(content="cuerpo recuperable via reindex", title="Recuperable")

    # 1. The canonical .md is on disk with the pending marker.
    abs_path = mem_with_stub.cfg.memory_dir / rec.path
    assert abs_path.is_file()
    on_disk = abs_path.read_text(encoding="utf-8")
    assert "_memo_embed_pending" in on_disk
    assert rec.extra.get("_memo_embed_pending") is True

    # 2. Text-only index row exists (no vector yet) → still BM25-searchable.
    assert mem_with_stub.store.get(rec.id) is not None
    assert mem_with_stub.store.has_vector(rec.id) is False
    hits = mem_with_stub.search("recuperable reindex", mode="bm25", limit=5)
    assert rec.id in [h.id for h in hits]

    # 3. Recovery: a working embedder + reindex fills the vector and clears
    #    the on-disk marker — the pending save is fully replayed.
    monkeypatch.setattr(mem_with_stub.embedder, "embed", _working_embed)
    counts = mem_with_stub.reindex()

    assert counts["reindexed"] >= 1
    assert mem_with_stub.store.has_vector(rec.id) is True
    assert "_memo_embed_pending" not in abs_path.read_text(encoding="utf-8")


def test_save_valueerror_branch_stamps_pending_on_disk_before_reraise(
    mem_with_stub: Memory, monkeypatch
):
    """A dims/norm validation failure is raised loudly (misconfigured
    embedder) — but only AFTER the already-written .md is stamped
    `_memo_embed_pending`, so `memo reindex` can still replay it once the
    embedder is fixed. The memory is never lost to the loud failure."""
    monkeypatch.setattr(mem_with_stub.embedder, "embed", _wrong_dim_embed)

    with pytest.raises(ValueError, match="dim mismatch"):
        mem_with_stub.save(content="cuerpo estampado antes del raise", title="WrongDim")

    # Even though save() raised, the .md is on disk WITH the pending marker.
    mds = list(mem_with_stub.cfg.memory_dir.rglob("*.md"))
    assert len(mds) == 1, f"expected exactly one .md on disk, got {mds}"
    text = mds[0].read_text(encoding="utf-8")
    assert "cuerpo estampado antes del raise" in text
    assert "_memo_embed_pending" in text


def test_save_survives_total_store_failure_and_recovers(mem_with_stub: Memory, monkeypatch):
    """Worst case: BOTH the embedder AND the text-only index write are down.
    markdown-is-truth still holds — the .md is on disk with the pending
    marker, save() returns the record without raising, and even though the
    store never saw the row, `memo reindex` recovers it from disk."""

    def _boom_upsert_text_only(*_a, **_k):
        raise RuntimeError("store down (injected)")

    monkeypatch.setattr(mem_with_stub.embedder, "embed", _explode_embed)
    monkeypatch.setattr(mem_with_stub.store, "upsert_text_only", _boom_upsert_text_only)

    rec = mem_with_stub.save(content="cuerpo con store caido", title="StoreDown")

    # The record is returned (never raised past a successful disk write).
    assert rec.extra.get("_memo_embed_pending") is True
    abs_path = mem_with_stub.cfg.memory_dir / rec.path
    assert abs_path.is_file()
    assert "_memo_embed_pending" in abs_path.read_text(encoding="utf-8")

    # The store genuinely never got the row (the text-only upsert failed).
    assert mem_with_stub.store.get(rec.id) is None

    # Recovery: disk is the truth. reindex (store.upsert is untouched) adds
    # the row with a real vector from a working embedder.
    monkeypatch.setattr(mem_with_stub.embedder, "embed", _working_embed)
    counts = mem_with_stub.reindex()

    assert counts["added"] >= 1
    assert mem_with_stub.store.get(rec.id) is not None
    assert mem_with_stub.store.has_vector(rec.id) is True


# ── delete(): unlink failure rolls back and stays consistent ─────────────────


def _unlink_boom_for_md(mem: Memory, monkeypatch):
    """Monkeypatch Path.unlink to raise OSError for `.md` files only."""
    real_unlink = type(mem.cfg.memory_dir).unlink

    def _boom(self, *a, **k):
        if self.name.endswith(".md"):
            raise OSError("permission denied (injected)")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr("pathlib.Path.unlink", _boom)


def test_delete_unlink_failure_rolls_back_index_and_keeps_md(mem_with_stub: Memory, monkeypatch):
    """The canonical .md is removed LAST. When the final unlink fails, the
    store is rolled back (row + vector + body restored) and StorageError is
    raised — leaving BOTH the .md and the index present (consistent), never
    the data-losing half-state (index dropped, .md gone)."""
    rec = mem_with_stub.save(content="protegido contra borrado parcial", title="Protegido")
    assert mem_with_stub.store.count() == 1

    _unlink_boom_for_md(mem_with_stub, monkeypatch)

    with pytest.raises(StorageError, match="delete partially failed"):
        mem_with_stub.delete(rec.id)

    # Index rolled back WITH its vector and body (not just bare metadata).
    assert mem_with_stub.store.count() == 1
    assert mem_with_stub.store.has_vector(rec.id) is True
    assert mem_with_stub.store.get_fts_body(rec.id) != ""

    # The .md is STILL on disk (unlink failed) → index and disk agree.
    assert (mem_with_stub.cfg.memory_dir / rec.path).is_file()

    # End-to-end consistency: the memory is fully readable again.
    fetched = mem_with_stub.get(rec.id)
    assert fetched is not None
    assert "protegido contra borrado parcial" in fetched.body


def test_delete_double_failure_reports_rollback_and_reindex_hint(
    mem_with_stub: Memory, monkeypatch
):
    """If the unlink fails AND the rollback upsert also fails, StorageError
    still surfaces — and its message tells the operator to run `memo reindex`
    to recover, rather than silently swallowing the double failure."""
    rec = mem_with_stub.save(content="doble falla", title="DobleFalla")

    def _boom_upsert(*_a, **_k):
        raise RuntimeError("store rollback down (injected)")

    _unlink_boom_for_md(mem_with_stub, monkeypatch)
    monkeypatch.setattr(mem_with_stub.store, "upsert", _boom_upsert)

    with pytest.raises(StorageError, match="rollback failed") as exc_info:
        mem_with_stub.delete(rec.id)

    assert "memo reindex" in str(exc_info.value)


def test_delete_rollback_preserves_topic_key_dedup(mem_with_stub: Memory, monkeypatch):
    """topic_key + normalized_hash live ONLY in the sqlite index (not in the
    .md frontmatter), so a delete-rollback that restores the row via
    store.get() (which omits them) would drop them — and a later same-topic
    save would then create a DUPLICATE instead of updating in place. The
    rollback must pre-fetch and restore the dedup keys so the row stays
    reachable by its topic_key."""
    rec = mem_with_stub.save(
        content="memoria con dedup key",
        title="DedupProtegido",
        topic_key="tk-dedup",
    )
    # Precondition: the row is reachable by its topic_key before the delete.
    assert mem_with_stub.store.find_by_topic_key("tk-dedup") is not None

    _unlink_boom_for_md(mem_with_stub, monkeypatch)
    with pytest.raises(StorageError, match="delete partially failed"):
        mem_with_stub.delete(rec.id)

    # After the rollback the dedup key survives → a same-topic save updates in
    # place instead of creating a duplicate. Without restoring topic_key on
    # rollback this returns None and the memory silently duplicates.
    assert mem_with_stub.store.find_by_topic_key("tk-dedup") is not None, (
        "topic_key dropped on delete-rollback → next same-topic save would duplicate"
    )


def test_delete_rollback_leaves_no_spurious_history_event(mem_with_stub: Memory, monkeypatch):
    """History log + graph-edge drop run only AFTER the authoritative unlink
    succeeds. A failed-unlink rollback must therefore leave NO 'delete' audit
    event — the memory was not actually deleted, so the audit trail must not
    claim it was."""
    rec = mem_with_stub.save(content="sin evento espurio", title="SinEvento")
    assert mem_with_stub.history.list_recent(op="delete", record_id=rec.id) == []

    _unlink_boom_for_md(mem_with_stub, monkeypatch)
    with pytest.raises(StorageError, match="delete partially failed"):
        mem_with_stub.delete(rec.id)

    assert mem_with_stub.history.list_recent(op="delete", record_id=rec.id) == [], (
        "delete-rollback logged a spurious 'delete' audit event for a surviving memory"
    )


def test_successful_delete_still_logs_history_event(mem_with_stub: Memory):
    """The reorder must not break the happy path: a completed delete logs
    exactly one 'delete' audit event."""
    rec = mem_with_stub.save(content="borrado real", title="BorradoReal")

    assert mem_with_stub.delete(rec.id) is True

    events = mem_with_stub.history.list_recent(op="delete", record_id=rec.id)
    assert len(events) == 1
