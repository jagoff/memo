from __future__ import annotations

from memo.memory import Memory


def test_save_index_failure_recovers_on_reindex(mem_with_stub: Memory, monkeypatch):
    calls = {"n": 0}
    real_embed = type(mem_with_stub.embedder).embed

    def _flaky(self, inputs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("embedder down")
        return real_embed(self, inputs)

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _flaky)
    rec = mem_with_stub.save(content="recupera via reindex", title="Reindexable")
    assert not mem_with_stub.store.has_vector(rec.id)

    out = mem_with_stub.reindex()
    assert out["reindexed"] >= 1
    assert mem_with_stub.store.has_vector(rec.id)


def test_edit_md_then_reindex_markdown_wins(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="contenido original", title="Editable")
    abs_path = mem_with_stub.cfg.memory_dir / rec.path
    text = abs_path.read_text(encoding="utf-8")
    abs_path.write_text(
        text.replace("contenido original", "contenido EDITADO a mano"), encoding="utf-8"
    )

    out = mem_with_stub.reindex()
    assert out["reindexed"] >= 1
    fetched = mem_with_stub.get(rec.id)
    assert fetched is not None
    assert "EDITADO a mano" in fetched.body
    assert "contenido original" not in fetched.body


def test_reindex_rebuild_preserves_signal(mem_with_stub: Memory):
    a = mem_with_stub.save(content="memoria una", title="A")
    b = mem_with_stub.save(content="memoria dos", title="B")
    store = mem_with_stub.store
    store.touch([a.id])
    store.touch([a.id])
    store.boost_roi_batch([b.id], delta=0.3)
    assert store.get_access(a.id)["access_count"] == 2
    assert store.get_health_batch([b.id])[b.id]["roi_score"] > 1.0

    out = mem_with_stub.reindex(rebuild=True)
    assert out["added"] == 2
    assert mem_with_stub.store.count() == 2
    assert store.get_access(a.id)["access_count"] == 2
    assert store.get_health_batch([b.id])[b.id]["roi_score"] > 1.0
    assert mem_with_stub.get(a.id) is not None
    assert mem_with_stub.get(b.id) is not None


def test_reindex_rebuild_refused_when_data_dir_empty(mem_with_stub: Memory):
    """Rebuild against a data_dir with 0 .md must refuse, not wipe the index.
    Markdown is the only source that can repopulate it — if it vanished (deleted
    dir / half-broken clone), truncating the index destroys the last copy."""
    import pytest

    from memo.errors import StorageError

    a = mem_with_stub.save(content="no me borres", title="A")
    b = mem_with_stub.save(content="ni a mi", title="B")
    for md in mem_with_stub.cfg.memory_dir.rglob("*.md"):
        md.unlink()
    assert next(mem_with_stub.cfg.memory_dir.rglob("*.md"), None) is None
    assert mem_with_stub.store.count() == 2

    with pytest.raises(StorageError, match="refused"):
        mem_with_stub.reindex(rebuild=True)

    # index untouched — both memorias still recoverable
    assert mem_with_stub.store.count() == 2
    assert mem_with_stub.get(a.id) is not None
    assert mem_with_stub.get(b.id) is not None


def test_reindex_rebuild_drops_orphans(mem_with_stub: Memory):
    a = mem_with_stub.save(content="vive", title="A")
    b = mem_with_stub.save(content="muere", title="B")
    (mem_with_stub.cfg.memory_dir / b.path).unlink()

    out = mem_with_stub.reindex(rebuild=True)
    assert out["added"] == 1
    assert mem_with_stub.get(a.id) is not None
    assert mem_with_stub.get(b.id) is None
    assert mem_with_stub.store.count() == 1


def test_reindex_embedding_reuse_skips_second_pass(mem_with_stub: Memory, monkeypatch):
    mem_with_stub.save(content="alpha", title="A")
    mem_with_stub.save(content="beta", title="B")
    mem_with_stub.reindex(force=True)

    calls = {"n": 0}
    real_embed = type(mem_with_stub.embedder).embed

    def _counting(self, inputs):
        calls["n"] += 1
        return real_embed(self, inputs)

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _counting)
    out = mem_with_stub.reindex(force=True)
    assert out["reindexed"] == 2
    assert calls["n"] == 0


def test_reindex_force_reuses_warm_embed_cache(mem_with_stub: Memory, monkeypatch):
    rec = mem_with_stub.save(content="cuerpo", title="X")
    calls: list[int] = []
    orig = mem_with_stub.embedder.embed

    def _spy(inputs):
        calls.append(len(inputs))
        return orig(inputs)

    monkeypatch.setattr(mem_with_stub.embedder, "embed", _spy)

    counts = mem_with_stub.reindex()
    assert counts["reindexed"] == 0
    assert calls == []

    counts = mem_with_stub.reindex(force=True)
    assert counts["reindexed"] == 1
    assert calls == []

    with mem_with_stub.store._conn:
        mem_with_stub.store._conn.execute("DELETE FROM repo_embedding_cache")
    calls.clear()
    counts = mem_with_stub.reindex(force=True)
    assert counts["reindexed"] == 1
    assert calls == [1]
    fetched = mem_with_stub.get(rec.id)
    assert fetched is not None
    assert fetched.title == "X"


def test_reindex_picks_up_external_edit(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="primero", title="X")
    abs_path = mem_with_stub.cfg.memory_dir / rec.path
    import frontmatter as fm

    post = fm.loads(abs_path.read_text())
    post.content = "cuerpo editado a mano"
    abs_path.write_text(fm.dumps(post), encoding="utf-8")

    counts = mem_with_stub.reindex()
    assert counts["reindexed"] == 1
    assert counts["added"] == 0
    fetched = mem_with_stub.get(rec.id)
    assert fetched is not None
    assert "editado a mano" in fetched.body


def test_reindex_adds_orphan_disk_file(mem_with_stub: Memory):
    import frontmatter as fm

    md = mem_with_stub.cfg.memory_dir / "2026-05-06-restored.md"
    post = fm.Post(
        "memo restaurado de un backup",
        id="ffeeddccbbaa00112233445566778899",
        title="Restored",
        type="note",
        tags=["backup"],
        created="2026-05-06T19:00:00-03:00",
        updated="2026-05-06T19:00:00-03:00",
    )
    md.write_text(fm.dumps(post), encoding="utf-8")
    counts = mem_with_stub.reindex()
    assert counts["added"] == 1
    fetched = mem_with_stub.get("ffeeddccbbaa00112233445566778899")
    assert fetched is not None
    assert fetched.title == "Restored"


def test_reindex_rebuilds_declared_fact_edges_from_frontmatter(mem_with_stub: Memory):
    import frontmatter as fm

    rec = mem_with_stub.save(content="placeholder", title="Fact Source", type_="note")
    md = mem_with_stub.cfg.memory_dir / rec.path
    post = fm.loads(md.read_text(encoding="utf-8"))
    post.metadata["fact_edges"] = [
        {
            "subject": "memo",
            "predicate": "stores",
            "object": "temporal facts",
            "valid_at": "2026-01-01T00:00:00+00:00",
        }
    ]
    md.write_text(fm.dumps(post), encoding="utf-8")

    counts = mem_with_stub.reindex()
    rows = mem_with_stub.fact_edges.query(
        subject="memo",
        as_of="2026-02-01T00:00:00+00:00",
    )

    assert counts["facts"] == 1
    assert len(rows) == 1
    assert rows[0]["source_record_id"] == rec.id
    assert rows[0]["predicate"] == "stores"
    assert rows[0]["object"] == "temporal facts"


def test_reindex_updates_fact_edges_when_frontmatter_changes(mem_with_stub: Memory):
    import frontmatter as fm

    rec = mem_with_stub.save(
        content="memo backend fact",
        title="Backend",
        type_="note",
        extra={
            "fact_edges": [
                {
                    "subject": "memo",
                    "predicate": "backend",
                    "object": "old",
                    "valid_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        },
    )
    assert mem_with_stub.fact_edges.query(subject="memo", as_of="2026-02-01T00:00:00+00:00")

    md = mem_with_stub.cfg.memory_dir / rec.path
    post = fm.loads(md.read_text(encoding="utf-8"))
    extra = dict(post.metadata.get("extra") or {})
    extra["fact_edges"] = [
        {
            "subject": "memo",
            "predicate": "backend",
            "object": "new",
            "valid_at": "2026-01-01T00:00:00+00:00",
        }
    ]
    post.metadata["extra"] = extra
    md.write_text(fm.dumps(post), encoding="utf-8")

    counts = mem_with_stub.reindex()
    rows = mem_with_stub.fact_edges.query(subject="memo", as_of="2026-02-01T00:00:00+00:00")

    assert counts["facts"] == 1
    assert [r["object"] for r in rows] == ["new"]


def test_reindex_rebuild_drops_fact_edges_for_deleted_markdown(mem_with_stub: Memory):
    mem_with_stub.save(content="keep this markdown", title="Keep")
    rec = mem_with_stub.save(
        content="memo backend fact",
        title="Backend",
        type_="note",
        extra={
            "fact_edges": [
                {
                    "subject": "memo",
                    "predicate": "backend",
                    "object": "sqlite",
                    "valid_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        },
    )
    assert mem_with_stub.fact_edges.query(subject="memo", as_of="2026-02-01T00:00:00+00:00")
    (mem_with_stub.cfg.memory_dir / rec.path).unlink()

    counts = mem_with_stub.reindex(rebuild=True)

    assert counts["facts"] == 0
    assert mem_with_stub.fact_edges.query(
        subject="memo",
        as_of="2026-02-01T00:00:00+00:00",
    ) == []


def test_reindex_reclaims_path_held_by_soft_deleted_tombstone(mem_with_stub: Memory, monkeypatch):
    # Prod incident 2026-07-05: with MEMO_SOFT_DELETE on, a deleted memory
    # leaves a tombstone row that still occupies UNIQUE(meta.path). When a
    # new file (new id) reclaims the same path, reindex used to fail the
    # INSERT ("UNIQUE constraint failed: meta.path") because the collision
    # guard's lookup excluded soft-deleted rows and its soft delete kept
    # the path occupied.
    import frontmatter as fm

    monkeypatch.setenv("MEMO_SOFT_DELETE", "1")
    rec = mem_with_stub.save(content="primera encarnación", title="Reclaimable")
    rel = rec.path
    mem_with_stub.delete(rec.id)
    assert mem_with_stub.get(rec.id) is None

    new_id = "aabbccdd00112233445566778899ffee"
    md = mem_with_stub.cfg.memory_dir / rel
    md.parent.mkdir(parents=True, exist_ok=True)
    post = fm.Post(
        "segunda encarnación, mismo path",
        id=new_id,
        title="Reclaimable",
        type="note",
        created="2026-07-05T19:00:00-03:00",
        updated="2026-07-05T19:00:00-03:00",
    )
    md.write_text(fm.dumps(post), encoding="utf-8")

    counts = mem_with_stub.reindex()
    assert counts["added"] == 1
    assert counts["skipped"] == 0
    fetched = mem_with_stub.get(new_id)
    assert fetched is not None
    assert "segunda encarnación" in fetched.body
    # The tombstone was purged — the path now belongs to the new id only.
    row = mem_with_stub.store.get_by_path(rel, include_deleted=True)
    assert row is not None and row["id"] == new_id


def test_gc_reports_and_fixes_orphans(mem_with_stub: Memory):
    a = mem_with_stub.save(content="vivo", title="A")
    b = mem_with_stub.save(content="vivo", title="B")
    (mem_with_stub.cfg.memory_dir / b.path).unlink()
    report = mem_with_stub.gc(fix=False)
    assert b.id in report["orphan_store"]
    assert a.id not in report["orphan_store"]
    assert mem_with_stub.store.get(b.id) is not None
    mem_with_stub.gc(fix=True)
    assert mem_with_stub.store.get(b.id) is None
