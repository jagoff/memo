from __future__ import annotations

import pytest

from memo.memory import AmbiguousIdError, Memory


def test_get_returns_record_with_body(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="cuerpo del memo", title="X")
    fetched = mem_with_stub.get(rec.id)
    assert fetched is not None
    assert fetched.title == "X"
    assert "cuerpo del memo" in fetched.body


def test_get_missing_returns_none(mem_with_stub: Memory):
    assert mem_with_stub.get("nope") is None


def test_list_orders_recent_first(mem_with_stub: Memory):
    a = mem_with_stub.save(content="primero", title="A")
    import time

    time.sleep(1.1)
    b = mem_with_stub.save(content="segundo", title="B")
    items = mem_with_stub.list(limit=10)
    titles = [r.title for r in items]
    assert titles.index("B") < titles.index("A")
    assert {r.id for r in items} == {a.id, b.id}


def test_delete_removes_disk_and_index(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="borrar este", title="X")
    assert (mem_with_stub.cfg.memory_dir / rec.path).is_file()
    assert mem_with_stub.delete(rec.id) is True
    assert mem_with_stub.store.count() == 0
    assert not (mem_with_stub.cfg.memory_dir / rec.path).is_file()


def test_delete_missing_returns_false(mem_with_stub: Memory):
    assert mem_with_stub.delete("nope") is False


def test_delete_aborts_when_md_unlink_fails(mem_with_stub: Memory, monkeypatch):
    from memo.errors import StorageError

    rec = mem_with_stub.save(content="protegido", title="X")
    assert mem_with_stub.store.count() == 1

    real_unlink = type(mem_with_stub.cfg.memory_dir).unlink

    def _boom(self, *a, **k):
        if self.name.endswith(".md"):
            raise OSError("permission denied")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr("pathlib.Path.unlink", _boom)
    with pytest.raises(StorageError, match="delete partially failed"):
        mem_with_stub.delete(rec.id)
    # Store operations complete first, then file deletion fails → the record is
    # restored, INCLUDING its embedding. The rollback reads the vec0 blob and
    # deserializes it; a regression here (dropping the vector) would leave the
    # restored row unsearchable until the next reindex.
    assert mem_with_stub.store.count() == 1
    assert mem_with_stub.store.has_vector(rec.id) is True


def test_hard_delete_rollback_preserves_signal_tables(mem_with_stub: Memory, monkeypatch):
    """With MEMO_SOFT_DELETE=0, store.delete() wipes the user-signal tables
    (access, memory_health, source_feedback) — PRIMARY data not in the .md.
    A hard-delete that must roll back (unlink fails) must restore them, not
    reset access counts / drop feedback to defaults."""
    monkeypatch.setenv("MEMO_SOFT_DELETE", "0")
    from memo.errors import StorageError

    rec = mem_with_stub.save(content="con señal", title="Señal")
    # Accumulate user signal: an access hit + a 👍 on a query.
    mem_with_stub.store.touch([rec.id])
    mem_with_stub.store.touch([rec.id])
    mem_with_stub.store.record_source_feedback(
        source_id=rec.id,
        query_text="una consulta",
        query_emb=[0.0, 1.0, 0.0, 0.0],
        rating=1,
    )
    assert mem_with_stub.store.get_access(rec.id)["access_count"] == 2
    assert mem_with_stub.store.sources_with_feedback([rec.id]) == {rec.id}

    real_unlink = type(mem_with_stub.cfg.memory_dir).unlink

    def _boom(self, *a, **k):
        if self.name.endswith(".md"):
            raise OSError("permission denied")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr("pathlib.Path.unlink", _boom)
    with pytest.raises(StorageError, match="delete partially failed"):
        mem_with_stub.delete(rec.id)

    # Meta row restored, AND its signal preserved (not reset to defaults).
    assert mem_with_stub.store.count() == 1
    assert mem_with_stub.store.get_access(rec.id)["access_count"] == 2
    assert mem_with_stub.store.sources_with_feedback([rec.id]) == {rec.id}


def test_hard_delete_rollback_preserves_validity_and_review_state(
    mem_with_stub: Memory, monkeypatch
):
    """A rolled-back hard delete must not silently re-open a superseded fact.

    `upsert()` carries neither the validity interval nor the review state, so
    the restored row used to come back with invalid_at=NULL (re-entering normal
    recall as if it were still current) and verification reset to 'unverified'.
    """
    monkeypatch.setenv("MEMO_SOFT_DELETE", "0")
    from memo.errors import StorageError

    rec = mem_with_stub.save(content="hecho superado", title="Superado")
    mem_with_stub.store.update_validity(
        id_=rec.id,
        valid_at="2026-01-01T00:00:00+00:00",
        invalid_at="2026-06-01T00:00:00+00:00",
    )
    mem_with_stub.store.update_review_state(
        id_=rec.id,
        review_after="2026-12-01T00:00:00+00:00",
        verification_state="verified",
        verified_at=1,
    )

    real_unlink = type(mem_with_stub.cfg.memory_dir).unlink

    def _boom(self, *a, **k):
        if self.name.endswith(".md"):
            raise OSError("permission denied")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr("pathlib.Path.unlink", _boom)
    with pytest.raises(StorageError, match="delete partially failed"):
        mem_with_stub.delete(rec.id)

    restored = mem_with_stub.store.get(rec.id)
    assert restored is not None
    assert restored["invalid_at"] == "2026-06-01T00:00:00+00:00"
    assert restored["valid_at"] == "2026-01-01T00:00:00+00:00"
    assert restored["verification_state"] == "verified"


def test_delete_proceeds_when_md_already_missing(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="huérfano", title="X")
    (mem_with_stub.cfg.memory_dir / rec.path).unlink()
    assert mem_with_stub.delete(rec.id) is True
    assert mem_with_stub.store.count() == 0


def test_delete_succeeds_when_graph_cleanup_fails(mem_with_stub: Memory, monkeypatch):
    """graph.drop_for_memoria runs after the point of no return (index dropped,
    .md unlinked): a graph-sidecar failure (e.g. locked graph.db) must not make
    the already-completed delete report failure. Edges are rebuildable via reindex."""
    rec = mem_with_stub.save(content="borrar con grafo roto", title="X")

    def _boom(*a, **k):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(mem_with_stub.graph, "drop_for_memoria", _boom)
    assert mem_with_stub.delete(rec.id) is True
    assert mem_with_stub.store.count() == 0
    assert not (mem_with_stub.cfg.memory_dir / rec.path).is_file()


def test_update_skips_reembed_for_pure_retag(mem_with_stub: Memory, monkeypatch):
    rec = mem_with_stub.save(content="cuerpo", title="orig", type_="note", tags=["x"])
    calls: list[int] = []
    orig = mem_with_stub.embedder.embed

    def _spy(inputs):
        calls.append(len(inputs))
        return orig(inputs)

    monkeypatch.setattr(mem_with_stub.embedder, "embed", _spy)
    updated = mem_with_stub.update(rec.id, type_="decision", tags=["y", "Z"])
    assert updated is not None
    assert updated.type == "decision"
    assert updated.tags == ["y", "z"]
    assert updated.title == "orig"
    assert updated.body == "cuerpo"
    assert calls == []


def test_update_reembeds_when_title_changes(mem_with_stub: Memory, monkeypatch):
    rec = mem_with_stub.save(content="cuerpo", title="orig", type_="note")
    calls: list[int] = []
    orig = mem_with_stub.embedder.embed

    def _spy(inputs):
        calls.append(len(inputs))
        return orig(inputs)

    monkeypatch.setattr(mem_with_stub.embedder, "embed", _spy)
    updated = mem_with_stub.update(rec.id, title="renamed")
    assert updated is not None
    assert updated.title == "renamed"
    assert calls == [1]


def test_update_reembeds_when_content_changes(mem_with_stub: Memory, monkeypatch):
    rec = mem_with_stub.save(content="cuerpo viejo", title="X")
    calls: list[int] = []
    orig = mem_with_stub.embedder.embed

    def _spy(inputs):
        calls.append(len(inputs))
        return orig(inputs)

    monkeypatch.setattr(mem_with_stub.embedder, "embed", _spy)
    updated = mem_with_stub.update(rec.id, content="cuerpo nuevo y diferente")
    assert updated is not None
    assert updated.body == "cuerpo nuevo y diferente"
    assert calls == [1]
    on_disk = (mem_with_stub.cfg.memory_dir / updated.path).read_text()
    assert "cuerpo nuevo y diferente" in on_disk


def test_update_missing_returns_none(mem_with_stub: Memory):
    assert mem_with_stub.update("nope", title="x") is None


def test_update_rejects_invalid_type(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="x", title="X")
    with pytest.raises(ValueError, match="not in valid set"):
        mem_with_stub.update(rec.id, type_="bogus")


def test_get_by_unique_prefix(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="x", title="X")
    short = rec.id[:7]
    fetched = mem_with_stub.get(short)
    assert fetched is not None
    assert fetched.id == rec.id


def test_get_unknown_prefix_returns_none(mem_with_stub: Memory):
    mem_with_stub.save(content="x", title="X")
    assert mem_with_stub.get("ffffffff") is None


def test_get_ambiguous_prefix_raises(mem_with_stub: Memory, monkeypatch):
    import uuid

    fixed = iter(
        [
            uuid.UUID("aaaaaaaa1111000000000000000000ff"),
            uuid.UUID("aaaaaaaa2222000000000000000000ff"),
        ]
    )
    monkeypatch.setattr("memo.memory.uuid.uuid4", lambda: next(fixed))
    mem_with_stub.save(content="a", title="A")
    mem_with_stub.save(content="b", title="B")
    with pytest.raises(AmbiguousIdError) as exc_info:
        mem_with_stub.get("aaaaaaaa")
    assert len(exc_info.value.matches) == 2


def test_update_and_delete_accept_prefix(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="x", title="X")
    short = rec.id[:6]
    updated = mem_with_stub.update(short, title="X2")
    assert updated is not None
    assert updated.title == "X2"
    assert mem_with_stub.delete(short) is True


# ---------------------------------------------------------------------------
# last_saved_id — "rename what I just saved" resolution
# ---------------------------------------------------------------------------


def test_last_saved_id_returns_most_recent_save(mem_with_stub: Memory):
    mem_with_stub.save(content="primero", title="A")
    b = mem_with_stub.save(content="segundo", title="B")
    assert mem_with_stub.last_saved_id() == b.id


def test_last_saved_id_skips_deleted(mem_with_stub: Memory):
    a = mem_with_stub.save(content="primero", title="A")
    b = mem_with_stub.save(content="segundo", title="B")
    mem_with_stub.delete(b.id)
    assert mem_with_stub.last_saved_id() == a.id


def test_last_saved_id_none_when_no_saves(mem_with_stub: Memory):
    assert mem_with_stub.last_saved_id() is None


def test_last_saved_id_ignores_other_device_events(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="local", title="Local")
    # Simulate a synced save event from another machine, newer than ours.
    other = mem_with_stub.history
    own_device = other.device_id
    other.device_id = "other-device"
    try:
        other.log_save(
            ts="2099-01-01T00:00:00+00:00",
            record_id="deadbeefdeadbeefdeadbeefdeadbeef",
            title="Remote",
            type_="note",
        )
    finally:
        other.device_id = own_device
    assert mem_with_stub.last_saved_id() == rec.id


def test_update_replace_exact_unique(mock_memory):
    rec = mock_memory.save(content="port is 8080 and host is local", title="Cfg")
    out = mock_memory.update(rec.id, replace=("8080", "9090"))
    assert "port is 9090" in out.body
    assert "host is local" in out.body  # untouched text preserved byte-identical


def test_update_replace_not_found_raises(mock_memory):
    rec = mock_memory.save(content="alpha", title="R1")
    with pytest.raises(ValueError, match="not found"):
        mock_memory.update(rec.id, replace=("beta", "gamma"))


def test_update_replace_ambiguous_raises(mock_memory):
    rec = mock_memory.save(content="x y x", title="R2")
    with pytest.raises(ValueError, match="2 times"):
        mock_memory.update(rec.id, replace=("x", "z"))


def test_update_append_adds_paragraph(mock_memory):
    rec = mock_memory.save(content="first", title="R3")
    out = mock_memory.update(rec.id, append="second")
    assert out.body == "first\n\nsecond"


def test_update_content_and_replace_mutually_exclusive(mock_memory):
    rec = mock_memory.save(content="a", title="R4")
    with pytest.raises(ValueError, match="at most one"):
        mock_memory.update(rec.id, content="b", replace=("a", "c"))


def test_memo_update_replace_params(mock_memory):
    from memo.server_core_records import register

    tools: dict = {}

    class _Srv:
        def tool(self, *a, **k):
            def wrap(fn):
                tools[fn.__name__] = fn
                return fn

            return wrap

    register(_Srv(), mock_memory)
    rec = mock_memory.save(content="version 1 of the fact", title="F")
    out = tools["memo_update"](id=rec.id, replace_old="version 1", replace_new="version 2")
    assert "version 2" in out["body"]
    bad = tools["memo_update"](id=rec.id, replace_old="missing", replace_new="x")
    assert bad["error"] == "edit_failed"
    half = tools["memo_update"](id=rec.id, replace_old="x")
    assert half["error"] == "replace_incomplete"


def test_update_migrates_legacy_vault_copy(mem_with_stub: Memory):
    """update() on a legacy vault-layout row must leave exactly ONE canonical
    .md: the rewritten `memory_dir` copy. Regression: the stale `vault_path`
    duplicate used to be left in place, resurrecting the pre-update body on a
    later reindex."""
    rec = mem_with_stub.save(content="legacy body", title="Legacy migrate")
    canonical_path = mem_with_stub.cfg.memory_dir / rec.path
    assert mem_with_stub.cfg.vault_path is not None
    legacy_path = mem_with_stub.cfg.vault_path / rec.path
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.replace(legacy_path)
    assert not canonical_path.exists()

    updated = mem_with_stub.update(rec.id, append="nuevo parrafo")
    assert updated is not None
    assert "nuevo parrafo" in updated.body
    # New-layout file carries the new content; stale vault copy is gone.
    assert canonical_path.is_file()
    assert "nuevo parrafo" in canonical_path.read_text(encoding="utf-8")
    assert not legacy_path.exists()
    fetched = mem_with_stub.get(rec.id)
    assert fetched is not None
    assert "nuevo parrafo" in fetched.body


def test_append_refuses_when_the_canonical_body_cannot_be_read(mem_with_stub: Memory, monkeypatch):
    """An unreadable .md must abort the edit, not overwrite it with the fragment.

    `_read_body` falls back to the FTS body and then to "" — fine for a search
    snippet, catastrophic for `append=`, which derives the new body from the old
    one: the canonical file used to be rewritten with just the appended text,
    and the pre-update snapshot recorded body='' so version rollback could not
    recover it either.
    """
    from memo.errors import StorageError

    rec = mem_with_stub.save(content="PARA-1 original\n\nPARA-2 original", title="Larga")
    md = mem_with_stub.cfg.memory_dir / rec.path
    original = md.read_text(encoding="utf-8")

    real_read_text = type(md).read_text

    def _boom(self, *a, **k):
        if self.name == md.name:
            raise OSError("permission denied")
        return real_read_text(self, *a, **k)

    monkeypatch.setattr("pathlib.Path.read_text", _boom)
    with pytest.raises(StorageError, match="cannot read the canonical body"):
        mem_with_stub.update(rec.id, append="APPENDED-ONLY")

    monkeypatch.undo()
    assert md.read_text(encoding="utf-8") == original  # untouched


def test_append_refuses_when_the_canonical_file_is_gone_and_index_has_no_body(
    mem_with_stub: Memory,
):
    from memo.errors import StorageError

    rec = mem_with_stub.save(content="PARA-1 original", title="Movida")
    (mem_with_stub.cfg.memory_dir / rec.path).unlink()
    mem_with_stub.store._conn.execute("UPDATE fts SET body = NULL WHERE id = ?", (rec.id,))
    mem_with_stub.store._conn.commit()

    with pytest.raises(StorageError, match="refusing to rewrite"):
        mem_with_stub.update(rec.id, append="APPENDED-ONLY")


def test_list_returns_the_full_page_despite_forgotten_rows(mem_with_stub: Memory):
    """The forgotten filter runs after the SQL LIMIT — it must refill, not shrink.

    A short page reads as "the corpus only holds this many", with no signal that
    anything was filtered.
    """
    ids = [mem_with_stub.save(content=f"cuerpo {i}", title=f"N{i}").id for i in range(10)]
    for id_ in ids[:5]:
        mem_with_stub.forget(id_)

    got = mem_with_stub.list(limit=5)

    assert len(got) == 5
    assert not set(r.id for r in got) & set(ids[:5])
