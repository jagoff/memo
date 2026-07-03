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


def test_delete_proceeds_when_md_already_missing(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="huérfano", title="X")
    (mem_with_stub.cfg.memory_dir / rec.path).unlink()
    assert mem_with_stub.delete(rec.id) is True
    assert mem_with_stub.store.count() == 0


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
