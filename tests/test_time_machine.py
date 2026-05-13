"""Time-machine reconstruction + diff."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest

from memo.config import Config
from memo.memory import Memory
from memo.time_machine import diff, reconstruct


@pytest.fixture
def mem(tmp_cfg: Config, monkeypatch) -> Memory:
    cfg = Config(
        data_dir=tmp_cfg.data_dir, vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir, embedder_dims=4,
    )

    def _embed(self, inputs):
        return [[0.0, 0.0, 0.0, 1.0]] * len(inputs)

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _embed)
    return Memory(cfg)


def _now() -> datetime:
    return datetime.now(UTC)


def test_empty_corpus_empty_snapshot(mem: Memory) -> None:
    snap = reconstruct(mem, as_of=_now())
    assert len(snap) == 0


def test_save_then_snapshot_at_now(mem: Memory) -> None:
    rec = mem.save(content="hello world", title="Hello", type_="note")
    snap = reconstruct(mem, as_of=_now() + timedelta(seconds=1))
    assert len(snap) == 1
    assert rec.id in snap.records
    assert snap.records[rec.id].title == "Hello"


def test_snapshot_before_save_is_empty(mem: Memory) -> None:
    before = _now()
    time.sleep(0.05)
    mem.save(content="x", title="Late", type_="note")
    snap = reconstruct(mem, as_of=before)
    # The save happened AFTER `before`, so the snapshot should exclude it.
    assert len(snap) == 0


def test_snapshot_after_delete_excludes_record(mem: Memory) -> None:
    rec = mem.save(content="bye", title="Doomed", type_="note")
    time.sleep(0.05)
    mem.delete(rec.id)
    time.sleep(0.05)
    snap = reconstruct(mem, as_of=_now())
    # Deleted before now, so snapshot should NOT contain it.
    assert rec.id not in snap.records


def test_snapshot_between_save_and_delete_includes_record(mem: Memory) -> None:
    rec = mem.save(content="bye", title="Existed", type_="note")
    time.sleep(0.05)
    middle = _now()
    time.sleep(0.05)
    mem.delete(rec.id)
    snap = reconstruct(mem, as_of=middle)
    # Existed between save and delete.
    assert rec.id in snap.records
    # Body unavailable because record was later deleted.
    assert snap.records[rec.id].body_unavailable


def test_snapshot_reverts_title_update(mem: Memory) -> None:
    rec = mem.save(content="x", title="Original Title", type_="note")
    time.sleep(0.05)
    snap_before = _now()
    time.sleep(0.05)
    mem.update(rec.id, title="New Title")
    # Live state has new title.
    live = mem.get(rec.id)
    assert live.title == "New Title"
    # Snapshot before update should still show original.
    snap = reconstruct(mem, as_of=snap_before)
    assert snap.records[rec.id].title == "Original Title"


def test_diff_reports_added(mem: Memory) -> None:
    t0 = _now()
    time.sleep(0.05)
    mem.save(content="first", title="A", type_="note")
    mem.save(content="second", title="B", type_="note")
    time.sleep(0.05)
    d = diff(mem, from_ts=t0, to_ts=_now())
    assert len(d.added) == 2
    titles = {r.title for r in d.added}
    assert titles == {"A", "B"}


def test_diff_reports_removed(mem: Memory) -> None:
    rec = mem.save(content="x", title="ToDelete", type_="note")
    time.sleep(0.05)
    t_before_delete = _now()
    time.sleep(0.05)
    mem.delete(rec.id)
    time.sleep(0.05)
    t_after_delete = _now()
    d = diff(mem, from_ts=t_before_delete, to_ts=t_after_delete)
    assert len(d.removed) == 1
    assert d.removed[0].title == "ToDelete"


def test_diff_reports_updated(mem: Memory) -> None:
    rec = mem.save(content="x", title="V1", type_="note")
    time.sleep(0.05)
    t_before_update = _now()
    time.sleep(0.05)
    mem.update(rec.id, title="V2")
    time.sleep(0.05)
    t_after = _now()
    d = diff(mem, from_ts=t_before_update, to_ts=t_after)
    assert len(d.updated) == 1
    u = d.updated[0]
    assert "title" in u["changed_fields"]
    assert u["before"]["title"] == "V1"
    assert u["after"]["title"] == "V2"


def test_diff_summary(mem: Memory) -> None:
    # Setup: one pre-existing record that will be updated in the diff
    # window, plus records that are added/removed inside it.
    b = mem.save(content="y", title="V1", type_="note")
    time.sleep(0.05)
    t0 = _now()
    time.sleep(0.05)
    mem.save(content="x", title="Added", type_="note")
    mem.update(b.id, title="V2")  # pre-existing record updated
    c = mem.save(content="z", title="Removed", type_="note")
    time.sleep(0.05)
    mem.delete(c.id)
    time.sleep(0.05)
    d = diff(mem, from_ts=t0, to_ts=_now())
    # Added: just "Added" (b existed in from too; c saved+deleted, cancels).
    assert len(d.added) == 1
    # Updated: b (V1 → V2) — existed in both snapshots.
    assert len(d.updated) == 1
    assert d.updated[0]["before"]["title"] == "V1"
    assert d.updated[0]["after"]["title"] == "V2"
    assert "added" in d.summary()


def test_snapshot_accepts_iso_string(mem: Memory) -> None:
    mem.save(content="x", title="t", type_="note")
    time.sleep(0.05)
    iso = _now().isoformat()
    snap = reconstruct(mem, as_of=iso)
    assert len(snap) == 1


def test_snapshot_search_filters_to_snapshot(mem: Memory) -> None:
    rec_old = mem.save(content="hello", title="Old", type_="note")
    time.sleep(0.05)
    snap_before_new = _now()
    time.sleep(0.05)
    rec_new = mem.save(content="hello", title="New", type_="note")
    # Snapshot from before `rec_new` should NOT return it.
    snap = reconstruct(mem, as_of=snap_before_new)
    hits = snap.search("hello", limit=10, mode="vec")
    hit_ids = {h.id for h in hits}
    assert rec_old.id in hit_ids
    assert rec_new.id not in hit_ids
