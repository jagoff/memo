"""support_count — corroboration counter in memory_health (workstream C1).

Store-layer contract: bump/get batch APIs, inline migration on legacy DBs,
and cross-machine signal export/import carrying the counter.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from memo.store.store import VecStore


def _make_store(tmp_path: Path) -> VecStore:
    return VecStore(tmp_path / "test.db", dims=4)


def test_bump_support_batch_upserts_and_increments(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.bump_support_batch(["m1"])
    store.bump_support_batch(["m1", "m2"])
    assert store.get_support_batch(["m1", "m2"]) == {"m1": 2, "m2": 1}


def test_bump_support_batch_empty_is_noop(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.bump_support_batch([])  # must not raise
    assert store.get_support_batch([]) == {}


def test_get_support_batch_missing_ids_are_absent(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    assert store.get_support_batch(["nope"]) == {}


def test_repeated_id_in_one_batch_bumps_n_times(tmp_path: Path) -> None:
    # consolidation passes [surviving_id] * n_archived
    store = _make_store(tmp_path)
    store.bump_support_batch(["m1", "m1", "m1"])
    assert store.get_support_batch(["m1"]) == {"m1": 3}


def test_inline_migration_adds_support_count_to_legacy_db(tmp_path: Path) -> None:
    db = tmp_path / "legacy.db"
    cx = sqlite3.connect(db)
    cx.execute(
        "CREATE TABLE memory_health ("
        "id TEXT PRIMARY KEY, confidence REAL NOT NULL DEFAULT 1.0, "
        "roi_score REAL NOT NULL DEFAULT 1.0, updated_at TEXT)"
    )
    cx.execute("INSERT INTO memory_health VALUES ('legacyid', 0.7, 1.2, '2026-01-01')")
    cx.commit()
    cx.close()

    store = VecStore(db, dims=4)  # init runs DDL + inline migrations
    assert store.get_support_batch(["legacyid"]) == {"legacyid": 0}


def test_signal_roundtrip_carries_support_count(tmp_path: Path) -> None:
    a = VecStore(tmp_path / "a.db", dims=4)
    b = VecStore(tmp_path / "b.db", dims=4)
    a.bump_support_batch(["m1"])
    a.bump_support_batch(["m1"])
    payload = a.dump_signal()
    row = next(r for r in payload["memory_health"] if r["id"] == "m1")
    assert row["support_count"] == 2
    b.merge_signal(payload)
    assert b.get_support_batch(["m1"]) == {"m1": 2}


def test_merge_signal_tolerates_old_payload_without_support(tmp_path: Path) -> None:
    b = VecStore(tmp_path / "b.db", dims=4)
    b.merge_signal(
        {
            "access": [],
            "memory_health": [
                {"id": "m1", "confidence": 0.9, "roi_score": 1.1, "updated_at": "2026-01-01"}
            ],
            "source_feedback": [],
        }
    )
    assert b.get_support_batch(["m1"]) == {"m1": 0}


import pytest

from memo.config import Config


@pytest.fixture
def mem_const(tmp_cfg, monkeypatch):
    """Memory with a constant-vector embedder: every text embeds to the same
    unit vector, so ANY two records are cosine-1.0 near-duplicates — the
    near-dup path (threshold 0.88) fires deterministically. Dims pinned to
    the stub's output via Config(embedder_dims=4)."""
    from memo.memory import Memory

    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
        reranker_enabled=False,
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    mem = Memory(cfg)
    yield mem
    mem.close()


def test_near_dup_save_bumps_existing_support(mem_const):
    r1 = mem_const.save(content="El dashboard corre en el puerto 8765", title="Dashboard port")
    r2 = mem_const.save(content="El dashboard escucha en 8765", title="Dashboard listens")
    assert r2.id != r1.id  # near-dup still only warns; record is created
    assert mem_const.store.get_support_batch([r1.id]) == {r1.id: 1}


def test_near_dup_bump_disabled_by_flag(mem_const, monkeypatch):
    monkeypatch.setenv("MEMO_SUPPORT_COUNT", "0")
    r1 = mem_const.save(content="El dashboard corre en el puerto 8765", title="Dashboard port")
    mem_const.save(content="El dashboard escucha en 8765", title="Dashboard listens")
    # Store auto-creates a memory_health row (support_count=0) on index; assert no bump occurred.
    assert mem_const.store.get_support_batch([r1.id]).get(r1.id, 0) == 0


def test_topic_key_upsert_bumps_support(mock_memory):
    r1 = mock_memory.save(
        content="El dashboard corre en :8765", title="Dashboard", topic_key="dashboard-port"
    )
    r2 = mock_memory.save(
        content="Confirmado: :8765", title="Dashboard", topic_key="dashboard-port"
    )
    assert r2.id == r1.id  # upsert reused the record
    assert mock_memory.store.get_support_batch([r1.id]) == {r1.id: 1}


def test_apply_merge_bumps_surviving_support(mock_memory):
    from memo.consolidation import AdvancedConsolidator, MergeProposal

    r1 = mock_memory.save(content="dato sobre puertos A", title="A")
    r2 = mock_memory.save(content="dato sobre puertos A bis", title="A bis")
    cons = AdvancedConsolidator(mock_memory)
    proposal = MergeProposal(
        cluster_id=0,
        memory_ids=[r1.id, r2.id],
        merged_title="",
        merged_body="",
        merge_strategy="keep_latest",
        rationale="dup",
        archived_ids=[r1.id],
    )
    res = cons.apply_merge(proposal)
    assert res.merged_id == r2.id
    assert res.archived_ids == [r1.id]
    assert mock_memory.store.get_support_batch([r2.id]) == {r2.id: 1}


# -- Task 3: bounded confidence lift from corroboration -------------------


def test_lift_restores_penalized_confidence_bounded(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.penalize_confidence_batch(["m1"], delta=0.4)  # 1.0 -> 0.6
    store.bump_support_batch(["m1"], lift=0.05)  # 0.6 -> 0.65
    assert store.get_health_batch(["m1"])["m1"]["confidence"] == pytest.approx(0.65)
    for _ in range(20):
        store.bump_support_batch(["m1"], lift=0.05)
    # capped at 1.0 — corroboration never boosts ABOVE neutral
    assert store.get_health_batch(["m1"])["m1"]["confidence"] == pytest.approx(1.0)


def test_lift_zero_default_leaves_confidence_untouched(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.penalize_confidence_batch(["m1"], delta=0.4)
    store.bump_support_batch(["m1"])  # default lift=0.0
    assert store.get_health_batch(["m1"])["m1"]["confidence"] == pytest.approx(0.6)


def test_support_lift_reads_flag(monkeypatch) -> None:
    from memo.memory.record import support_lift

    assert support_lift() == 0.0  # default off
    monkeypatch.setenv("MEMO_SUPPORT_CONFIDENCE_LIFT", "0.05")
    assert support_lift() == pytest.approx(0.05)
