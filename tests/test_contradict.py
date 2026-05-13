"""Tests for the persistent contradiction sidecar + corpus scanner.

These tests stub `MLXEmbedder.embed` and `TemporalAnalyzer._classify_pair`
so they run anywhere — no Apple Silicon needed.
"""

from __future__ import annotations

import pytest

from memo.config import Config
from memo.contradict import (
    ContradictionScanner,
    ContradictionStore,
    PairRecord,
    VALID_STATUSES,
    _canonical_pair,
    is_stale,
)
from memo.memory import Memory
from memo.temporal import Contradiction, TemporalAnalyzer


# -- ContradictionStore unit tests ---------------------------------------------


@pytest.fixture
def store(tmp_path):
    return ContradictionStore(tmp_path / "contradictions.db")


def test_canonical_pair_is_order_independent():
    a, b = _canonical_pair("zzz", "aaa")
    assert a == "aaa"
    assert b == "zzz"
    assert _canonical_pair("aaa", "zzz") == ("aaa", "zzz")


def test_upsert_open_inserts_and_returns_id(store):
    pid = store.upsert_open(
        "aaa", "bbb", "contradiction", 0.9, "rationale text",
    )
    assert pid > 0
    rows = store.list_open()
    assert len(rows) == 1
    assert rows[0].memoria_id_a == "aaa"
    assert rows[0].memoria_id_b == "bbb"
    assert rows[0].relationship == "contradiction"
    assert rows[0].confidence == 0.9
    assert rows[0].status == "open"


def test_upsert_open_dedupes_canonical(store):
    pid1 = store.upsert_open("bbb", "aaa", "contradiction", 0.8, "first")
    pid2 = store.upsert_open("aaa", "bbb", "contradiction", 0.9, "second")
    assert pid1 == pid2
    rows = store.list_open()
    assert len(rows) == 1
    assert rows[0].confidence == 0.9
    assert rows[0].rationale == "second"


def test_resolve_marks_pair_and_records_note(store):
    pid = store.upsert_open("aaa", "bbb", "contradiction", 0.9, "")
    assert store.resolve(pid, "dismissed", note="false positive") is True
    rec = store.get(pid)
    assert rec is not None
    assert rec.status == "dismissed"
    assert rec.resolution_note == "false positive"
    assert rec.resolved_at is not None


def test_resolve_rejects_invalid_status(store):
    pid = store.upsert_open("aaa", "bbb", "contradiction", 0.9, "")
    with pytest.raises(ValueError):
        store.resolve(pid, "bogus")
    with pytest.raises(ValueError):
        store.resolve(pid, "open")


def test_resolved_pairs_are_not_overwritten_by_rescan(store):
    pid = store.upsert_open("aaa", "bbb", "contradiction", 0.9, "first")
    store.resolve(pid, "dismissed")
    same_pid = store.upsert_open("aaa", "bbb", "contradiction", 0.95, "second-scan")
    assert same_pid == pid
    rec = store.get(pid)
    assert rec.status == "dismissed"
    assert rec.rationale == "first"


def test_already_resolved_flag(store):
    pid = store.upsert_open("aaa", "bbb", "contradiction", 0.9, "")
    assert store.already_resolved("aaa", "bbb") is False
    store.resolve(pid, "evolved")
    assert store.already_resolved("aaa", "bbb") is True
    assert store.already_resolved("bbb", "aaa") is True  # canonical


def test_reopen_restores_open_status(store):
    pid = store.upsert_open("aaa", "bbb", "contradiction", 0.9, "")
    store.resolve(pid, "dismissed")
    assert store.reopen(pid) is True
    rec = store.get(pid)
    assert rec.status == "open"
    assert rec.resolved_at is None
    assert rec.resolution_note is None


def test_drop_for_memoria_cleans_dangling_pairs(store):
    store.upsert_open("aaa", "bbb", "contradiction", 0.9, "")
    store.upsert_open("aaa", "ccc", "contradiction", 0.85, "")
    store.upsert_open("xxx", "yyy", "contradiction", 0.8, "")
    deleted = store.drop_for_memoria("aaa")
    assert deleted == 2
    assert len(store.list_open()) == 1


def test_list_open_filters_by_confidence_and_relationship(store):
    store.upsert_open("a1", "b1", "contradiction", 0.6, "low")
    store.upsert_open("a2", "b2", "contradiction", 0.95, "high")
    store.upsert_open("a3", "b3", "evolution", 0.92, "evo")

    only_high = store.list_open(min_confidence=0.9)
    assert {p.rationale for p in only_high} == {"high", "evo"}

    only_contr = store.list_open(relationship="contradiction")
    assert {p.relationship for p in only_contr} == {"contradiction"}


def test_stats_groups_by_status(store):
    p1 = store.upsert_open("a1", "b1", "contradiction", 0.9, "")
    p2 = store.upsert_open("a2", "b2", "contradiction", 0.9, "")
    store.upsert_open("a3", "b3", "evolution", 0.9, "")
    store.resolve(p1, "dismissed")
    store.resolve(p2, "fused")

    stats = store.stats()
    assert stats.get("open") == 1
    assert stats.get("dismissed") == 1
    assert stats.get("fused") == 1


def test_valid_statuses_does_not_include_open_as_resolution():
    # `open` is the initial state — resolve() rejects it, but the
    # canonical set still contains it (so list/filter calls accept it).
    assert "open" in VALID_STATUSES
    assert "dismissed" in VALID_STATUSES


def test_is_stale_helper():
    assert is_stale("2020-01-01T00:00:00+00:00", days_threshold=30) is True
    assert is_stale("2999-01-01T00:00:00+00:00", days_threshold=30) is False
    assert is_stale("garbage", days_threshold=30) is False


# -- ContradictionScanner integration tests -----------------------------------


@pytest.fixture
def mem_with_stub_embed(tmp_cfg: Config, monkeypatch) -> Memory:
    """Memory with a small deterministic embedder. Same body → same bucket
    so we can stage near-duplicate clusters."""
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
    )

    def _stub_embed(self, inputs):
        out = []
        for s in inputs:
            # Group by first char so "alpha-1" and "alpha-2" cluster together.
            bucket = (ord((s or " ")[0]) % 4)
            v = [0.0] * 4
            v[bucket] = 1.0
            out.append(v)
        return out

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _stub_embed)
    return Memory(cfg)


def _stage_classify_pair(monkeypatch, verdict: Contradiction | None):
    """Force TemporalAnalyzer._classify_pair to return a fixed verdict.

    Captures the pair of records it sees so tests can assert which pairs
    were actually classified (and which were skipped by the prefilter).
    """
    calls: list[tuple[str, str]] = []

    def _fake_classify(self, r1, r2):  # noqa: ANN001
        calls.append((r1.id, r2.id))
        if verdict is None:
            return None
        # Update the verdict's IDs to the actual pair so the store
        # gets canonicalized identifiers (mirrors real classifier).
        return Contradiction(
            memoria_id_a=r1.id,
            memoria_id_b=r2.id,
            title_a=r1.title,
            title_b=r2.title,
            date_a=r1.updated,
            date_b=r2.updated,
            relationship=verdict.relationship,
            rationale=verdict.rationale,
            confidence=verdict.confidence,
        )

    monkeypatch.setattr(TemporalAnalyzer, "_classify_pair", _fake_classify)
    return calls


def test_scan_persists_contradiction_pairs(mem_with_stub_embed, monkeypatch):
    mem = mem_with_stub_embed
    mem.save(content="alpha original — uso Ollama local", title="Stack A", type_="decision")
    mem.save(content="alpha actualizado — migré a MLX",  title="Stack B", type_="decision")

    verdict = Contradiction(
        memoria_id_a="x", memoria_id_b="y", title_a="", title_b="",
        date_a="", date_b="",
        relationship="contradiction", rationale="Ollama vs MLX",
        confidence=0.9,
    )
    _stage_classify_pair(monkeypatch, verdict)

    result = mem.contradict_scanner.scan_corpus(
        top_k=3, sim_floor=0.0, confidence_threshold=0.7, min_days_apart=0,
    )
    assert result.contradictions_found >= 1
    assert result.pairs_inserted >= 1

    pairs = mem.contradict_store.list_open()
    assert len(pairs) >= 1
    assert all(p.relationship == "contradiction" for p in pairs)


def test_scan_skips_pairs_already_resolved(mem_with_stub_embed, monkeypatch):
    mem = mem_with_stub_embed
    rec_a = mem.save(content="alpha v1", title="A", type_="note")
    rec_b = mem.save(content="alpha v2", title="B", type_="note")

    # Pre-resolve the pair as dismissed.
    pid = mem.contradict_store.upsert_open(
        rec_a.id, rec_b.id, "contradiction", 0.9, "seed",
    )
    mem.contradict_store.resolve(pid, "dismissed", note="seeded")

    verdict = Contradiction(
        memoria_id_a="x", memoria_id_b="y", title_a="", title_b="",
        date_a="", date_b="",
        relationship="contradiction", rationale="should not appear",
        confidence=0.99,
    )
    calls = _stage_classify_pair(monkeypatch, verdict)

    result = mem.contradict_scanner.scan_corpus(
        top_k=3, sim_floor=0.0, confidence_threshold=0.7, min_days_apart=0,
    )
    # The classifier should never have been asked about this pair.
    pair_key = tuple(sorted((rec_a.id, rec_b.id)))
    seen = {tuple(sorted(c)) for c in calls}
    assert pair_key not in seen
    assert result.pairs_skipped_resolved >= 1


def test_scan_respects_confidence_threshold(mem_with_stub_embed, monkeypatch):
    mem = mem_with_stub_embed
    mem.save(content="alpha 1", title="A", type_="note")
    mem.save(content="alpha 2", title="B", type_="note")

    low_conf = Contradiction(
        memoria_id_a="x", memoria_id_b="y", title_a="", title_b="",
        date_a="", date_b="",
        relationship="contradiction", rationale="weak",
        confidence=0.4,
    )
    _stage_classify_pair(monkeypatch, low_conf)

    result = mem.contradict_scanner.scan_corpus(
        top_k=3, sim_floor=0.0, confidence_threshold=0.7, min_days_apart=0,
    )
    assert result.contradictions_found == 0
    assert mem.contradict_store.list_open() == []


def test_scan_ignores_unrelated_verdicts(mem_with_stub_embed, monkeypatch):
    mem = mem_with_stub_embed
    mem.save(content="alpha 1", title="A", type_="note")
    mem.save(content="alpha 2", title="B", type_="note")

    # `_classify_pair` already filters out non-contradiction/non-evolution,
    # but the scanner also guards. Returning None here mirrors that.
    _stage_classify_pair(monkeypatch, None)

    result = mem.contradict_scanner.scan_corpus(
        top_k=3, sim_floor=0.0, confidence_threshold=0.7, min_days_apart=0,
    )
    assert result.contradictions_found == 0
    assert mem.contradict_store.list_open() == []


def test_memory_delete_cleans_orphan_pairs(mem_with_stub_embed):
    mem = mem_with_stub_embed
    rec_a = mem.save(content="alpha", title="A", type_="note")
    rec_b = mem.save(content="alpha bis", title="B", type_="note")

    mem.contradict_store.upsert_open(
        rec_a.id, rec_b.id, "contradiction", 0.9, "seeded",
    )
    assert len(mem.contradict_store.list_open()) == 1

    mem.delete(rec_a.id)
    assert mem.contradict_store.list_open() == []


def test_pair_record_dataclass():
    rec = PairRecord(
        pair_id=1,
        memoria_id_a="aaa",
        memoria_id_b="bbb",
        relationship="contradiction",
        confidence=0.9,
        rationale="why",
        status="open",
        detected_at="2026-05-13T00:00:00+00:00",
        resolved_at=None,
        resolution_note=None,
    )
    assert rec.pair_id == 1
    assert rec.status == "open"
