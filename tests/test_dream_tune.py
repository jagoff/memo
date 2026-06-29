"""dream_tune — measurement, label building, line search, apply/rollback."""

from __future__ import annotations

from memo import dream_tune as dt
from memo.eval_recall import LabelSet, Prompt


class _Hit:
    def __init__(self, id, score, title="t", tags=None, path="p", body="some body text"):
        self.id, self.score, self.title = id, score, title
        self.tags, self.path, self.body = tags or [], path, body


class _StubMem:
    """One relevant hit (aaaa1111 @0.9), one weaker non-relevant (bbbb2222 @0.5)."""

    def search(self, query, limit, mode="vec"):
        return [_Hit("aaaa1111", 0.9), _Hit("bbbb2222", 0.5)]


def test_measure_precision_counts_id_match():
    labels = LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["aaaa1111"])])
    m = dt.measure(_StubMem(), labels, k=3, floor=0.7)
    assert m["precision_at_k"] > 0


def test_baseline_roundtrip(tmp_path):
    dt.save_baseline(tmp_path, {"precision_at_k": 0.2, "noise_at_k": 0.0})
    assert dt.load_baseline(tmp_path)["precision_at_k"] == 0.2


def test_load_baseline_missing_is_none(tmp_path):
    assert dt.load_baseline(tmp_path) is None


def test_build_labels_no_grounding_is_safe(tmp_cfg):
    labels, _curated_used = dt.build_labels(tmp_cfg, min_used_score=0.5, limit=10)
    assert isinstance(labels.prompts, list)  # empty grounding log must not raise


def test_search_recovers_detuned_floor():
    labels = LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["aaaa1111"])])
    best, before, after = dt.search_min_sim(
        _StubMem(), labels, k=3, current=0.95, lo=0.5, hi=0.95, step=0.05, max_evals=20
    )
    assert after["precision_at_k"] >= before["precision_at_k"]
    assert before["precision_at_k"] == 0.0  # floor 0.95 dropped the 0.9 hit
    assert best <= 0.9


def test_run_tuning_pass_noop_when_already_optimal(tmp_cfg, monkeypatch):
    monkeypatch.setattr(
        dt, "build_labels",
        lambda *a, **k: (LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["aaaa1111"])]), True),
    )
    res = dt.run_tuning_pass(tmp_cfg, _StubMem(), k=3, max_evals=20, min_used_score=0.5, dry_run=False)
    assert res["status"] == "noop"


def test_run_tuning_pass_applies_when_detuned(tmp_cfg, monkeypatch):
    monkeypatch.setattr(
        dt, "build_labels",
        lambda *a, **k: (LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["aaaa1111"])]), True),
    )
    # pretend the live min_sim is detuned high (0.95) so the search finds a better floor
    monkeypatch.setattr("memo.flags.flag_float", lambda name, **kw: 0.95)
    res = dt.run_tuning_pass(tmp_cfg, _StubMem(), k=3, max_evals=20, min_used_score=0.5, dry_run=False)
    assert res["status"] == "applied"
    assert res["floor_after"] < 0.95
    # overlay was written
    assert dt.read_overlay(tmp_cfg.state_dir)["MEMO_RECALL_MIN_SIM"] == res["floor_after"]


def test_run_tuning_pass_dry_run_writes_nothing(tmp_cfg, monkeypatch):
    monkeypatch.setattr(
        dt, "build_labels",
        lambda *a, **k: (LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["aaaa1111"])]), True),
    )
    monkeypatch.setattr("memo.flags.flag_float", lambda name, **kw: 0.95)
    res = dt.run_tuning_pass(tmp_cfg, _StubMem(), k=3, max_evals=20, min_used_score=0.5, dry_run=True)
    assert res["status"] == "would_apply"
    assert dt.read_overlay(tmp_cfg.state_dir) == {}  # nothing written
