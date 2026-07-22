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

    def search(self, query, limit, mode="vec", exclude_types=None, exclude_tags=None):
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
        dt,
        "build_labels",
        lambda *a, **k: (
            LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["aaaa1111"])]),
            True,
        ),
    )
    res = dt.run_tuning_pass(
        tmp_cfg, _StubMem(), k=3, max_evals=20, min_used_score=0.5, dry_run=False
    )
    assert res["status"] == "noop"


def test_run_tuning_pass_applies_when_detuned(tmp_cfg, monkeypatch):
    monkeypatch.setattr(
        dt,
        "build_labels",
        lambda *a, **k: (
            LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["aaaa1111"])]),
            True,
        ),
    )
    # pretend the live min_sim is detuned high (0.95) so the search finds a better floor
    monkeypatch.setattr("memo.flags.flag_float", lambda name, **kw: 0.95)
    res = dt.run_tuning_pass(
        tmp_cfg, _StubMem(), k=3, max_evals=20, min_used_score=0.5, dry_run=False
    )
    assert res["status"] == "applied"
    assert res["floor_after"] < 0.95
    # overlay was written
    assert dt.read_overlay(tmp_cfg.state_dir)["MEMO_RECALL_MIN_SIM"] == res["floor_after"]


def test_run_tuning_pass_dry_run_writes_nothing(tmp_cfg, monkeypatch):
    monkeypatch.setattr(
        dt,
        "build_labels",
        lambda *a, **k: (
            LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["aaaa1111"])]),
            True,
        ),
    )
    monkeypatch.setattr("memo.flags.flag_float", lambda name, **kw: 0.95)
    res = dt.run_tuning_pass(
        tmp_cfg, _StubMem(), k=3, max_evals=20, min_used_score=0.5, dry_run=True
    )
    assert res["status"] == "would_apply"
    assert dt.read_overlay(tmp_cfg.state_dir) == {}  # nothing written


# --- curated graph-signal tuning --------------------------------------------

_GRAPH_SCORES = {
    (False, 0.0): 0.1,
    (True, 0.1): 0.15,
    (True, 0.15): 0.3,
    (True, 0.25): 0.2,
}


def _stub_graph_measure(mem, labels, *, k, enabled, alpha, floor):
    return {
        "precision_at_k": _GRAPH_SCORES.get((enabled, round(alpha, 4)), 0.0),
        "noise_at_k": 0.0,
    }


def test_graph_signal_candidates_only_use_curated_signal_flags():
    candidates = dt.graph_signal_candidates()
    assert candidates == [
        {"MEMO_GRAPH_SIGNAL_ENABLED": False, "MEMO_GRAPH_SIGNAL_ALPHA": 0.0},
        {"MEMO_GRAPH_SIGNAL_ENABLED": True, "MEMO_GRAPH_SIGNAL_ALPHA": 0.10},
        {"MEMO_GRAPH_SIGNAL_ENABLED": True, "MEMO_GRAPH_SIGNAL_ALPHA": 0.15},
        {"MEMO_GRAPH_SIGNAL_ENABLED": True, "MEMO_GRAPH_SIGNAL_ALPHA": 0.25},
    ]
    assert all("MEMO_GRAPH_RETRIEVAL_ENABLED" not in item for item in candidates)
    assert all("MEMO_GRAPH_EXPANSION_ENABLED" not in item for item in candidates)


def test_search_graph_signal_selects_max(monkeypatch):
    monkeypatch.setattr(dt, "measure_graph_signal", _stub_graph_measure)
    labels = LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["x"])])
    best, before, after = dt.search_graph_signal(
        _StubMem(),
        labels,
        k=3,
        current={"MEMO_GRAPH_SIGNAL_ENABLED": False, "MEMO_GRAPH_SIGNAL_ALPHA": 0.0},
        floor=0.5,
        candidates=dt.graph_signal_candidates(),
        max_evals=20,
    )
    assert best == {"MEMO_GRAPH_SIGNAL_ENABLED": True, "MEMO_GRAPH_SIGNAL_ALPHA": 0.15}
    assert after["precision_at_k"] == 0.3
    assert before["precision_at_k"] == 0.1


def test_run_graph_signal_pass_applies_and_writes_overlay(tmp_cfg, monkeypatch):
    monkeypatch.setattr(
        dt,
        "build_labels",
        lambda *a, **k: (LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["x"])]), True),
    )
    monkeypatch.setattr(dt, "measure_graph_signal", _stub_graph_measure)
    res = dt.run_graph_weight_pass(tmp_cfg, _StubMem(), k=3, max_evals=20, dry_run=False)
    assert res["status"] == "applied"
    assert res["config_after"]["MEMO_GRAPH_SIGNAL_ALPHA"] == 0.15
    overlay = dt.read_overlay(tmp_cfg.state_dir)
    assert overlay["MEMO_GRAPH_SIGNAL_ENABLED"] is True
    assert overlay["MEMO_GRAPH_SIGNAL_ALPHA"] == 0.15


def test_run_graph_weight_pass_dry_run_writes_nothing(tmp_cfg, monkeypatch):
    monkeypatch.setattr(
        dt,
        "build_labels",
        lambda *a, **k: (LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["x"])]), True),
    )
    monkeypatch.setattr(dt, "measure_graph_signal", _stub_graph_measure)
    res = dt.run_graph_weight_pass(tmp_cfg, _StubMem(), k=3, max_evals=20, dry_run=True)
    assert res["status"] == "would_apply"
    assert dt.read_overlay(tmp_cfg.state_dir) == {}


def test_run_graph_weight_pass_preserves_existing_overlay_params(tmp_cfg, monkeypatch):
    # A prior min_sim tune wrote the overlay; the graph pass must not clobber it.
    dt.write_overlay(tmp_cfg.state_dir, {"MEMO_RECALL_MIN_SIM": 0.6}, {"set_by": "dream"})
    monkeypatch.setattr(
        dt,
        "build_labels",
        lambda *a, **k: (LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["x"])]), True),
    )
    monkeypatch.setattr(dt, "measure_graph_signal", _stub_graph_measure)
    res = dt.run_graph_weight_pass(tmp_cfg, _StubMem(), k=3, max_evals=20, dry_run=False)
    assert res["status"] == "applied"
    overlay = dt.read_overlay(tmp_cfg.state_dir)
    assert overlay["MEMO_RECALL_MIN_SIM"] == 0.6  # preserved
    assert overlay["MEMO_GRAPH_SIGNAL_ENABLED"] is True
    assert overlay["MEMO_GRAPH_SIGNAL_ALPHA"] == 0.15


def test_run_tuning_pass_preserves_existing_overlay_params(tmp_cfg, monkeypatch):
    # Symmetric to the graph-pass test: a prior graph-weight tune wrote the
    # overlay; the min_sim pass must merge, not clobber it (else the next night
    # silently disables the graph boost).
    dt.write_overlay(
        tmp_cfg.state_dir,
        {"MEMO_GRAPH_SIGNAL_ENABLED": True, "MEMO_GRAPH_SIGNAL_ALPHA": 0.15},
        {"set_by": "dream-graph"},
    )
    monkeypatch.setattr(
        dt,
        "build_labels",
        lambda *a, **k: (
            LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["aaaa1111"])]),
            True,
        ),
    )
    monkeypatch.setattr("memo.flags.flag_float", lambda name, **kw: 0.95)  # detuned -> applies
    res = dt.run_tuning_pass(tmp_cfg, _StubMem(), k=3, max_evals=20, dry_run=False)
    assert res["status"] == "applied"
    overlay = dt.read_overlay(tmp_cfg.state_dir)
    assert overlay["MEMO_GRAPH_SIGNAL_ENABLED"] is True
    assert overlay["MEMO_GRAPH_SIGNAL_ALPHA"] == 0.15
    assert overlay["MEMO_RECALL_MIN_SIM"] == res["floor_after"]


def test_retired_graph_retrieval_pass_never_mutates_overlay(tmp_cfg):
    dt.write_overlay(tmp_cfg.state_dir, {"MEMO_RECALL_MIN_SIM": 0.6}, {"set_by": "test"})
    before = dt.read_overlay(tmp_cfg.state_dir)
    res = dt.run_graph_retrieval_pass(tmp_cfg, _StubMem())
    assert res == {"status": "retired", "replacement": "curated_graph_signal"}
    assert dt.read_overlay(tmp_cfg.state_dir) == before


def test_build_labels_merges_negative_verdicts(tmp_cfg) -> None:
    from types import SimpleNamespace

    from memo.dashboard import append_verdict_log
    from memo.dream_tune import build_labels

    append_verdict_log(
        tmp_cfg.state_dir,
        session_id="s1",
        turn=4,
        prior_turn=3,
        verdict="correction",
        prompt="dónde vive el registro de flags?",
        reaction="no, eso está mal",
        recall_ids=["aaaabbbb11112222"],
    )
    labels, _ = build_labels(SimpleNamespace(state_dir=tmp_cfg.state_dir))
    neg = [p for p in labels.prompts if p.avoid_ids]
    assert neg and neg[0].avoid_ids == ["aaaabbbb"]
