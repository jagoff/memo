from types import SimpleNamespace

from memo import dream_tune, dream_tune_online
from memo.eval_recall import LabelSet, Prompt


def _cfg(tmp_path):
    return SimpleNamespace(state_dir=tmp_path)


def _one_label():
    return LabelSet(prompts=[Prompt(text="q", relevant=True, expect_ids=[])]), True


def test_awaiting_online_skips_search(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(dream_tune, "build_labels", lambda cfg, **k: _one_label())
    # a pending exists; cohort too small → waiting → must return before measuring
    dream_tune_online.write_pending(tmp_path, {"version_after": "v2", "online_before": 0.5})
    monkeypatch.setattr(dream_tune_online, "online_fraction", lambda sd, v, **k: (0.9, 3))
    # live_version must match version_after so drift-check keeps it "waiting" (not "expired")
    monkeypatch.setattr(dream_tune, "params_version", lambda sd: "v2")

    def _boom(*a, **k):
        raise AssertionError("search/measure must not run while awaiting online")

    monkeypatch.setattr(dream_tune, "search_min_sim", _boom)
    monkeypatch.setattr(dream_tune, "measure", _boom)

    res = dream_tune.run_tuning_pass(_cfg(tmp_path), object(), k=5)
    assert res["status"] == "awaiting_online"
    assert res["online"]["status"] == "waiting"
    assert dream_tune_online.read_pending(tmp_path) is not None


def test_online_reverted_restores_floor_before_and_baseline(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(dream_tune, "build_labels", lambda cfg, **k: _one_label())
    dream_tune_online.write_pending(
        tmp_path,
        {
            "version_before": "v1",
            "version_after": "v2",
            "floor_before": 0.5,
            "online_before": 0.6,
            "offline_before": {"precision_at_k": 0.2, "noise_at_k": 0.0},
        },
    )
    monkeypatch.setattr(dream_tune_online, "online_fraction", lambda sd, v, **k: (0.40, 50))
    calls = {"overlay": None, "baseline": None}
    monkeypatch.setattr(
        dream_tune,
        "write_overlay",
        lambda sd, params, meta: calls.__setitem__("overlay", dict(params)),
    )
    monkeypatch.setattr(dream_tune, "save_baseline", lambda sd, m: calls.__setitem__("baseline", m))
    monkeypatch.setattr(
        dream_tune, "measure", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no measure"))
    )

    res = dream_tune.run_tuning_pass(_cfg(tmp_path), object(), k=5)
    assert res["status"] == "online_reverted"
    assert calls["overlay"]["MEMO_RECALL_MIN_SIM"] == 0.5  # restored floor_before
    assert calls["baseline"] == {"precision_at_k": 0.2, "noise_at_k": 0.0}


def test_apply_writes_pending(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(dream_tune, "build_labels", lambda cfg, **k: _one_label())
    # no pending → resolve returns "none" → proceeds to search
    monkeypatch.setattr(dream_tune, "load_baseline", lambda sd: None)
    before = {"precision_at_k": 0.2, "noise_at_k": 0.0}
    after = {"precision_at_k": 0.3, "noise_at_k": 0.0}
    monkeypatch.setattr(dream_tune, "search_min_sim", lambda *a, **k: (0.65, before, after))
    monkeypatch.setattr(dream_tune, "params_version", lambda sd: "vNEW")
    # record_pending computes version_after via dream_tune_online.params_version;
    # patch both so the pending reflects the mocked version token.
    monkeypatch.setattr(dream_tune_online, "params_version", lambda sd: "vNEW")
    monkeypatch.setattr(dream_tune_online, "online_fraction", lambda sd, v, **k: (0.5, 10))
    monkeypatch.setattr(dream_tune, "curated_gate_min_sim", lambda *a, **k: {"ok": True})

    res = dream_tune.run_tuning_pass(_cfg(tmp_path), object(), k=5)
    assert res["status"] == "applied"
    pend = dream_tune_online.read_pending(tmp_path)
    assert pend is not None
    assert pend["version_after"] == "vNEW"
    assert pend["floor_after"] == 0.65
    assert pend["online_before"] == 0.5


def test_graph_weight_reverted_restores_graph_knob_and_baseline(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(dream_tune, "build_labels", lambda cfg, **k: _one_label())
    # a graph-weight change is pending; its online cohort regressed
    dream_tune_online.write_pending(
        tmp_path,
        {
            "knob": "MEMO_RECALL_GRAPH_PROXIMITY_WEIGHT",
            "version_after": "v2",
            "floor_before": 0.0,
            "online_before": 0.6,
            "offline_before": {"precision_at_k": 0.2, "noise_at_k": 0.0},
        },
    )
    monkeypatch.setattr(dream_tune_online, "online_fraction", lambda sd, v, **k: (0.40, 50))
    calls = {"overlay": None, "graph_baseline": None, "min_baseline": None}
    monkeypatch.setattr(
        dream_tune,
        "write_overlay",
        lambda sd, params, meta: calls.__setitem__("overlay", dict(params)),
    )
    monkeypatch.setattr(
        dream_tune, "save_graph_baseline", lambda sd, m: calls.__setitem__("graph_baseline", m)
    )
    monkeypatch.setattr(
        dream_tune, "save_baseline", lambda sd, m: calls.__setitem__("min_baseline", m)
    )
    monkeypatch.setattr(
        dream_tune, "measure", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no measure"))
    )

    res = dream_tune.run_tuning_pass(_cfg(tmp_path), object(), k=5)
    assert res["status"] == "online_reverted"
    assert calls["overlay"]["MEMO_RECALL_GRAPH_PROXIMITY_WEIGHT"] == 0.0  # graph knob restored
    assert calls["graph_baseline"] == {
        "precision_at_k": 0.2,
        "noise_at_k": 0.0,
    }  # graph baseline restored
    assert calls["min_baseline"] is None  # NOT the min_sim baseline


def test_graph_weight_apply_records_pending(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(dream_tune, "build_labels", lambda cfg, **k: _one_label())
    monkeypatch.setattr(dream_tune, "load_graph_baseline", lambda sd: None)
    before = {"precision_at_k": 0.2, "noise_at_k": 0.0}
    after = {"precision_at_k": 0.3, "noise_at_k": 0.0}
    monkeypatch.setattr(dream_tune, "search_graph_weight", lambda *a, **k: (0.2, before, after))
    monkeypatch.setattr(dream_tune, "write_overlay", lambda sd, params, meta: None)
    monkeypatch.setattr(dream_tune, "save_graph_baseline", lambda sd, m: None)
    monkeypatch.setattr(dream_tune, "params_version", lambda sd: "vGW")
    monkeypatch.setattr(dream_tune_online, "online_fraction", lambda sd, v, **k: (0.5, 10))
    monkeypatch.setattr(dream_tune, "_curated_label_set", lambda sd: None)

    res = dream_tune.run_graph_weight_pass(_cfg(tmp_path), object(), k=5)
    assert res["status"] == "applied"
    pend = dream_tune_online.read_pending(tmp_path)
    assert pend["knob"] == "MEMO_RECALL_GRAPH_PROXIMITY_WEIGHT"
    assert pend["floor_after"] == 0.2


def test_online_revert_pins_prev_so_offline_rollback_is_noop(tmp_path, monkeypatch):
    from memo.tuned_overlay import _scalar_params, read_overlay, rollback_overlay, write_overlay

    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    # good baseline 0.62, then a BAD change 0.7 applied over it (so _meta.prev = 0.62)
    write_overlay(tmp_path, {"MEMO_RECALL_MIN_SIM": 0.62}, {"set_by": "test"})
    write_overlay(tmp_path, {"MEMO_RECALL_MIN_SIM": 0.7}, {"set_by": "test"})
    monkeypatch.setattr(dream_tune, "build_labels", lambda cfg, **k: _one_label())
    dream_tune_online.write_pending(
        tmp_path,
        {
            "knob": "MEMO_RECALL_MIN_SIM",
            "version_after": "v2",
            "floor_before": 0.62,
            "online_before": 0.6,
            "offline_before": {"precision_at_k": 0.2, "noise_at_k": 0.0},
        },
    )
    monkeypatch.setattr(dream_tune_online, "online_fraction", lambda sd, v, **k: (0.40, 50))
    monkeypatch.setattr(
        dream_tune, "measure", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no measure"))
    )

    res = dream_tune.run_tuning_pass(_cfg(tmp_path), object(), k=5)
    assert res["status"] == "online_reverted"
    # the revert restored the good 0.62
    assert _scalar_params(read_overlay(tmp_path))["MEMO_RECALL_MIN_SIM"] == 0.62
    # WITHOUT the fix, _meta.prev would be the bad 0.7 and this rollback would resurrect it.
    rolled = rollback_overlay(tmp_path)
    assert rolled["MEMO_RECALL_MIN_SIM"] == 0.62
    assert _scalar_params(read_overlay(tmp_path))["MEMO_RECALL_MIN_SIM"] == 0.62
