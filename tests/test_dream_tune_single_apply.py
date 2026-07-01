from types import SimpleNamespace

from memo import dream_tune, dream_tune_online
from memo.eval_recall import LabelSet, Prompt


def _cfg(tmp_path):
    return SimpleNamespace(state_dir=tmp_path)


def _one_label():
    return LabelSet(prompts=[Prompt(text="q", relevant=True, expect_ids=[])]), True


def test_has_unresolved_pending(tmp_path):
    assert dream_tune_online.has_unresolved_pending(tmp_path) is False
    dream_tune_online.write_pending(tmp_path, {"version_after": "v2"})
    assert dream_tune_online.has_unresolved_pending(tmp_path) is True


def test_graph_weight_defers_while_pending_in_flight(tmp_path, monkeypatch):
    dream_tune_online.write_pending(tmp_path, {"version_after": "v2"})  # min_sim change in flight
    monkeypatch.setattr(dream_tune, "build_labels", lambda cfg, **k: _one_label())
    monkeypatch.setattr(dream_tune, "load_graph_baseline", lambda sd: None)
    before = {"precision_at_k": 0.2, "noise_at_k": 0.0}
    after = {"precision_at_k": 0.3, "noise_at_k": 0.0}
    monkeypatch.setattr(dream_tune, "search_graph_weight", lambda *a, **k: (0.2, before, after))

    def _boom(*a, **k):
        raise AssertionError("overlay must not be written while a pending is in flight")

    monkeypatch.setattr(dream_tune, "write_overlay", _boom)

    res = dream_tune.run_graph_weight_pass(_cfg(tmp_path), object(), k=5)
    assert res["status"] == "deferred_pending"
    assert dream_tune_online.read_pending(tmp_path) is not None  # untouched


def test_graph_weight_applies_when_no_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(dream_tune, "build_labels", lambda cfg, **k: _one_label())
    monkeypatch.setattr(dream_tune, "load_graph_baseline", lambda sd: None)
    before = {"precision_at_k": 0.2, "noise_at_k": 0.0}
    after = {"precision_at_k": 0.3, "noise_at_k": 0.0}
    monkeypatch.setattr(dream_tune, "search_graph_weight", lambda *a, **k: (0.2, before, after))
    calls = {"overlay": None}
    monkeypatch.setattr(dream_tune, "write_overlay", lambda sd, params, meta: calls.__setitem__("overlay", dict(params)))
    monkeypatch.setattr(dream_tune, "save_graph_baseline", lambda sd, m: None)

    res = dream_tune.run_graph_weight_pass(_cfg(tmp_path), object(), k=5)
    assert res["status"] == "applied"
    assert calls["overlay"]["MEMO_RECALL_GRAPH_PROXIMITY_WEIGHT"] == 0.2
