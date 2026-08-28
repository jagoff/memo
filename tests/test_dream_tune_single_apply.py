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
    monkeypatch.setattr(
        dream_tune,
        "search_graph_signal",
        lambda *a, **k: (
            {"MEMO_GRAPH_SIGNAL_ENABLED": True, "MEMO_GRAPH_SIGNAL_ALPHA": 0.2},
            before,
            after,
        ),
    )

    def _boom(*a, **k):
        raise AssertionError("overlay must not be written while a pending is in flight")

    monkeypatch.setattr(dream_tune, "write_overlay", _boom)

    res = dream_tune.run_graph_weight_pass(_cfg(tmp_path), object(), k=5)
    assert res["status"] == "deferred_pending"
    assert dream_tune_online.read_pending(tmp_path) is not None  # untouched


def test_retired_graph_retrieval_pass_is_always_inert(tmp_path):
    dream_tune_online.write_pending(tmp_path, {"version_after": "v2"})
    res = dream_tune.run_graph_retrieval_pass(_cfg(tmp_path), object(), k=5)
    assert res["status"] == "retired"


def test_graph_weight_applies_when_no_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(dream_tune, "build_labels", lambda cfg, **k: _one_label())
    # This test isolates the apply path. The curated no-regression gate is
    # covered separately and otherwise loads the repository's real labels,
    # which require a searchable Memory rather than this test's object() stub.
    monkeypatch.setattr(dream_tune, "_curated_label_set", lambda _state_dir: None)
    monkeypatch.setattr(dream_tune, "load_graph_baseline", lambda sd: None)
    before = {"precision_at_k": 0.2, "noise_at_k": 0.0}
    after = {"precision_at_k": 0.3, "noise_at_k": 0.0}
    monkeypatch.setattr(
        dream_tune,
        "search_graph_signal",
        lambda *a, **k: (
            {"MEMO_GRAPH_SIGNAL_ENABLED": True, "MEMO_GRAPH_SIGNAL_ALPHA": 0.2},
            before,
            after,
        ),
    )
    calls = {"overlay": None}
    monkeypatch.setattr(
        dream_tune,
        "write_overlay",
        lambda sd, params, meta: calls.__setitem__("overlay", dict(params)),
    )
    monkeypatch.setattr(dream_tune, "save_graph_baseline", lambda sd, m: None)

    res = dream_tune.run_graph_weight_pass(_cfg(tmp_path), object(), k=5)
    assert res["status"] == "applied"
    assert calls["overlay"]["MEMO_GRAPH_SIGNAL_ENABLED"] is True
    assert calls["overlay"]["MEMO_GRAPH_SIGNAL_ALPHA"] == 0.2


def test_graph_weight_defers_during_revert_cooldown(tmp_path, monkeypatch):
    dream_tune_online.set_revert_cooldown(tmp_path)  # a revert happened earlier this cycle
    monkeypatch.setattr(dream_tune, "build_labels", lambda cfg, **k: _one_label())
    monkeypatch.setattr(dream_tune, "load_graph_baseline", lambda sd: None)
    before = {"precision_at_k": 0.2, "noise_at_k": 0.0}
    after = {"precision_at_k": 0.3, "noise_at_k": 0.0}
    monkeypatch.setattr(
        dream_tune,
        "search_graph_signal",
        lambda *a, **k: (
            {"MEMO_GRAPH_SIGNAL_ENABLED": True, "MEMO_GRAPH_SIGNAL_ALPHA": 0.2},
            before,
            after,
        ),
    )

    def _boom(*a, **k):
        raise AssertionError("must not apply during revert cooldown")

    monkeypatch.setattr(dream_tune, "write_overlay", _boom)
    res = dream_tune.run_graph_weight_pass(_cfg(tmp_path), object(), k=5)
    assert res["status"] == "deferred_pending"


def test_tuning_pass_sets_cooldown_on_online_revert(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(dream_tune, "build_labels", lambda cfg, **k: _one_label())
    dream_tune_online.write_pending(
        tmp_path,
        {
            "knob": "MEMO_RECALL_MIN_SIM",
            "version_after": "v2",
            "floor_before": 0.5,
            "online_before": 0.6,
            "offline_before": {"precision_at_k": 0.2, "noise_at_k": 0.0},
        },
    )
    monkeypatch.setattr(dream_tune_online, "online_fraction", lambda sd, v, **k: (0.40, 50))
    monkeypatch.setattr(dream_tune, "write_overlay", lambda sd, params, meta: None)
    monkeypatch.setattr(dream_tune, "save_baseline", lambda sd, m, **kw: None)
    monkeypatch.setattr(
        dream_tune, "measure", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no measure"))
    )

    res = dream_tune.run_tuning_pass(_cfg(tmp_path), object(), k=5)
    assert res["status"] == "online_reverted"
    assert dream_tune_online.in_revert_cooldown(tmp_path) is True


def test_graph_weight_skips_search_when_deferring(tmp_path, monkeypatch):
    dream_tune_online.write_pending(tmp_path, {"version_after": "v2"})  # pending in flight
    monkeypatch.setattr(dream_tune, "build_labels", lambda cfg, **k: _one_label())

    def _boom_search(*a, **k):
        raise AssertionError("search must not run when the pass is going to defer")

    monkeypatch.setattr(dream_tune, "search_graph_signal", _boom_search)
    # a real object() as mem would also fail in search; the guard must return first
    res = dream_tune.run_graph_weight_pass(_cfg(tmp_path), object(), k=5)
    assert res["status"] == "deferred_pending"


def test_tuning_pass_clears_cooldown_at_cycle_start(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    dream_tune_online.set_revert_cooldown(tmp_path)  # stale marker from a prior cycle
    monkeypatch.setattr(dream_tune, "build_labels", lambda cfg, **k: _one_label())
    monkeypatch.setattr(dream_tune, "load_baseline", lambda sd: None)
    # no pending → resolve returns "none"; search finds no improvement → noop, but cooldown must be cleared
    monkeypatch.setattr(
        dream_tune,
        "search_min_sim",
        lambda *a, **k: (
            0.5,
            {"precision_at_k": 0.2, "noise_at_k": 0.0},
            {"precision_at_k": 0.2, "noise_at_k": 0.0},
        ),
    )
    dream_tune.run_tuning_pass(_cfg(tmp_path), object(), k=5)
    assert dream_tune_online.in_revert_cooldown(tmp_path) is False
