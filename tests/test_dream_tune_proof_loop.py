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

    def _boom(*a, **k):
        raise AssertionError("search/measure must not run while awaiting online")

    monkeypatch.setattr(dream_tune, "search_min_sim", _boom)
    monkeypatch.setattr(dream_tune, "measure", _boom)

    res = dream_tune.run_tuning_pass(_cfg(tmp_path), object(), k=5)
    assert res["status"] == "awaiting_online"
    assert res["online"]["status"] == "waiting"
    assert dream_tune_online.read_pending(tmp_path) is not None


def test_online_reverted_rolls_back_and_restores_baseline(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(dream_tune, "build_labels", lambda cfg, **k: _one_label())
    dream_tune_online.write_pending(
        tmp_path,
        {"version_after": "v2", "online_before": 0.6,
         "offline_before": {"precision_at_k": 0.2, "noise_at_k": 0.0}},
    )
    monkeypatch.setattr(dream_tune_online, "online_fraction", lambda sd, v, **k: (0.40, 50))
    calls = {"rollback": 0, "baseline": None}
    monkeypatch.setattr(dream_tune, "rollback_overlay", lambda sd: calls.__setitem__("rollback", 1))
    monkeypatch.setattr(dream_tune, "save_baseline", lambda sd, m: calls.__setitem__("baseline", m))
    monkeypatch.setattr(dream_tune, "measure", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no measure")))

    res = dream_tune.run_tuning_pass(_cfg(tmp_path), object(), k=5)
    assert res["status"] == "online_reverted"
    assert calls["rollback"] == 1
    assert calls["baseline"] == {"precision_at_k": 0.2, "noise_at_k": 0.0}  # restored offline baseline


def test_apply_writes_pending(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(dream_tune, "build_labels", lambda cfg, **k: _one_label())
    # no pending → resolve returns "none" → proceeds to search
    monkeypatch.setattr(dream_tune, "load_baseline", lambda sd: None)
    before = {"precision_at_k": 0.2, "noise_at_k": 0.0}
    after = {"precision_at_k": 0.3, "noise_at_k": 0.0}
    monkeypatch.setattr(dream_tune, "search_min_sim", lambda *a, **k: (0.65, before, after))
    monkeypatch.setattr(dream_tune, "params_version", lambda sd: "vNEW")
    monkeypatch.setattr(dream_tune_online, "online_fraction", lambda sd, v, **k: (0.5, 10))

    res = dream_tune.run_tuning_pass(_cfg(tmp_path), object(), k=5)
    assert res["status"] == "applied"
    pend = dream_tune_online.read_pending(tmp_path)
    assert pend is not None
    assert pend["version_after"] == "vNEW"
    assert pend["floor_after"] == 0.65
    assert pend["online_before"] == 0.5
