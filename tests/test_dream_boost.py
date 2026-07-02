from types import SimpleNamespace

from memo import dream_tune, dream_tune_online


def _cfg(tmp_path):
    return SimpleNamespace(state_dir=tmp_path)


def test_boost_defers_under_pending(tmp_path, monkeypatch):
    dream_tune_online.write_pending(tmp_path, {"version_after": "v2"})
    monkeypatch.setattr(
        dream_tune,
        "write_overlay",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no apply")),
    )
    res = dream_tune.run_boost_pass(_cfg(tmp_path), object(), step=0.05)
    assert res["status"] == "deferred_pending"


def test_boost_applies_and_records_pending(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(dream_tune, "write_overlay", lambda sd, params, meta: None)
    monkeypatch.setattr(dream_tune, "params_version", lambda sd: "vB")
    monkeypatch.setattr(dream_tune_online, "online_fraction", lambda sd, v, **k: (0.5, 10))
    res = dream_tune.run_boost_pass(_cfg(tmp_path), object(), step=0.05)
    assert res["status"] == "applied"
    assert res["boost_after"] == 0.3  # 0.25 + 0.05, explore up with no history
    pend = dream_tune_online.read_pending(tmp_path)
    assert pend["knob"] == "MEMO_RECALL_PROJECT_BOOST" and pend["floor_after"] == 0.3


def test_boost_direction_reverses_after_revert(tmp_path):
    dream_tune_online.append_ledger(
        tmp_path,
        {
            "knob": "MEMO_RECALL_PROJECT_BOOST",
            "verdict": "reverted",
            "floor_before": 0.25,
            "floor_after": 0.30,
        },
    )
    assert dream_tune._boost_direction(tmp_path, 0.05) == -0.05  # up was reverted → go down


def test_boost_direction_repeats_after_confirm(tmp_path):
    dream_tune_online.append_ledger(
        tmp_path,
        {
            "knob": "MEMO_RECALL_PROJECT_BOOST",
            "verdict": "confirmed",
            "floor_before": 0.25,
            "floor_after": 0.30,
        },
    )
    assert dream_tune._boost_direction(tmp_path, 0.05) == 0.05  # up was confirmed → keep going up


def test_boost_boundary_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("MEMO_RECALL_PROJECT_BOOST", "0.5")  # at hi
    res = dream_tune.run_boost_pass(_cfg(tmp_path), object(), step=0.05)
    assert res["status"] == "noop" and res.get("reason") == "boundary"
