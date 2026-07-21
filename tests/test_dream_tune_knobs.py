"""Fase 3 — nightly rank-knob tuner (MEMO_RECALL_MMR_LAMBDA /
MEMO_RECALL_SYNTHESIS_BOOST): faithful line-search via Cfg.knob_overrides,
curated no-regression gate, latency gate, single-apply guard, overlay
round-trip, online revert with per-knob baseline restore."""

from __future__ import annotations

import json
from types import SimpleNamespace

from memo import dream_tune as dt
from memo import dream_tune_online
from memo.eval_recall import LabelSet, Prompt, Row

_MMR = "MEMO_RECALL_MMR_LAMBDA"
_SYNTH = "MEMO_RECALL_SYNTHESIS_BOOST"


class _Hit:
    def __init__(self, id, score, title="t", tags=None, path="p", body="some body text"):
        self.id, self.score, self.title = id, score, title
        self.tags, self.path, self.body = tags or [], path, body


class _StubMem:
    def search(self, query, limit, mode="vec"):
        return [_Hit("aaaa1111", 0.9), _Hit("bbbb2222", 0.5)]


def _cfg(tmp_path):
    return SimpleNamespace(state_dir=tmp_path)


def _labels():
    return LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["aaaa1111"])])


def _one_label():
    return _labels(), True


def _metrics(prec, noise=0.0, lat=10.0):
    return {"precision_at_k": prec, "noise_at_k": noise, "latency_ms_p50": lat}


def _flat_min_sim(monkeypatch):
    """search_min_sim stub: no improvement (before == best, floor unchanged)."""
    flat = {"precision_at_k": 0.2, "noise_at_k": 0.0}
    monkeypatch.setattr(dt, "search_min_sim", lambda *a, **k: (0.5, dict(flat), dict(flat)))


def test_rank_knob_grids_match_spec():
    assert dt.RANK_KNOB_GRIDS[_MMR] == (0.0, 0.3, 0.5, 0.7)
    assert dt.RANK_KNOB_GRIDS[_SYNTH] == (0.0, 0.05, 0.10)


# --- measurement goes through the knob_overrides seam -------------------------


def test_measure_rank_knob_pins_value_via_knob_overrides(monkeypatch):
    captured = {}

    def _fake_evaluate(mem, *, k, labels, configs, **kw):
        captured["cfg"] = configs[0]
        return [
            Row(config=configs[0].name, precision_at_k=0.5, noise_at_k=0.1, latency_ms_p50=12.0)
        ]

    monkeypatch.setattr(dt, "evaluate", _fake_evaluate)
    m = dt.measure_rank_knob(_StubMem(), _labels(), k=5, floor=0.6, knob=_MMR, value=0.3)
    assert captured["cfg"].knob_overrides == {"mmr_lambda": 0.3}
    assert captured["cfg"].floor == 0.6
    assert m["precision_at_k"] == 0.5
    assert m["latency_ms_p50"] == 12.0

    dt.measure_rank_knob(_StubMem(), _labels(), k=5, floor=0.6, knob=_SYNTH, value=0.05)
    assert captured["cfg"].knob_overrides == {"synthesis_boost": 0.05}


# --- line search ---------------------------------------------------------------


def test_search_rank_knob_evaluates_full_grid(monkeypatch):
    seen = []
    table = {0.0: 0.2, 0.3: 0.25, 0.5: 0.4, 0.7: 0.3}

    def _fake_measure(mem, labels, *, k, floor, knob, value):
        seen.append(value)
        return _metrics(table[value])

    monkeypatch.setattr(dt, "measure_rank_knob", _fake_measure)
    best, before, best_m, rejected = dt.search_rank_knob(
        _StubMem(),
        _labels(),
        k=5,
        floor=0.5,
        knob=_MMR,
        current=0.0,
        grid=(0.0, 0.3, 0.5, 0.7),
        max_evals=20,
    )
    # before(current) once, then every non-current grid candidate
    assert seen == [0.0, 0.3, 0.5, 0.7]
    assert best == 0.5
    assert best_m["precision_at_k"] == 0.4
    assert before["precision_at_k"] == 0.2
    assert rejected == []


def test_search_rank_knob_latency_gate_rejects(monkeypatch):
    def _fake_measure(mem, labels, *, k, floor, knob, value):
        if value == 0.0:
            return _metrics(0.2, lat=10.0)  # current config
        if value == 0.5:
            return _metrics(0.9, lat=20.0)  # best precision but 2x p50 -> rejected
        return _metrics(0.25, lat=11.0)  # modest in-budget improvement

    monkeypatch.setattr(dt, "measure_rank_knob", _fake_measure)
    best, _before, best_m, rejected = dt.search_rank_knob(
        _StubMem(),
        _labels(),
        k=5,
        floor=0.5,
        knob=_MMR,
        current=0.0,
        grid=(0.0, 0.3, 0.5, 0.7),
        max_evals=20,
    )
    assert rejected == [0.5]
    assert best in (0.3, 0.7)  # the in-budget improvement wins instead
    assert best_m["precision_at_k"] == 0.25


def test_search_rank_knob_latency_gate_skipped_when_before_p50_zero(monkeypatch):
    def _fake_measure(mem, labels, *, k, floor, knob, value):
        if value == 0.0:
            return _metrics(0.2, lat=0.0)  # stub/tiny corpus rounds to 0
        return _metrics(0.4, lat=50.0)

    monkeypatch.setattr(dt, "measure_rank_knob", _fake_measure)
    best, _before, _best_m, rejected = dt.search_rank_knob(
        _StubMem(),
        _labels(),
        k=5,
        floor=0.5,
        knob=_MMR,
        current=0.0,
        grid=(0.0, 0.3),
        max_evals=20,
    )
    assert rejected == []  # a 0-latency baseline must not reject everything
    assert best == 0.3


# --- run_tuning_pass: apply + single-apply guard -------------------------------


def test_run_tuning_pass_applies_rank_knob(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(dt, "build_labels", lambda cfg, **k: _one_label())
    monkeypatch.setattr(dt, "load_baseline", lambda sd: None)
    _flat_min_sim(monkeypatch)

    def _fake_search(mem, labels, *, k, floor, knob, current, grid, max_evals):
        if knob == _MMR:
            return 0.5, _metrics(0.2), _metrics(0.4), []
        return current, _metrics(0.2), _metrics(0.2), []

    monkeypatch.setattr(dt, "search_rank_knob", _fake_search)
    monkeypatch.setattr(dt, "curated_gate", lambda *a, **k: {"ok": True})

    res = dt.run_tuning_pass(_cfg(tmp_path), _StubMem(), k=5)
    assert res["status"] == "applied"
    assert res["applied_knob"] == _MMR
    overlay = dt.read_overlay(tmp_path)
    assert overlay[_MMR] == 0.5
    assert "MEMO_RECALL_MIN_SIM" not in overlay  # min_sim was a noop — untouched
    # record_pending for the online proof loop, attributed to the new knob
    pend = dream_tune_online.read_pending(tmp_path)
    assert pend["knob"] == _MMR
    assert pend["floor_after"] == 0.5
    # per-knob offline baseline saved via the registered saver
    assert dt.load_rank_knob_baseline(tmp_path, _MMR)["precision_at_k"] == 0.4
    # every searched knob reports a verdict (the CLI JSON surfaces this)
    assert set(res["knobs"]) == {"MEMO_RECALL_MIN_SIM", _MMR, _SYNTH}
    assert res["knobs"][_MMR]["verdict"] == "applied"
    assert res["knobs"][_SYNTH]["verdict"] == "noop"


def test_single_apply_guard_picks_one_best_knob(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(dt, "build_labels", lambda cfg, **k: _one_label())
    monkeypatch.setattr(dt, "load_baseline", lambda sd: None)
    # ALL three knobs improve; mmr improves the most
    monkeypatch.setattr(
        dt,
        "search_min_sim",
        lambda *a, **k: (
            0.65,
            {"precision_at_k": 0.2, "noise_at_k": 0.0},
            {"precision_at_k": 0.35, "noise_at_k": 0.0},
        ),
    )

    def _fake_search(mem, labels, *, k, floor, knob, current, grid, max_evals):
        if knob == _MMR:
            return 0.3, _metrics(0.2), _metrics(0.45), []
        return 0.05, _metrics(0.2), _metrics(0.30), []

    monkeypatch.setattr(dt, "search_rank_knob", _fake_search)
    monkeypatch.setattr(dt, "curated_gate", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(dt, "curated_gate_min_sim", lambda *a, **k: {"ok": True})

    res = dt.run_tuning_pass(_cfg(tmp_path), _StubMem(), k=5)
    assert res["status"] == "applied"
    assert res["applied_knob"] == _MMR  # the single best change
    overlay = dt.read_overlay(tmp_path)
    assert overlay[_MMR] == 0.3
    assert "MEMO_RECALL_MIN_SIM" not in overlay  # NOT co-applied
    assert _SYNTH not in overlay  # NOT co-applied
    assert res["knobs"]["MEMO_RECALL_MIN_SIM"]["verdict"] == "deferred_single_apply"
    assert res["knobs"][_SYNTH]["verdict"] == "deferred_single_apply"
    # exactly one pending change for the proof loop
    assert dream_tune_online.read_pending(tmp_path)["knob"] == _MMR


def test_pending_in_flight_blocks_rank_knob_search(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(dt, "build_labels", lambda cfg, **k: _one_label())
    dream_tune_online.write_pending(tmp_path, {"version_after": "v2", "online_before": 0.5})
    monkeypatch.setattr(dream_tune_online, "online_fraction", lambda sd, v, **k: (0.9, 3))
    monkeypatch.setattr(dt, "params_version", lambda sd: "v2")

    def _boom(*a, **k):
        raise AssertionError("no knob search may run while a pending awaits online proof")

    monkeypatch.setattr(dt, "search_min_sim", _boom)
    monkeypatch.setattr(dt, "search_rank_knob", _boom)
    res = dt.run_tuning_pass(_cfg(tmp_path), _StubMem(), k=5)
    assert res["status"] == "awaiting_online"


def test_rank_knob_dry_run_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(dt, "build_labels", lambda cfg, **k: _one_label())
    monkeypatch.setattr(dt, "load_baseline", lambda sd: None)
    _flat_min_sim(monkeypatch)

    def _fake_search(mem, labels, *, k, floor, knob, current, grid, max_evals):
        if knob == _MMR:
            return 0.5, _metrics(0.2), _metrics(0.4), []
        return current, _metrics(0.2), _metrics(0.2), []

    monkeypatch.setattr(dt, "search_rank_knob", _fake_search)
    monkeypatch.setattr(dt, "curated_gate", lambda *a, **k: {"ok": True})

    res = dt.run_tuning_pass(_cfg(tmp_path), _StubMem(), k=5, dry_run=True)
    assert res["status"] == "would_apply"
    assert res["would_apply"] == {"knob": _MMR, "value": 0.5}
    assert dt.read_overlay(tmp_path) == {}
    assert dream_tune_online.read_pending(tmp_path) is None


# --- curated no-regression gate -------------------------------------------------


def test_curated_gate_rejects_regressing_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    # curated regression set in state_dir (resolved BEFORE the repo fallback)
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "regression_labels.json").write_text(
        json.dumps(
            {
                "schema": "memo.eval_recall.labels.v1",
                "prompts": [{"text": "curated-q", "relevant": True, "expect_ids": ["cccc3333"]}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dt, "build_labels", lambda cfg, **k: _one_label())
    monkeypatch.setattr(dt, "load_baseline", lambda sd: None)
    _flat_min_sim(monkeypatch)

    def _fake_search(mem, labels, *, k, floor, knob, current, grid, max_evals):
        if knob == _MMR:
            return 0.5, _metrics(0.2), _metrics(0.6), []  # improves on mined∪curated
        return current, _metrics(0.2), _metrics(0.2), []

    monkeypatch.setattr(dt, "search_rank_knob", _fake_search)

    # ...but on the CURATED set the candidate regresses (0.5 -> 0.1)
    def _fake_measure(mem, labels, *, k, floor, knob, value):
        assert labels.prompts[0].text == "curated-q"  # gate measures curated-only
        return _metrics(0.5 if value == 0.0 else 0.1)

    monkeypatch.setattr(dt, "measure_rank_knob", _fake_measure)

    res = dt.run_tuning_pass(_cfg(tmp_path), _StubMem(), k=5)
    assert res["status"] == "noop"
    assert res["knobs"][_MMR]["verdict"] == "curated_rejected"
    assert res["knobs"][_MMR]["curated"]["ok"] is False
    assert dt.read_overlay(tmp_path) == {}  # nothing applied
    assert dream_tune_online.read_pending(tmp_path) is None


def test_curated_gate_vacuous_without_curated_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "_curated_prompts", lambda sd: [])
    gate = dt.curated_gate(
        _StubMem(), tmp_path, k=5, floor=0.5, knob=_MMR, value_before=0.0, value_after=0.5
    )
    assert gate["ok"] is True
    assert gate["reason"] == "no_curated_labels"


def test_min_sim_curated_gate_rejects_regressing_floor(tmp_path, monkeypatch):
    """A floor that wins on mined labels but buries a curated must-surface
    memory is rejected — the min_sim candidate passes the same curated bar as
    the rank knobs."""
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "regression_labels.json").write_text(
        json.dumps(
            {
                "schema": "memo.eval_recall.labels.v1",
                "prompts": [{"text": "curated-q", "relevant": True, "expect_ids": ["cccc3333"]}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dt, "build_labels", lambda cfg, **k: _one_label())
    monkeypatch.setattr(dt, "load_baseline", lambda sd: None)
    # min_sim wins on the mined labels...
    monkeypatch.setattr(dt, "search_min_sim", lambda *a, **k: (0.65, _metrics(0.2), _metrics(0.4)))
    # ...rank knobs don't move...
    monkeypatch.setattr(
        dt,
        "search_rank_knob",
        lambda mem, labels, *, k, floor, knob, current, grid, max_evals: (
            current,
            _metrics(0.2),
            _metrics(0.2),
            [],
        ),
    )

    # ...but on the CURATED set the new floor regresses (0.5 -> 0.1)
    def _fake_measure(mem, labels, *, k, floor):
        assert labels.prompts[0].text == "curated-q"  # gate measures curated-only
        return _metrics(0.5 if floor == 0.5 else 0.1)

    monkeypatch.setattr(dt, "measure", _fake_measure)

    res = dt.run_tuning_pass(_cfg(tmp_path), _StubMem(), k=5)
    assert res["status"] == "noop"
    assert res["knobs"]["MEMO_RECALL_MIN_SIM"]["verdict"] == "curated_rejected"
    assert res["knobs"]["MEMO_RECALL_MIN_SIM"]["curated"]["ok"] is False
    assert dt.read_overlay(tmp_path) == {}  # nothing applied
    assert dream_tune_online.read_pending(tmp_path) is None


def _write_curated_with_noise(state_dir):
    eval_dir = state_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "regression_labels.json").write_text(
        json.dumps(
            {
                "schema": "memo.eval_recall.labels.v1",
                "prompts": [{"text": "curated-q", "relevant": True, "expect_ids": ["cccc3333"]}],
                "noise_tags": ["Screenshot", "garbled-ocr"],
                "noise_path_fragments": ["inactive/", "/old/"],
            }
        ),
        encoding="utf-8",
    )


def test_curated_label_set_passes_noise_fields_through(tmp_path):
    """The document-level noise_tags/noise_path_fragments must reach the
    LabelSet (same parsing as eval_recall.load_labels) — otherwise the gate's
    noise@K dimension is a vacuous 0.0."""
    _write_curated_with_noise(tmp_path)
    labels = dt._curated_label_set(tmp_path)
    assert labels is not None
    assert labels.prompts[0].text == "curated-q"
    assert labels.noise_tags == {"screenshot", "garbled-ocr"}  # lowercased like load_labels
    assert labels.noise_path_fragments == ("inactive/", "/old/")


def test_curated_gate_rejects_noise_raising_candidate(tmp_path, monkeypatch):
    """Equal curated precision but HIGHER curated noise ⇒ gate rejects."""
    _write_curated_with_noise(tmp_path)

    def _fake_measure(mem, labels, *, k, floor, knob, value):
        # The gate's label set must carry the noise fields through to the
        # measurement — the noise dimension is only real if they arrive here.
        assert labels.noise_tags == {"screenshot", "garbled-ocr"}
        assert labels.noise_path_fragments == ("inactive/", "/old/")
        return _metrics(0.5, noise=0.0 if value == 0.0 else 0.2)

    monkeypatch.setattr(dt, "measure_rank_knob", _fake_measure)
    gate = dt.curated_gate(
        _StubMem(), tmp_path, k=5, floor=0.5, knob=_MMR, value_before=0.0, value_after=0.5
    )
    assert gate["ok"] is False
    assert gate["after"]["noise_at_k"] > gate["before"]["noise_at_k"]


# --- overlay round-trip: write -> flag() -> knobs_from_flags --------------------


def test_overlay_roundtrip_reaches_knobs_from_flags(tmp_path, monkeypatch):
    from memo.flags import flag_float
    from memo.recall_logic import knobs_from_flags
    from memo.tuned_overlay import write_overlay

    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    monkeypatch.delenv(_MMR, raising=False)
    monkeypatch.delenv(_SYNTH, raising=False)
    write_overlay(tmp_path, {_MMR: 0.5, _SYNTH: 0.1}, {"set_by": "test"})

    # env unset -> the overlay value wins (env > overlay > default)
    assert flag_float(_MMR) == 0.5
    assert flag_float(_SYNTH) == 0.1
    knobs = knobs_from_flags()
    assert knobs.mmr_lambda == 0.5
    assert knobs.synthesis_boost == 0.1


# --- online revert + offline rollback guard ------------------------------------


def test_online_revert_restores_rank_knob_and_its_baseline(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(dt, "build_labels", lambda cfg, **k: _one_label())
    dream_tune_online.write_pending(
        tmp_path,
        {
            "knob": _MMR,
            "version_after": "v2",
            "floor_before": 0.0,
            "online_before": 0.6,
            "offline_before": {"precision_at_k": 0.2, "noise_at_k": 0.0},
        },
    )
    monkeypatch.setattr(dream_tune_online, "online_fraction", lambda sd, v, **k: (0.40, 50))

    res = dt.run_tuning_pass(_cfg(tmp_path), _StubMem(), k=5)
    assert res["status"] == "online_reverted"
    assert dt.read_overlay(tmp_path)[_MMR] == 0.0  # knob restored to pre-apply value
    # the reverted knob's OWN offline baseline was restored, not min_sim's
    assert dt.load_rank_knob_baseline(tmp_path, _MMR) == {
        "precision_at_k": 0.2,
        "noise_at_k": 0.0,
    }
    assert dt.load_baseline(tmp_path) is None
    assert dream_tune_online.in_revert_cooldown(tmp_path) is True


def test_rank_knob_rollback_guard_reverts_regressed_live(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(dt, "build_labels", lambda cfg, **k: _one_label())
    monkeypatch.setattr(dt, "load_baseline", lambda sd: None)  # min_sim guard inert
    # overlay history: min_sim first, then mmr applied over it (_meta.prev keeps min_sim)
    dt.write_overlay(tmp_path, {"MEMO_RECALL_MIN_SIM": 0.6}, {"set_by": "dream"})
    dt.write_overlay(
        tmp_path, {"MEMO_RECALL_MIN_SIM": 0.6, _MMR: 0.5}, {"set_by": "dream", "knob": _MMR}
    )
    dt.save_rank_knob_baseline(tmp_path, _MMR, {"precision_at_k": 0.4, "noise_at_k": 0.0})
    # the live config now regresses vs the saved mmr baseline
    monkeypatch.setattr(dt, "measure_rank_knob", lambda *a, **k: _metrics(0.1))

    def _boom(*a, **k):
        raise AssertionError("no search may run after the rollback guard fires")

    monkeypatch.setattr(dt, "search_min_sim", _boom)
    monkeypatch.setattr(dt, "search_rank_knob", _boom)

    res = dt.run_tuning_pass(_cfg(tmp_path), _StubMem(), k=5)
    assert res["status"] == "rolled_back"
    assert res["restored"] == {"MEMO_RECALL_MIN_SIM": 0.6}
    assert _MMR not in dt.read_overlay(tmp_path)


def test_baseline_savers_registered_for_rank_knobs(tmp_path):
    for knob in (_MMR, _SYNTH):
        saver = dt._KNOB_BASELINE_SAVERS[knob]
        saver(tmp_path, {"precision_at_k": 0.3, "noise_at_k": 0.0})
        assert dt.load_rank_knob_baseline(tmp_path, knob)["precision_at_k"] == 0.3
