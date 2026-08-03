"""dream_flags — dark-feature (flag) graduation: gate completeness, A/B
streaks, overlay apply/revert, cooldown, deadline cull, dry-run."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from memo import dream_flags as df
from memo.eval_recall import LabelSet, Prompt
from memo.flags import REGISTRY
from memo.tuned_overlay import read_overlay, write_overlay

TODAY = date(2026, 7, 15)


# --- the program's lock: every dark flag declares its gate ---------------------


def test_every_dark_flag_has_a_gate():
    """A new default-off *_ENABLED flag cannot merge without declaring its
    graduation gate in dream_flags.GATES (recall / tuner / manual + reason)."""
    missing = sorted({s.name for s in df.dark_flags()} - set(df.GATES))
    assert not missing, (
        f"dark flags without a declared graduation gate: {missing}. "
        "Add a GateSpec to dream_flags.GATES — kind 'recall' if the flag is "
        "measurable via the eval flag_overrides seam, 'tuner' if an existing "
        "nightly pass owns it, else 'manual' with the reason."
    )


def test_dark_scalar_boost_flags_are_inventoried_with_a_recall_gate():
    """The spec's dark flags include the float proximity boost (default 0.0 =
    OFF): it must be tracked by the graduation inventory and declare a recall
    A/B gate, not escape GATES because it is not a bool *_ENABLED flag."""
    name = "MEMO_RECALL_CODE_PROXIMITY_BOOST"
    assert name in {s.name for s in df.dark_flags()}
    assert name in df.GATES
    assert df.GATES[name].kind == "recall"


def test_no_graduated_or_stale_gates():
    dark = {s.name for s in df.dark_flags()}
    stale = sorted(set(df.GATES) - dark)
    assert not stale, f"GATES entries for flags that are no longer dark: {stale}"


def test_gate_reasons_are_documented():
    assert all(g.reason.strip() for g in df.GATES.values())


def test_recall_gates_extra_flags_exist():
    for g in df.GATES.values():
        for extra, _v in g.extra_flags:
            assert extra in REGISTRY, f"{g.flag}: unknown companion flag {extra}"


# --- helpers -------------------------------------------------------------------


_FLAG = "MEMO_TEST_DARK_ENABLED"


def _mini_registry(monkeypatch, *, kind="recall", extra_flags=()):
    """Shrink the world to ONE tracked dark flag so selection is deterministic."""
    gate = df.GateSpec(_FLAG, kind, "test", extra_flags=tuple(extra_flags))

    class _Spec:
        name = _FLAG

    monkeypatch.setattr(df, "GATES", {_FLAG: gate})
    monkeypatch.setattr(df, "dark_flags", lambda: [_Spec()])
    monkeypatch.setattr(df, "_human_value", lambda name: None)
    monkeypatch.setattr(df, "_curated_label_set", lambda sd: None)
    monkeypatch.setattr(
        df,
        "build_labels",
        lambda cfg, **kw: (
            LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["aaaa1111"])]),
            False,
        ),
    )
    return gate


def _metrics(prec, noise=0.0, p50=10.0):
    return {"precision_at_k": prec, "noise_at_k": noise, "latency_ms_p50": p50}


def _stub_measure(monkeypatch, *, on, off):
    calls = []

    def fake(mem, labels, *, k, spec, enabled, floor):
        calls.append((spec.flag, enabled))
        return dict(on if enabled else off)

    monkeypatch.setattr(df, "measure_flag", fake)
    return calls


# --- streak / graduation ---------------------------------------------------------


def test_win_streak_graduates_flag_to_overlay(tmp_cfg, monkeypatch):
    _mini_registry(monkeypatch)
    _stub_measure(monkeypatch, on=_metrics(0.8), off=_metrics(0.4))
    monkeypatch.setenv("MEMO_FLAG_GRADUATION_WIN_NIGHTS", "2")

    r1 = df.run_flag_graduation_pass(tmp_cfg, mem=None, today=TODAY)
    assert r1["flags"][_FLAG]["verdict"] == "win"
    assert df.load_state(tmp_cfg.state_dir)["flags"][_FLAG]["streak"] == 1
    assert not read_overlay(tmp_cfg.state_dir).get(_FLAG)

    r2 = df.run_flag_graduation_pass(tmp_cfg, mem=None, today=TODAY + timedelta(days=1))
    assert r2["flags"][_FLAG]["verdict"] == "graduated"
    assert r2["status"] == "applied"
    assert read_overlay(tmp_cfg.state_dir)[_FLAG] is True
    entry = df.load_state(tmp_cfg.state_dir)["flags"][_FLAG]
    assert entry["status"] == "graduated"
    assert entry["baseline"]["precision_at_k"] == 0.8


def test_absolute_latency_ceiling_blocks_win_when_relative_gate_skipped(tmp_cfg, monkeypatch):
    """Fase 8: when OFF p50 rounds to 0 the RELATIVE headroom gate is skipped, so
    a candidate whose ON p50 balloons to seconds would otherwise win. The
    absolute latency-ceiling backstop must reject it (no streak, no graduate)."""
    _mini_registry(monkeypatch)
    # precision improves, but ON latency 6000ms >> ceiling; OFF p50 == 0 so the
    # relative headroom gate (budget = 0) does not fire — only the ceiling can.
    _stub_measure(monkeypatch, on=_metrics(0.9, p50=6000.0), off=_metrics(0.4, p50=0.0))
    monkeypatch.setenv("MEMO_FLAG_GRADUATION_LATENCY_CEILING_MS", "1500")

    res = df.run_flag_graduation_pass(tmp_cfg, mem=None, today=TODAY)
    verdict = res["flags"][_FLAG]
    assert verdict["verdict"] == "lose"
    assert verdict.get("latency_ceiling_rejected") is True
    assert df.load_state(tmp_cfg.state_dir)["flags"][_FLAG]["streak"] == 0


def test_lose_resets_streak(tmp_cfg, monkeypatch):
    _mini_registry(monkeypatch)
    _stub_measure(monkeypatch, on=_metrics(0.8), off=_metrics(0.4))
    df.run_flag_graduation_pass(tmp_cfg, mem=None, today=TODAY)
    _stub_measure(monkeypatch, on=_metrics(0.4), off=_metrics(0.8))
    res = df.run_flag_graduation_pass(tmp_cfg, mem=None, today=TODAY + timedelta(days=1))
    assert res["flags"][_FLAG]["verdict"] == "lose"
    assert df.load_state(tmp_cfg.state_dir)["flags"][_FLAG]["streak"] == 0


def test_graduation_writes_companion_flags(tmp_cfg, monkeypatch):
    _mini_registry(monkeypatch, extra_flags=(("MEMO_TEST_COMPANION_ENABLED", "1"),))
    _stub_measure(monkeypatch, on=_metrics(0.8), off=_metrics(0.4))
    monkeypatch.setenv("MEMO_FLAG_GRADUATION_WIN_NIGHTS", "1")
    res = df.run_flag_graduation_pass(tmp_cfg, mem=None, today=TODAY)
    assert res["flags"][_FLAG]["verdict"] == "graduated"
    ov = read_overlay(tmp_cfg.state_dir)
    assert ov[_FLAG] is True and ov["MEMO_TEST_COMPANION_ENABLED"] is True


# --- gates: latency + curated ----------------------------------------------------


def test_latency_headroom_rejects_win(tmp_cfg, monkeypatch):
    _mini_registry(monkeypatch)
    _stub_measure(monkeypatch, on=_metrics(0.8, p50=100.0), off=_metrics(0.4, p50=10.0))
    res = df.run_flag_graduation_pass(tmp_cfg, mem=None, today=TODAY)
    assert res["flags"][_FLAG]["verdict"] == "lose"
    assert res["flags"][_FLAG]["latency_rejected"] is True


def test_curated_regression_rejects_win(tmp_cfg, monkeypatch):
    _mini_registry(monkeypatch)
    curated = LabelSet(prompts=[Prompt("c", relevant=True, expect_ids=["bbbb2222"])])
    monkeypatch.setattr(df, "_curated_label_set", lambda sd: curated)

    def fake(mem, labels, *, k, spec, enabled, floor):
        if labels is curated:  # curated set regresses when ON
            return _metrics(0.1) if enabled else _metrics(0.9)
        return _metrics(0.8) if enabled else _metrics(0.4)

    monkeypatch.setattr(df, "measure_flag", fake)
    res = df.run_flag_graduation_pass(tmp_cfg, mem=None, today=TODAY)
    assert res["flags"][_FLAG]["verdict"] == "lose"
    assert res["flags"][_FLAG]["curated_rejected"] is True


# --- post-graduation guard / cooldown --------------------------------------------


def _seed_graduated(tmp_cfg, baseline_prec=0.8):
    write_overlay(tmp_cfg.state_dir, {_FLAG: True}, {"set_by": "test"})
    df.save_state(
        tmp_cfg.state_dir,
        {
            "flags": {
                _FLAG: {
                    "first_tracked": TODAY.isoformat(),
                    "streak": 3,
                    "status": "graduated",
                    "baseline": _metrics(baseline_prec),
                }
            }
        },
    )


def test_regression_guard_reverts_graduated_flag(tmp_cfg, monkeypatch):
    _mini_registry(monkeypatch)
    _seed_graduated(tmp_cfg)
    _stub_measure(monkeypatch, on=_metrics(0.2), off=_metrics(0.2))  # live ON regressed
    res = df.run_flag_graduation_pass(tmp_cfg, mem=None, today=TODAY + timedelta(days=3))
    assert res["flags"][_FLAG]["verdict"] == "reverted"
    assert _FLAG not in read_overlay(tmp_cfg.state_dir) or not read_overlay(tmp_cfg.state_dir).get(
        _FLAG
    )
    entry = df.load_state(tmp_cfg.state_dir)["flags"][_FLAG]
    assert entry["status"] == "reverted" and entry["streak"] == 0


def test_regression_guard_defers_revert_while_pending(tmp_cfg, monkeypatch):
    """F6: the post-graduation revert write must be gated like every other
    overlay writer — a same-night tuner experiment is pending, so bumping
    params_version now would orphan its cohort. Defer the revert instead."""
    from memo import dream_tune_online

    _mini_registry(monkeypatch)
    _seed_graduated(tmp_cfg)
    _stub_measure(monkeypatch, on=_metrics(0.2), off=_metrics(0.2))  # live ON regressed
    # A tuner change is in flight this cycle.
    dream_tune_online.write_pending(
        tmp_cfg.state_dir, {"knob": "MEMO_RECALL_MIN_SIM", "version_after": "v1"}
    )

    res = df.run_flag_graduation_pass(tmp_cfg, mem=None, today=TODAY + timedelta(days=3))

    assert res["flags"][_FLAG]["verdict"] == "deferred_pending"
    # overlay untouched: the flag stays graduated (ON) this cycle
    assert read_overlay(tmp_cfg.state_dir).get(_FLAG) is True
    assert df.load_state(tmp_cfg.state_dir)["flags"][_FLAG]["status"] == "graduated"


def test_overlay_rollback_unstrands_graduated_flag(tmp_cfg, monkeypatch):
    """Deleting tuned_params.json (the documented overlay rollback) leaves
    state saying 'graduated' with no overlay key: the flag resolves OFF live,
    the guard skips it, and _eligible never re-measures it. The pass must
    reset it to tracking so it re-enters the A/B pool."""
    _mini_registry(monkeypatch)
    _stub_measure(monkeypatch, on=_metrics(0.8), off=_metrics(0.4))
    # Graduated in state, but the overlay was rolled back out-of-band (no key).
    df.save_state(
        tmp_cfg.state_dir,
        {
            "flags": {
                _FLAG: {
                    "first_tracked": TODAY.isoformat(),
                    "streak": 3,
                    "status": "graduated",
                    "baseline": _metrics(0.8),
                }
            }
        },
    )
    res = df.run_flag_graduation_pass(tmp_cfg, mem=None, today=TODAY + timedelta(days=3))
    assert res["measured"] == [_FLAG]  # re-entered the A/B pool immediately
    entry = df.load_state(tmp_cfg.state_dir)["flags"][_FLAG]
    assert entry["status"] == "tracking"
    assert entry["streak"] == 1  # fresh streak from this night's win, not the stale 3
    assert "baseline" not in entry


def test_reverted_flag_cools_down_before_remeasure(tmp_cfg, monkeypatch):
    _mini_registry(monkeypatch)
    _stub_measure(monkeypatch, on=_metrics(0.8), off=_metrics(0.4))
    df.save_state(
        tmp_cfg.state_dir,
        {
            "flags": {
                _FLAG: {
                    "first_tracked": TODAY.isoformat(),
                    "streak": 0,
                    "status": "reverted",
                    "reverted_at": (TODAY - timedelta(days=5)).isoformat(),
                }
            }
        },
    )
    res = df.run_flag_graduation_pass(tmp_cfg, mem=None, today=TODAY)
    assert res["measured"] == []  # still cooling down (retry default 14d)
    late = df.run_flag_graduation_pass(tmp_cfg, mem=None, today=TODAY + timedelta(days=20))
    assert late["measured"] == [_FLAG]


# --- ownership / deadline / dry-run ----------------------------------------------


def test_human_pinned_flag_is_human_graduated_and_skipped(tmp_cfg, monkeypatch):
    _mini_registry(monkeypatch)
    monkeypatch.setattr(df, "_human_value", lambda name: "1")
    calls = _stub_measure(monkeypatch, on=_metrics(0.8), off=_metrics(0.4))
    res = df.run_flag_graduation_pass(tmp_cfg, mem=None, today=TODAY)
    assert res["measured"] == [] and calls == []
    assert df.load_state(tmp_cfg.state_dir)["flags"][_FLAG]["status"] == "human_graduated"


def test_deadline_marks_cull_candidate(tmp_cfg, monkeypatch):
    _mini_registry(monkeypatch, kind="manual")
    df.save_state(
        tmp_cfg.state_dir,
        {
            "flags": {
                _FLAG: {
                    "first_tracked": (TODAY - timedelta(days=60)).isoformat(),
                    "streak": 0,
                    "status": "tracking",
                }
            }
        },
    )
    res = df.run_flag_graduation_pass(tmp_cfg, mem=None, today=TODAY)
    assert res["cull_candidates"] == [_FLAG]
    assert df.load_state(tmp_cfg.state_dir)["flags"][_FLAG]["status"] == "cull_candidate"


def test_dry_run_writes_nothing(tmp_cfg, monkeypatch):
    _mini_registry(monkeypatch)
    _stub_measure(monkeypatch, on=_metrics(0.8), off=_metrics(0.4))
    monkeypatch.setenv("MEMO_FLAG_GRADUATION_WIN_NIGHTS", "1")
    res = df.run_flag_graduation_pass(tmp_cfg, mem=None, today=TODAY, dry_run=True)
    assert res["flags"][_FLAG]["verdict"] == "would_graduate"
    assert read_overlay(tmp_cfg.state_dir) == {}
    assert df.load_state(tmp_cfg.state_dir) == {}


def test_pass_never_raises(tmp_cfg, monkeypatch):
    def boom(cfg, **kw):
        raise RuntimeError("labels exploded")

    monkeypatch.setattr(df, "build_labels", boom)
    res = df.run_flag_graduation_pass(tmp_cfg, mem=None, today=TODAY)
    assert res["status"] == "error" and "labels exploded" in res["error"]


# --- status view -----------------------------------------------------------------


def test_status_rows_cover_every_dark_flag(tmp_cfg, monkeypatch):
    monkeypatch.setattr(df, "_human_value", lambda name: None)
    rows = df.status_rows(tmp_cfg, today=TODAY)
    assert {r["flag"] for r in rows} == {s.name for s in df.dark_flags()}
    assert all(r["kind"] != "MISSING_GATE" for r in rows)


# --- measure_flag plumbing (no MLX: stub search) -----------------------------------


class _Hit:
    def __init__(self, id, score):
        self.id, self.score, self.title = id, score, "t"
        self.tags, self.path, self.body = [], "p", "some body text"


class _StubMem:
    def search(self, query, limit, mode="vec", exclude_types=None, exclude_tags=None):
        return [_Hit("aaaa1111", 0.9)]


@pytest.mark.parametrize("enabled", [True, False])
def test_measure_flag_pins_env_through_eval(monkeypatch, enabled):
    spec = df.GateSpec(_FLAG, "recall", "test", extra_flags=(("MEMO_TEST_EXTRA", "1"),))
    seen = {}

    import os

    real_evaluate = df.evaluate

    def spying_evaluate(mem, *, k, labels, configs):
        seen["pins"] = dict(configs[0].flag_overrides or {})
        return real_evaluate(mem, k=k, labels=labels, configs=configs)

    monkeypatch.setattr(df, "evaluate", spying_evaluate)
    labels = LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["aaaa1111"])])
    m = df.measure_flag(_StubMem(), labels, k=3, spec=spec, enabled=enabled, floor=0.5)
    assert "precision_at_k" in m and "latency_ms_p50" in m
    assert seen["pins"][_FLAG] == ("1" if enabled else "0")
    assert ("MEMO_TEST_EXTRA" in seen["pins"]) is enabled
    assert os.environ.get(_FLAG) is None  # pins restored after the run
