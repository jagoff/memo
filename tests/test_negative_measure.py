"""Negative-recall MEASURE slice — avoid@k as a first-class eval metric.

Covers the ⛔ AVOID channel's measurement machinery in ``memo.eval_recall``:

  * ``avoid_at_k``   — coverage: did the failure_pattern the ⛔ channel MUST
    surface actually surface (a dedicated type=failure_pattern vec pass, gated
    by the ⛔ min_sim floor and capped at MEMO_NEGATIVE_RECALL_K)?
  * ``avoid_leak_at_k`` — leakage: did a known-bad ``avoid_id`` stay OUT of the
    normal top-K?
  * the enforced gate (``avoid_gate_metrics`` / ``full_gate_metrics`` /
    ``check_gate``) that flags a regression in either.

All stubbed — NO real MLX. The harness's ``mem.search`` is a plain object that
partitions the normal pool from the failure_pattern ⛔ pool by the ``type_``
kwarg, so the two passes are independently observable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from memo import eval_recall
from memo.eval_recall import (
    Cfg,
    LabelSet,
    Prompt,
    Row,
    avoid_gate_metrics,
    check_gate,
    full_gate_metrics,
    gate_metrics,
    load_labels,
    rows_to_table,
    run_config,
)

_LABELS_PATH = Path(__file__).parent.parent / "eval" / "regression_labels.json"


@dataclass
class _Hit:
    id: str
    score: float | None
    title: str = "t"
    body: str = "distinct body long enough for ranking to keep it"
    type: str = "note"
    tags: list[str] = field(default_factory=list)
    path: str = "p.md"
    extra: dict[str, Any] = field(default_factory=dict)


class _Mem:
    """Stub Memory whose search partitions the normal pool from the ⛔
    (type=failure_pattern) pool, and records how many times each was hit."""

    def __init__(self, normal: list[_Hit], failure: list[_Hit] | None = None) -> None:
        self._normal = normal
        self._failure = failure or []
        self.normal_calls = 0
        self.failure_calls = 0

    def search(self, query: str, **kw: Any) -> list[_Hit]:
        if kw.get("type_") == "failure_pattern":
            self.failure_calls += 1
            return list(self._failure)
        self.normal_calls += 1
        return list(self._normal)


def _cfg(floor: float = 0.0) -> Cfg:
    return Cfg("X vec/0.0/keep", "vec", floor, exclude_archived=False)


@pytest.fixture(autouse=True)
def _pin_neg_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic ⛔ params for every test unless a test overrides them."""
    monkeypatch.setenv("MEMO_NEGATIVE_RECALL_K", "2")
    monkeypatch.setenv("MEMO_NEGATIVE_RECALL_MIN_SIM", "0.5")
    # Keep the pre-top-K paraphrase collapse out of these synthetic corpora.
    monkeypatch.setenv("MEMO_RECALL_DEDUP_COLLAPSE", "0")


# --- label plumbing ----------------------------------------------------------


def test_label_parses_expect_avoid_ids() -> None:
    lab = eval_recall._label_from_dict({"text": "q", "expect_avoid_ids": ["de5df482", "8f5fa2e2"]})
    assert lab.expect_avoid_ids == ["de5df482", "8f5fa2e2"]
    # absent → empty (schema-additive, old files unaffected)
    assert eval_recall._label_from_dict({"text": "q"}).expect_avoid_ids == []


def test_fingerprint_changes_with_expect_avoid_ids() -> None:
    a = LabelSet(prompts=[Prompt("q")])
    b = LabelSet(prompts=[Prompt("q", expect_avoid_ids=["de5df482"])])
    c = LabelSet(prompts=[Prompt("q", expect_avoid_ids=["de5df482"])])
    assert a.fingerprint() != b.fingerprint()
    assert b.fingerprint() == c.fingerprint()


# --- avoid@k coverage (the ⛔ channel surfaced the right failure_pattern) ------


def test_avoid_coverage_hit_when_failure_pattern_surfaces() -> None:
    mem = _Mem(normal=[], failure=[_Hit("aa11bb2233334444", 0.9, type="failure_pattern")])
    labels = LabelSet(
        prompts=[Prompt("risky release", relevant=False, expect_avoid_ids=["aa11bb22"])]
    )

    row = run_config(mem, _cfg(), 5, labels)

    assert row.avoid_at_k == 1.0  # the expected ⛔ id surfaced above the floor
    assert mem.failure_calls == 1  # a dedicated type=failure_pattern pass ran


def test_avoid_coverage_miss_when_below_floor() -> None:
    # 0.4 < the 0.5 ⛔ min_sim floor → filtered out of the ⛔ pass.
    mem = _Mem(normal=[], failure=[_Hit("aa11bb2233334444", 0.4, type="failure_pattern")])
    labels = LabelSet(prompts=[Prompt("q", relevant=False, expect_avoid_ids=["aa11bb22"])])

    row = run_config(mem, _cfg(), 5, labels)

    assert row.avoid_at_k == 0.0


def test_avoid_coverage_respects_k_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    # The expected id trails a stronger unrelated failure_pattern → only inside
    # the ⛔ pass's K window does it count.
    def _mem() -> _Mem:
        return _Mem(
            normal=[],
            failure=[
                _Hit("cccc000011112222", 0.9, type="failure_pattern"),
                _Hit("aa11bb2233334444", 0.8, type="failure_pattern"),
            ],
        )

    labels = LabelSet(prompts=[Prompt("q", relevant=False, expect_avoid_ids=["aa11bb22"])])

    monkeypatch.setenv("MEMO_NEGATIVE_RECALL_K", "1")
    assert run_config(_mem(), _cfg(), 5, labels).avoid_at_k == 0.0  # rank 2, excluded at K=1

    monkeypatch.setenv("MEMO_NEGATIVE_RECALL_K", "2")
    assert run_config(_mem(), _cfg(), 5, labels).avoid_at_k == 1.0  # now inside the window


def test_avoid_coverage_is_fraction_over_multiple_expected() -> None:
    mem = _Mem(
        normal=[],
        failure=[_Hit("aa11bb2233334444", 0.9, type="failure_pattern")],  # only one of two present
    )
    labels = LabelSet(
        prompts=[Prompt("q", relevant=False, expect_avoid_ids=["aa11bb22", "ffffffff"])]
    )
    # K=2 (fixture): only one of the two expected ⛔ ids surfaced → coverage 0.5.
    assert run_config(mem, _cfg(), 5, labels).avoid_at_k == 0.5


# --- avoid_leak@k (known-bad ids must stay OUT of the normal section) ----------


def test_avoid_leak_when_bad_id_surfaces_in_normal_topk() -> None:
    mem = _Mem(
        normal=[_Hit("bad0000011112222", 0.9), _Hit("good000011112222", 0.8)],
        failure=[],
    )
    labels = LabelSet(prompts=[Prompt("q", relevant=False, avoid_ids=["bad00000"])])

    row = run_config(mem, _cfg(), 2, labels)

    assert row.avoid_leak_at_k == 1.0  # the single avoid_id leaked into the top-2
    assert mem.failure_calls == 0  # no expect_avoid_ids → no ⛔ pass


def test_avoid_leak_zero_when_bad_id_absent() -> None:
    mem = _Mem(normal=[_Hit("good000011112222", 0.9)], failure=[])
    labels = LabelSet(prompts=[Prompt("q", relevant=False, avoid_ids=["bad00000"])])

    assert run_config(mem, _cfg(), 2, labels).avoid_leak_at_k == 0.0


def test_avoid_leak_is_independent_of_noise_fold() -> None:
    # avoid_ids ALSO fold into noise@k (pre-existing behavior). Confirm the new
    # first-class leak metric is computed WITHOUT disturbing that fold.
    mem = _Mem(normal=[_Hit("bad0000011112222", 0.9), _Hit("good000011112222", 0.8)], failure=[])
    labels = LabelSet(prompts=[Prompt("q", relevant=False, avoid_ids=["bad00000"])])

    row = run_config(mem, _cfg(), 2, labels)

    assert row.avoid_leak_at_k == 1.0
    assert row.noise_at_k == 0.5  # unchanged: 1 avoid-hit in top-2 / (1 prompt * k=2)


# --- inert without ⛔ labels (existing metrics unchanged when off) -------------


def test_no_avoid_labels_means_no_extra_search_and_zero_metrics() -> None:
    mem = _Mem(
        normal=[_Hit("aaaa111122223333", 0.9, title="hit")],
        failure=[_Hit("should-not-be-read", 0.99, type="failure_pattern")],
    )
    labels = LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["aaaa1111"])])

    row = run_config(mem, _cfg(), 1, labels)

    # ⛔ pass never issued; avoid metrics inert.
    assert mem.failure_calls == 0
    assert row.avoid_at_k == 0.0
    assert row.avoid_leak_at_k == 0.0
    # existing precision/noise unaffected by the presence of the avoid path.
    assert row.precision_at_k == 1.0
    assert row.noise_at_k == 0.0


def test_coverage_and_leak_compute_together_in_one_run() -> None:
    mem = _Mem(
        normal=[_Hit("bad0000011112222", 0.9), _Hit("keep000011112222", 0.8)],
        failure=[_Hit("aa11bb2233334444", 0.9, type="failure_pattern")],
    )
    labels = LabelSet(
        prompts=[
            Prompt("risky", relevant=False, avoid_ids=["bad00000"], expect_avoid_ids=["aa11bb22"])
        ]
    )

    row = run_config(mem, _cfg(), 2, labels)

    assert row.avoid_at_k == 1.0
    assert row.avoid_leak_at_k == 1.0
    assert mem.failure_calls == 1


# --- normal-pool exclusion mirrors production (parity with the hook) ----------


class _RecordingMem(_Mem):
    """`_Mem` that records the ``exclude_types`` the NORMAL search received."""

    def __init__(self, normal: list[_Hit], failure: list[_Hit] | None = None) -> None:
        super().__init__(normal, failure)
        self.seen_exclude_types: Any = "unset"

    def search(self, query: str, **kw: Any) -> list[_Hit]:
        if kw.get("type_") == "failure_pattern":
            self.failure_calls += 1
            return list(self._failure)
        self.seen_exclude_types = kw.get("exclude_types")
        self.normal_calls += 1
        return list(self._normal)


def test_eval_normal_pool_excludes_failure_pattern_when_enabled(monkeypatch) -> None:
    """When MEMO_NEGATIVE_RECALL_ENABLED is on the eval must exclude
    failure_pattern from the NORMAL pool — the same exclusion the recall hook
    applies (_recall_excluded_types) — so eval mirrors production."""
    monkeypatch.setenv("MEMO_NEGATIVE_RECALL_ENABLED", "1")
    mem = _RecordingMem(normal=[_Hit("aaaa111122223333", 0.9)])
    labels = LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["aaaa1111"])])

    run_config(mem, _cfg(), 1, labels)

    assert "failure_pattern" in (mem.seen_exclude_types or set())


def test_eval_normal_pool_keeps_failure_pattern_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv("MEMO_NEGATIVE_RECALL_ENABLED", raising=False)
    mem = _RecordingMem(normal=[_Hit("aaaa111122223333", 0.9)])
    labels = LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["aaaa1111"])])

    run_config(mem, _cfg(), 1, labels)

    assert "failure_pattern" not in (mem.seen_exclude_types or set())


# --- gate metrics + enforced gate --------------------------------------------


def test_avoid_gate_metrics_reads_the_best_row() -> None:
    rows = [
        Row(config="A", precision_at_k=0.4, noise_at_k=0.1, avoid_at_k=0.9, avoid_leak_at_k=0.3),
        Row(config="B", precision_at_k=0.8, noise_at_k=0.1, avoid_at_k=0.2, avoid_leak_at_k=0.0),
    ]
    # best_row is the higher-precision B — its avoid pair is what the gate tracks.
    assert avoid_gate_metrics(rows) == {"avoid_at_k": 0.2, "avoid_leak_at_k": 0.0}


def test_full_gate_metrics_is_gate_plus_avoid() -> None:
    rows = [
        Row(config="A", precision_at_k=0.6, noise_at_k=0.1, avoid_at_k=0.7, avoid_leak_at_k=0.2)
    ]
    full = full_gate_metrics(rows)
    assert full["precision_at_k"] == 0.6 and full["noise_at_k"] == 0.1
    assert full["avoid_at_k"] == 0.7 and full["avoid_leak_at_k"] == 0.2
    # superset of the legacy gate_metrics contract
    assert gate_metrics(rows).items() <= full.items()


def test_gate_flags_avoid_coverage_regression() -> None:
    rows = [Row(config="A", precision_at_k=0.6, noise_at_k=0.1, avoid_at_k=0.5)]
    baseline = {"precision_at_k": 0.6, "noise_at_k": 0.1, "avoid_at_k": 1.0, "avoid_leak_at_k": 0.0}

    res = check_gate(rows, baseline)

    assert not res.passed
    assert "avoid@k" in res.message


def test_gate_flags_avoid_leak_regression() -> None:
    rows = [
        Row(config="A", precision_at_k=0.6, noise_at_k=0.1, avoid_at_k=1.0, avoid_leak_at_k=0.5)
    ]
    baseline = {"precision_at_k": 0.6, "noise_at_k": 0.1, "avoid_at_k": 1.0, "avoid_leak_at_k": 0.0}

    res = check_gate(rows, baseline)

    assert not res.passed
    assert "avoid_leak@k" in res.message


def test_gate_passes_when_avoid_metrics_hold_or_improve() -> None:
    rows = [
        Row(config="A", precision_at_k=0.6, noise_at_k=0.1, avoid_at_k=1.0, avoid_leak_at_k=0.0)
    ]
    baseline = {"precision_at_k": 0.6, "noise_at_k": 0.1, "avoid_at_k": 1.0, "avoid_leak_at_k": 0.0}
    assert check_gate(rows, baseline).passed
    # improved coverage / lower leak still passes
    better = [
        Row(config="A", precision_at_k=0.7, noise_at_k=0.05, avoid_at_k=1.0, avoid_leak_at_k=0.0)
    ]
    assert check_gate(better, {**baseline, "avoid_at_k": 0.5, "avoid_leak_at_k": 0.5}).passed


def test_gate_is_backward_compatible_with_a_legacy_baseline() -> None:
    # A baseline seeded before avoid@k existed carries neither key → the ⛔
    # checks are vacuous (coverage floor 0.0, leak ceiling 1.0), so a run with
    # zero avoid signal still passes on precision/noise alone.
    legacy = {"precision_at_k": 0.6, "noise_at_k": 0.1}
    rows = [Row(config="A", precision_at_k=0.6, noise_at_k=0.1)]
    assert check_gate(rows, legacy).passed
    # a genuine precision drop is still caught under a legacy baseline.
    dropped = [Row(config="A", precision_at_k=0.4, noise_at_k=0.1)]
    res = check_gate(dropped, legacy)
    assert not res.passed and "precision@k" in res.message


def test_gate_message_reports_multiple_regressions_together() -> None:
    rows = [
        Row(config="A", precision_at_k=0.4, noise_at_k=0.3, avoid_at_k=0.2, avoid_leak_at_k=0.5)
    ]
    baseline = {"precision_at_k": 0.6, "noise_at_k": 0.1, "avoid_at_k": 1.0, "avoid_leak_at_k": 0.0}

    res = check_gate(rows, baseline)

    assert not res.passed
    for token in ("precision@k", "noise@k", "avoid@k", "avoid_leak@k"):
        assert token in res.message


# --- reporting ---------------------------------------------------------------


def test_rows_to_table_shows_avoid_section_only_when_present() -> None:
    with_avoid = [Row(config="A", precision_at_k=0.6, noise_at_k=0.1, avoid_at_k=1.0)]
    without = [Row(config="A", precision_at_k=0.6, noise_at_k=0.1)]
    assert "avoid@k" in rows_to_table(with_avoid, k=5)
    assert "avoid@k" not in rows_to_table(without, k=5)


# --- committed regression labels ---------------------------------------------


def test_update_baseline_persists_full_avoid_metrics(tmp_path, monkeypatch) -> None:
    """`memo eval recall --update-baseline` must persist full_gate_metrics so the
    saved baseline carries avoid@k / avoid_leak@k — otherwise check_gate's ⛔
    checks read the non-enforcing defaults (0.0 / 1.0) and are vacuously true
    forever. The wiring bug was that the CLI wrote bare gate_metrics."""
    import json

    from click.testing import CliRunner

    from memo.cli_eval import eval_group

    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    monkeypatch.setattr("memo.cli_eval._get_memory", lambda cfg: object())
    monkeypatch.setattr(eval_recall, "load_labels", lambda path: LabelSet(prompts=[Prompt("q")]))
    monkeypatch.setattr(
        eval_recall,
        "evaluate",
        lambda mem, *, k, labels, configs, progress=None: [
            Row(config="A", precision_at_k=0.6, noise_at_k=0.1, avoid_at_k=1.0, avoid_leak_at_k=0.0)
        ],
    )
    labels_file = tmp_path / "labels.json"  # load_labels is stubbed; only existence matters
    labels_file.write_text("{}", encoding="utf-8")

    res = CliRunner().invoke(
        eval_group,
        ["recall", "--labels", str(labels_file), "--update-baseline"],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output

    baseline = json.loads(
        (tmp_path / "state" / "eval" / "recall_baseline.json").read_text(encoding="utf-8")
    )
    # The avoid pair is now persisted (bare gate_metrics wrote neither).
    assert baseline["avoid_at_k"] == 1.0
    assert baseline["avoid_leak_at_k"] == 0.0

    # And a later avoid@k regression is now CAUGHT against this baseline.
    regressed = [
        Row(config="A", precision_at_k=0.6, noise_at_k=0.1, avoid_at_k=0.5, avoid_leak_at_k=0.0)
    ]
    gate = check_gate(regressed, baseline, labels_fingerprint=baseline["labels_fingerprint"])
    assert not gate.passed and "avoid@k" in gate.message


def test_committed_labels_carry_negative_recall_coverage() -> None:
    """eval/regression_labels.json must ship ⛔ coverage probes: prompts with
    expect_avoid_ids (>=8 hex chars) so the enforced gate has something to
    guard once the baseline is seeded with full_gate_metrics."""
    labels = load_labels(_LABELS_PATH)
    avoid_labels = [p for p in labels.prompts if p.expect_avoid_ids]
    assert len(avoid_labels) >= 2, "expected at least two committed ⛔ coverage probes"
    for p in avoid_labels:
        # coverage-only probes must not perturb precision@K / noise@K.
        assert not p.expect_ids, f"⛔ probe {p.text!r} must have empty expect_ids"
        for eid in p.expect_avoid_ids:
            assert len(eid) >= 8, f"expect_avoid_id {eid!r} shorter than 8 chars"
