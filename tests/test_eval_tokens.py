import json as _json
from dataclasses import dataclass as _dc
from pathlib import Path

import pytest

from memo import eval_tokens


def test_count_tokens_is_ceil_chars_over_four():
    assert eval_tokens.count_tokens("") == 0
    assert eval_tokens.count_tokens("abcd") == 1
    assert eval_tokens.count_tokens("abcde") == 2  # 5 chars -> ceil(5/4) == 2


def test_surviving_ids_matches_eight_char_prefix_in_block():
    block = "**[5d7d253a] Some title**\n> body text with [ee73e5e9] too"
    candidates = ["5d7d253a1122", "ee73e5e9ffff", "deadbeefcafe"]
    assert eval_tokens.surviving_ids(block, candidates) == {"5d7d253a1122", "ee73e5e9ffff"}


def test_lever_row_passed_requires_saving_and_no_quality_drop():
    # 100 -> 90 tokens (10% saving), precision unchanged -> PASS
    good = eval_tokens.LeverRow("compact", "recall_output", 100, 90, 1.0, 1.0)
    assert good.saved_frac == 0.1
    assert good.quality_delta == 0.0
    assert good.passed is True
    # saves tokens but drops precision -> FAIL
    lossy = eval_tokens.LeverRow("aggressive", "recall_output", 100, 50, 1.0, 0.5)
    assert lossy.passed is False
    # keeps precision but no saving -> FAIL
    nosave = eval_tokens.LeverRow("noop", "recall_output", 100, 99, 1.0, 1.0)
    assert nosave.passed is False


def test_measure_recall_sample_scores_surviving_expect_ids():
    off = "**[aaaaaaaa] t**\n> long body here that is bigger\n**[bbbbbbbb] u**"
    on = "**[aaaaaaaa] t**"  # smaller block, but bbbbbbbb dropped
    s = eval_tokens.measure_recall_sample(off, on, expect_ids=["aaaaaaaa11", "bbbbbbbb22"])
    assert s.tokens_on < s.tokens_off
    assert s.prec_off == 1.0  # both expected ids present in off
    assert s.prec_on == 0.5  # only aaaaaaaa survived in on


def test_aggregate_recall_sums_tokens_and_means_precision():
    samples = [
        eval_tokens.P1Sample(tokens_off=100, tokens_on=80, prec_off=1.0, prec_on=1.0),
        eval_tokens.P1Sample(tokens_off=60, tokens_on=60, prec_off=1.0, prec_on=0.0),
    ]
    row = eval_tokens.aggregate_recall("compact", samples)
    assert row.tokens_off == 160 and row.tokens_on == 140
    assert row.quality_off == 1.0
    assert row.quality_on == 0.5


@_dc
class _FakeHit:
    id: str
    title: str
    body: str
    score: float | None = 0.9
    tags: tuple[str, ...] = ()


def test_env_pins_sets_and_restores(monkeypatch):
    monkeypatch.delenv("MEMO_RECALL_FORMAT", raising=False)
    with eval_tokens.env_pins({"MEMO_RECALL_FORMAT": "compact"}):
        import os

        assert os.environ["MEMO_RECALL_FORMAT"] == "compact"
    import os

    assert "MEMO_RECALL_FORMAT" not in os.environ


def test_render_block_verbosity_level_appends_steering():
    hits = [_FakeHit(id="aaaaaaaa11", title="T", body="b")]
    plain = eval_tokens.render_block(hits, {}, body_chars=200, token_budget=200)
    steered = eval_tokens.render_block(
        hits, {"MEMO_RECALL_VERBOSITY_LEVEL": "2"}, body_chars=200, token_budget=200
    )
    # L4 adds a steering block -> steered is LONGER (the paradox: L4 costs P1 tokens)
    assert len(steered) > len(plain)
    assert "aaaaaaaa" in plain  # id short-prefix present for surviving_ids()


def _write_corpus(tmp_path: Path) -> Path:
    corpus = {
        "schema": "memo.token_corpus.v1",
        "cases": [
            {
                "name": "late",
                "must_keep_index": 11,
                "rows": [{"i": i, "text": f"row {i}"} for i in range(11)]
                + [{"i": 11, "text": "THE ANSWER"}],
            },
            {
                "name": "early",
                "must_keep_index": 0,
                "rows": [{"i": 0, "text": "THE ANSWER"}]
                + [{"i": i, "text": f"row {i}"} for i in range(1, 12)],
            },
        ],
    }
    p = tmp_path / "token_corpus.json"
    p.write_text(_json.dumps(corpus), encoding="utf-8")
    return p


def test_load_capture_corpus(tmp_path):
    cases = eval_tokens.load_capture_corpus(_write_corpus(tmp_path))
    assert [c.name for c in cases] == ["late", "early"]
    assert cases[0].must_keep_index == 11
    assert len(cases[1].rows) == 12


def test_measure_crush_case_flags_dropped_answer():
    case = eval_tokens.CaptureCase(
        name="late",
        must_keep_index=11,
        rows=[{"i": i, "text": f"row {i}"} for i in range(11)] + [{"i": 11, "text": "ANSWER"}],
    )

    def crush_fn(content: str) -> tuple[str, str | None]:
        # Simulate a position-only crusher: keep first 10 rows, drop the rest.
        arr = _json.loads(content)
        return _json.dumps(arr[:10]), "hash"

    s = eval_tokens.measure_crush_case(case, crush_fn)
    assert s.tokens_on < s.tokens_off  # crushing saved tokens
    assert s.survived is False  # index 11 was dropped -> quality FAIL

    row = eval_tokens.aggregate_capture("crusher", [s])
    assert row.plane == "capture"
    # Multi-row quality: 10 of 12 rows survived (must_keep row dropped)
    assert row.quality_on == pytest.approx(10 / 12)
    assert row.passed is False  # saved tokens but quality degraded


def test_gate_metrics_snapshots_each_lever():
    rows = [eval_tokens.LeverRow("compact", "recall_output", 100, 80, 1.0, 1.0)]
    m = eval_tokens.gate_metrics(rows)
    assert m["compact"]["passed"] is True
    assert round(m["compact"]["saved_frac"], 3) == 0.2


def test_check_gate_passes_when_no_regression():
    baseline = {"compact": {"saved_frac": 0.2, "quality_delta": 0.0, "passed": True}}
    rows = [eval_tokens.LeverRow("compact", "recall_output", 100, 78, 1.0, 1.0)]  # 22% now
    res = eval_tokens.check_gate(rows, baseline)
    assert res.passed is True


def test_check_gate_fails_when_passing_lever_regresses():
    baseline = {"compact": {"saved_frac": 0.2, "quality_delta": 0.0, "passed": True}}
    rows = [eval_tokens.LeverRow("compact", "recall_output", 100, 95, 1.0, 1.0)]  # 5% now
    res = eval_tokens.check_gate(rows, baseline)
    assert res.passed is False
    assert "compact" in res.message


def test_check_gate_ignores_levers_that_never_passed():
    baseline = {"verbosity": {"saved_frac": -0.1, "quality_delta": 0.0, "passed": False}}
    rows = [eval_tokens.LeverRow("verbosity", "recall_output", 100, 130, 1.0, 1.0)]
    res = eval_tokens.check_gate(rows, baseline)
    assert res.passed is True  # a never-passing lever can't regress


def test_run_all_measures_p1_and_p2_without_mlx(monkeypatch):
    from memo.eval_recall import Prompt

    hits = [_FakeHit(id="aaaaaaaa11", title="Answer", body="the answer body " * 10)]

    def fake_search(text: str) -> list:
        return hits

    def fake_crush(content: str) -> tuple[str, str | None]:
        import json as j

        arr = j.loads(content)
        return j.dumps(arr[:10]), "h"  # position-only: drops late rows

    corpus = [
        eval_tokens.CaptureCase(
            "late",
            must_keep_index=11,
            rows=[{"i": i} for i in range(11)] + [{"i": 11, "answer": True}],
        )
    ]
    rows = eval_tokens.run_all(
        prompts=[Prompt("q", expect_ids=["aaaaaaaa11"])],
        search=fake_search,
        corpus=corpus,
        crush_fn=fake_crush,
    )
    planes = {r.plane for r in rows}
    assert planes == {"recall_output", "capture"}
    # The crusher lever dropped the answer -> capture lever FAILs quality.
    cap = next(r for r in rows if r.plane == "capture")
    assert cap.passed is False


def test_token_count_falls_back_when_the_encoder_raises(monkeypatch):
    """A broken encoder must degrade to chars/4, not fail the measurement.

    `eval_tokens` is a reporting path: an encoder that raises on some input
    should cost accuracy, never the whole run.
    """
    import memo.eval_tokens as et

    def _boom(_t):
        raise RuntimeError("encoder exploded")

    monkeypatch.setattr(et, "count_tokens_accurate", _boom, raising=False)
    text = "x" * 41
    assert et.count_tokens(text) == (len(text) + 4 - 1) // 4


def test_crushed_rows_survived_counts_only_rows_still_present():
    """How many original rows are still present in the crushed JSON."""
    import json

    from memo.eval_tokens import _count_rows_survived

    rows = [{"id": 1}, {"id": 2}, {"id": 3}]
    assert _count_rows_survived(rows, json.dumps([{"id": 1}, {"id": 3}])) == 2
    assert _count_rows_survived(rows, json.dumps(rows)) == 3
    # Anything that is not a JSON array of rows recovers nothing, and says so
    # rather than raising.
    assert _count_rows_survived(rows, "not json at all") == 0
    assert _count_rows_survived(rows, json.dumps({"not": "a list"})) == 0
    assert _count_rows_survived(rows, None) == 0


def test_gate_metrics_records_the_sample_each_lever_was_measured_on():
    """`saved_frac` without its sample size cannot be read as evidence.

    The baseline is the only thing `memo token-savings` sees, so a number
    whose basis is dropped here is unrecoverable downstream.
    """
    from memo import eval_tokens

    rows = [
        eval_tokens.aggregate_capture(
            "crusher_L1",
            [
                eval_tokens.P2Sample(
                    tokens_off=100,
                    tokens_on=50,
                    survived=True,
                    rows_survived=10,
                    rows_total=10,
                )
            ],
        )
    ]
    metrics = eval_tokens.gate_metrics(rows)
    assert metrics["crusher_L1"]["n_samples"] == 1
