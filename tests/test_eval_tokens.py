import json as _json
from dataclasses import dataclass as _dc
from pathlib import Path

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
            {"name": "late", "must_keep_index": 11,
             "rows": [{"i": i, "text": f"row {i}"} for i in range(11)]
                     + [{"i": 11, "text": "THE ANSWER"}]},
            {"name": "early", "must_keep_index": 0,
             "rows": [{"i": 0, "text": "THE ANSWER"}]
                     + [{"i": i, "text": f"row {i}"} for i in range(1, 12)]},
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
        name="late", must_keep_index=11,
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
    assert row.quality_on == 0.0
    assert row.passed is False  # saved tokens but dropped the answer
