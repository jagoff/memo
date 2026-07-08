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
