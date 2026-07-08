from memo import eval_tokens


def test_count_tokens_is_ceil_chars_over_four():
    assert eval_tokens.count_tokens("") == 0
    assert eval_tokens.count_tokens("abcd") == 1
    assert eval_tokens.count_tokens("abcde") == 2  # 5 chars -> ceil(5/4) == 2


def test_surviving_ids_matches_eight_char_prefix_in_block():
    block = "**[5d7d253a] Some title**\n> body text with [ee73e5e9] too"
    candidates = ["5d7d253a1122", "ee73e5e9ffff", "deadbeefcafe"]
    assert eval_tokens.surviving_ids(block, candidates) == {"5d7d253a1122", "ee73e5e9ffff"}
