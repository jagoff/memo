from memo.flags import flag_bool, flag_float, flag_int


def test_proactive_flags_defaults():
    assert flag_bool("MEMO_PROACTIVE_ENABLED") is False
    assert flag_int("MEMO_PROACTIVE_PUSH_COOLDOWN_H") == 6
    assert flag_int("MEMO_PROACTIVE_DAILY_CAP") == 3
    assert flag_float("MEMO_PROACTIVE_MULT_FLOOR") == 0.2
    assert flag_float("MEMO_PROACTIVE_URGENT_MIN") == 0.7
    assert flag_int("MEMO_PROACTIVE_DIGEST_TOP") == 7
