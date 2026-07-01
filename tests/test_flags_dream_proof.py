def test_proof_loop_flags_registered_with_defaults(monkeypatch):
    # ensure no env override leaks in
    monkeypatch.delenv("MEMO_DREAM_TUNE_MIN_COHORT", raising=False)
    monkeypatch.delenv("MEMO_DREAM_TUNE_ONLINE_EPS", raising=False)
    from memo.flags import flag_float, flag_int

    assert flag_int("MEMO_DREAM_TUNE_MIN_COHORT") == 20
    assert flag_float("MEMO_DREAM_TUNE_ONLINE_EPS") == 0.02
