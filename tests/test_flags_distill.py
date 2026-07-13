from memo.flags import flag_bool, flag_float, flag_int


def test_distill_pass_defaults_off(monkeypatch):
    monkeypatch.delenv("MEMO_DREAM_DISTILL_ENABLED", raising=False)
    assert flag_bool("MEMO_DREAM_DISTILL_ENABLED") is False


def test_distill_tuning_defaults(monkeypatch):
    for name in (
        "MEMO_DREAM_DISTILL_MIN_CLUSTER",
        "MEMO_DREAM_DISTILL_MIN_SUPPORT",
        "MEMO_DREAM_DISTILL_MIN_AGE_DAYS",
        "MEMO_DREAM_DISTILL_MAX",
    ):
        monkeypatch.delenv(name, raising=False)
    assert flag_int("MEMO_DREAM_DISTILL_MIN_CLUSTER") == 3
    assert flag_int("MEMO_DREAM_DISTILL_MIN_SUPPORT") == 2
    assert flag_int("MEMO_DREAM_DISTILL_MIN_AGE_DAYS") == 14
    assert flag_int("MEMO_DREAM_DISTILL_MAX") == 5


def test_distill_threshold_and_min_confidence_defaults(monkeypatch):
    monkeypatch.delenv("MEMO_DREAM_DISTILL_THRESHOLD", raising=False)
    monkeypatch.delenv("MEMO_DREAM_DISTILL_MIN_CONFIDENCE", raising=False)
    assert flag_float("MEMO_DREAM_DISTILL_THRESHOLD") == 0.78
    assert flag_float("MEMO_DREAM_DISTILL_MIN_CONFIDENCE") == 0.5


def test_distill_threshold_and_min_confidence_override(monkeypatch):
    monkeypatch.setenv("MEMO_DREAM_DISTILL_THRESHOLD", "0.9")
    monkeypatch.setenv("MEMO_DREAM_DISTILL_MIN_CONFIDENCE", "0.7")
    assert flag_float("MEMO_DREAM_DISTILL_THRESHOLD") == 0.9
    assert flag_float("MEMO_DREAM_DISTILL_MIN_CONFIDENCE") == 0.7


def test_recall_altitude_defaults_off(monkeypatch):
    monkeypatch.delenv("MEMO_RECALL_ALTITUDE", raising=False)
    # float flag: 0.0 (falsy) = OFF. flag_float returns 0.0 or None -> normalize.
    assert (flag_float("MEMO_RECALL_ALTITUDE") or 0.0) == 0.0


def test_distill_flags_are_registered():
    from memo.flags import REGISTRY

    for expected in (
        "MEMO_DREAM_DISTILL_ENABLED",
        "MEMO_DREAM_DISTILL_MIN_CLUSTER",
        "MEMO_DREAM_DISTILL_MIN_SUPPORT",
        "MEMO_DREAM_DISTILL_MIN_AGE_DAYS",
        "MEMO_DREAM_DISTILL_MAX",
        "MEMO_DREAM_DISTILL_THRESHOLD",
        "MEMO_DREAM_DISTILL_MIN_CONFIDENCE",
        "MEMO_RECALL_ALTITUDE",
    ):
        assert expected in REGISTRY
