from memo.flags import REGISTRY, flag_bool, flag_int


def test_interject_enabled_defaults_off(monkeypatch):
    monkeypatch.delenv("MEMO_INTERJECT_ENABLED", raising=False)
    assert flag_bool("MEMO_INTERJECT_ENABLED") is False


def test_ask_gaps_enabled_defaults_off(monkeypatch):
    monkeypatch.delenv("MEMO_ASK_GAPS_ENABLED", raising=False)
    assert flag_bool("MEMO_ASK_GAPS_ENABLED") is False


def test_interject_max_per_session_default(monkeypatch):
    monkeypatch.delenv("MEMO_INTERJECT_MAX_PER_SESSION", raising=False)
    assert flag_int("MEMO_INTERJECT_MAX_PER_SESSION") == 1


def test_phase3_flags_registered():
    for name in (
        "MEMO_INTERJECT_ENABLED",
        "MEMO_INTERJECT_MAX_PER_SESSION",
        "MEMO_ASK_GAPS_ENABLED",
    ):
        assert name in REGISTRY
