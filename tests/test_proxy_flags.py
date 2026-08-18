from memo.flags import REGISTRY, flag_bool, flag_float, flag_int
from memo.flags_proxy import SPECS


def test_every_proxy_flag_is_registered_globally():
    for spec in SPECS:
        assert spec.name in REGISTRY, f"{spec.name} missing from memo.flags.REGISTRY"


def test_proxy_defaults(monkeypatch):
    for spec in SPECS:
        monkeypatch.delenv(spec.name, raising=False)
    assert flag_bool("MEMO_PROXY_ENABLED") is True
    assert flag_int("MEMO_PROXY_PORT") == 8768
    assert flag_float("MEMO_PROXY_HOLDOUT_FRAC") == 0.05
    assert flag_int("MEMO_PROXY_TOOL_WINDOW_SESSIONS") == 20


def test_holdout_fraction_is_bounded():
    spec = next(s for s in SPECS if s.name == "MEMO_PROXY_HOLDOUT_FRAC")
    assert spec.min_val == 0.0
    assert spec.max_val == 0.5
