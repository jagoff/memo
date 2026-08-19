from memo.flags import REGISTRY, flag_bool, flag_float, flag_int, flag_str
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
    assert flag_str("MEMO_PROXY_TOOL_SCHEMAS_SCOPE") == "all"


def test_holdout_fraction_is_bounded():
    spec = next(s for s in SPECS if s.name == "MEMO_PROXY_HOLDOUT_FRAC")
    assert spec.min_val == 0.0
    assert spec.max_val == 0.5


def test_tool_schemas_scope_only_accepts_all_or_memo():
    spec = next(s for s in SPECS if s.name == "MEMO_PROXY_TOOL_SCHEMAS_SCOPE")
    assert spec.choices == ("all", "memo")


def test_tool_schemas_scope_can_be_set_to_the_conservative_fallback(monkeypatch):
    monkeypatch.setenv("MEMO_PROXY_TOOL_SCHEMAS_SCOPE", "memo")
    assert flag_str("MEMO_PROXY_TOOL_SCHEMAS_SCOPE") == "memo"


def test_content_scope_defaults_to_all(monkeypatch):
    monkeypatch.delenv("MEMO_PROXY_CONTENT_SCOPE", raising=False)
    assert flag_str("MEMO_PROXY_CONTENT_SCOPE") == "all"


def test_content_scope_only_accepts_all_or_tail():
    spec = next(s for s in SPECS if s.name == "MEMO_PROXY_CONTENT_SCOPE")
    assert spec.choices == ("all", "tail")


def test_content_scope_can_be_set_to_the_tail_only_fallback(monkeypatch):
    monkeypatch.setenv("MEMO_PROXY_CONTENT_SCOPE", "tail")
    assert flag_str("MEMO_PROXY_CONTENT_SCOPE") == "tail"
