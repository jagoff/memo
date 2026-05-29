"""Tests for the central MEMO_* flag registry (memo.flags)."""

from __future__ import annotations

from memo import flags


def test_every_spec_has_a_group_and_help() -> None:
    for name, spec in flags.REGISTRY.items():
        assert spec.name == name
        assert spec.group, f"{name} missing group"
        assert spec.help, f"{name} missing help"
        assert spec.kind in ("bool", "int", "float", "str")


def test_flag_returns_default_when_unset() -> None:
    env: dict[str, str] = {}
    assert flags.flag("MEMO_RECALL_TOP_K", env=env) == 3
    assert flags.flag("MEMO_RECALL_MIN_SIM", env=env) == 0.6
    assert flags.flag("MEMO_RECALL_MODE", env=env) == "vec"
    assert flags.flag("MEMO_RECALL_DISABLE", env=env) is False
    # opt-out default-on bool
    assert flags.flag("MEMO_EMIT_LEDGER", env=env) is True


def test_typed_coercion() -> None:
    env = {
        "MEMO_RECALL_TOP_K": "7",
        "MEMO_RECALL_MIN_SIM": "0.42",
        "MEMO_RECALL_DISABLE": "true",
        "MEMO_RECALL_MODE": "hybrid",
    }
    assert flags.flag_int("MEMO_RECALL_TOP_K", env=env) == 7
    assert flags.flag_float("MEMO_RECALL_MIN_SIM", env=env) == 0.42
    assert flags.flag_bool("MEMO_RECALL_DISABLE", env=env) is True
    assert flags.flag_str("MEMO_RECALL_MODE", env=env) == "hybrid"


def test_bool_spellings() -> None:
    for truthy in ("1", "true", "YES", "on"):
        assert flags.flag_bool("MEMO_RECALL_DEBUG", env={"MEMO_RECALL_DEBUG": truthy}) is True
    for falsy in ("0", "false", "no", "off"):
        assert flags.flag_bool("MEMO_RECALL_DEBUG", env={"MEMO_RECALL_DEBUG": falsy}) is False


def test_bad_value_falls_back_to_default() -> None:
    # flag() is lenient (returns default) so a typo never crashes a hot path
    assert flags.flag("MEMO_RECALL_TOP_K", env={"MEMO_RECALL_TOP_K": "abc"}) == 3


def test_validate_flags_bad_int_and_unknown_var() -> None:
    env = {"MEMO_RECALL_TOP_K": "abc", "MEMO_TYPO_FLAG": "1"}
    problems = flags.validate(env=env)
    by_flag = {p["flag"]: p for p in problems}
    assert "MEMO_RECALL_TOP_K" in by_flag
    assert "MEMO_TYPO_FLAG" in by_flag
    assert "unknown" in by_flag["MEMO_TYPO_FLAG"]["error"]


def test_validate_clean_env_is_empty() -> None:
    assert flags.validate(env={"MEMO_RECALL_TOP_K": "5", "MEMO_RECALL_MODE": "vec"}) == []


def test_owned_config_vars_not_flagged_unknown() -> None:
    env = {"MEMO_DATA_DIR": "/tmp/x", "MEMO_RERANKER_REVISION": "abc123"}
    assert flags.unknown_memo_vars(env=env) == []


def test_active_flags_lists_only_set() -> None:
    env = {"MEMO_RECALL_TOP_K": "5", "MEMO_RECALL_DEBUG": ""}
    active = flags.active_flags(env=env)
    assert active == {"MEMO_RECALL_TOP_K": "5"}
