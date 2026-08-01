"""Tests for the central MEMO_* flag registry (memo.flags)."""

from __future__ import annotations

from pathlib import Path

from memo import flags


def _isolated_env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_CONFIG_DIR": str(tmp_path / "no-md-config"),
        "MEMO_STATE_DIR": str(tmp_path / "no-overlay-state"),
    }


def test_every_spec_has_a_group_and_help() -> None:
    for name, spec in flags.REGISTRY.items():
        assert spec.name == name
        assert spec.group, f"{name} missing group"
        assert spec.help, f"{name} missing help"
        assert spec.kind in ("bool", "int", "float", "str")


def test_no_duplicate_flag_names_across_spec_modules() -> None:
    """A flag defined in two flags_* modules silently shadows the earlier spec
    in REGISTRY (dict comprehension — last wins). Regression: the stale
    MEMO_GRAPH_SEMANTIC_RELATIONS 'stub' spec in flags_behavior.py shadowed
    the live flags_search.py spec."""
    names = [s.name for s in flags._SPECS]
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f"flags defined in more than one flags_* module: {dupes}"
    assert len(flags._SPECS) == len(flags.REGISTRY)


def test_graph_semantic_relations_uses_the_graph_domain_spec() -> None:
    spec = flags.REGISTRY["MEMO_GRAPH_SEMANTIC_RELATIONS"]
    assert spec.group == "graph"
    assert "stub" not in spec.help.lower()


def test_flag_returns_default_when_unset(tmp_path: Path) -> None:
    env = _isolated_env(tmp_path)
    assert flags.flag("MEMO_RECALL_TOP_K", env=env) == 3
    assert flags.flag("MEMO_RECALL_MIN_SIM", env=env) == 0.5
    assert flags.flag("MEMO_RECALL_MODE", env=env) == "vec"
    assert flags.flag("MEMO_RECALL_DISABLE", env=env) is False
    assert flags.flag("MEMO_RECALL_TOKEN_BUDGET", env=env) == 600
    assert flags.flag("MEMO_CONTRADICT_PENALTY_ENABLED", env=env) is False
    assert "MEMO_EMIT_LEDGER" not in flags.REGISTRY
    assert flags.flag_int("MEMO_DREAM_COMPRESS_THRESHOLD", env=env) == 0


def test_graph_integration_flags_have_safe_defaults(tmp_path: Path) -> None:
    env = _isolated_env(tmp_path)
    assert flags.flag_bool("MEMO_GRAPH_SIGNAL_ENABLED", env=env) is False
    assert flags.flag_bool("MEMO_GRAPH_REASON_ENABLED", env=env) is False
    assert flags.flag_bool("MEMO_GRAPH_SEMANTIC_RELATIONS", env=env) is False
    assert flags.flag_bool("MEMO_GRAPH_HUB_SUPPRESSION", env=env) is True
    assert flags.flag_int("MEMO_GRAPH_SIGNAL_BUDGET_MS", env=env) == 150
    assert flags.flag_float("MEMO_GRAPH_HUB_MAX_DOC_FREQ_RATIO", env=env) == 0.25
    assert flags.flag_float("MEMO_GRAPH_MIN_ENTITY_IDF", env=env) == 0.5


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


def test_validate_rejects_invalid_mcp_runtime_values() -> None:
    env = {
        "MEMO_MCP_TRANSPORT": "htpp",
        "MEMO_MCP_PROFILE": "typo",
        "MEMO_MCP_PORT": "0",
    }

    by_flag = {problem["flag"]: problem for problem in flags.validate(env=env)}

    assert "expected one of" in by_flag["MEMO_MCP_TRANSPORT"]["error"]
    assert "expected one of" in by_flag["MEMO_MCP_PROFILE"]["error"]
    assert "must be >= 1" in by_flag["MEMO_MCP_PORT"]["error"]


def test_validate_rejects_mcp_port_above_tcp_limit() -> None:
    problems = flags.validate(env={"MEMO_MCP_PORT": "70000"})

    assert len(problems) == 1
    assert problems[0]["flag"] == "MEMO_MCP_PORT"
    assert "must be <= 65535" in problems[0]["error"]


def test_validate_rejects_garbage_config_owned_typed_vars(tmp_path: Path) -> None:
    """config.py-owned vars are outside REGISTRY, but Config.from_env() feeds
    their raw strings to pydantic — a garbage value must fail `memo config
    validate` instead of passing green and then hard-crashing every command."""
    env = {
        "MEMO_CONFIG_DIR": str(tmp_path),
        "MEMO_EMBEDDER_DIMS": "10z4",
        "MEMO_MAX_CONTENT_CHARS": "64k",
        "MEMO_SEARCH_DEFAULT_LIMIT": "500",
        "MEMO_RERANK_INPUT_K": "lots",
        "MEMO_RERANK_FUSION_ALPHA": "high",
        "MEMO_RERANKER_ENABLED": "maybe",
    }
    by_flag = {p["flag"]: p for p in flags.validate(env=env)}
    for name in env:
        if name == "MEMO_CONFIG_DIR":
            continue
        assert name in by_flag, f"garbage {name} passed validate()"
    assert "must be <= 100" in by_flag["MEMO_SEARCH_DEFAULT_LIMIT"]["error"]


def test_validate_accepts_valid_config_owned_typed_vars(tmp_path: Path) -> None:
    env = {
        "MEMO_CONFIG_DIR": str(tmp_path),
        "MEMO_EMBEDDER_DIMS": "2560",
        "MEMO_MAX_CONTENT_CHARS": "64000",
        "MEMO_SEARCH_DEFAULT_LIMIT": "10",
        "MEMO_RERANK_INPUT_K": "30",
        "MEMO_RERANK_FUSION_ALPHA": "0.7",
        "MEMO_RERANKER_ENABLED": "1",
    }
    assert flags.validate(env=env) == []


def test_config_owned_typed_specs_stay_out_of_registry() -> None:
    # They exist for validate() only; REGISTRY (and active_flags/config show)
    # must keep excluding config.py-owned storage/model vars.
    for spec in flags._CONFIG_OWNED_TYPED_SPECS:
        assert spec.name not in flags.REGISTRY, spec.name
        assert flags.unknown_memo_vars(env={spec.name: "x"}) == []


def test_owned_config_vars_not_flagged_unknown() -> None:
    env = {
        "MEMO_CONFIG_DIR": "/tmp/memo-config",
        "MEMO_DATA_DIR": "/tmp/x",
        "MEMO_RERANKER_REVISION": "abc123",
    }
    assert flags.unknown_memo_vars(env=env) == []


def test_http_api_token_is_owned_without_being_exposed_as_an_active_flag() -> None:
    token = "s" * 32
    env = {"MEMO_HTTP_API_TOKEN": token}

    assert flags.unknown_memo_vars(env=env) == []
    assert flags.validate(env=env) == []
    assert flags.active_flags(env=env) == {}


def test_internal_shim_state_vars_not_flagged_unknown() -> None:
    env = {
        "MEMO_STARTUP_BANNER_SHOWN": "1",
        "MEMO_CODEX_BADGE_SHOWN": "1",
        "MEMO_AGENT_TTY": "/dev/ttys001",
    }
    assert flags.unknown_memo_vars(env=env) == []
    assert flags.validate(env=env) == []


def test_chat_config_vars_not_flagged_unknown() -> None:
    # chat/config.py's 9 MEMO_CHAT_* knobs are env-only (read directly, not
    # through this registry) but must not be reported as typos.
    env = {"MEMO_CHAT_BASE_K": "5"}
    assert flags.unknown_memo_vars(env=env) == []
    assert flags.validate(env=env) == []


def test_active_flags_lists_only_set() -> None:
    env = {"MEMO_RECALL_TOP_K": "5", "MEMO_RECALL_DEBUG": ""}
    active = flags.active_flags(env=env)
    assert active == {"MEMO_RECALL_TOP_K": "5"}


def test_markdown_flag_value_is_used_when_env_unset(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "recall-config.md").write_text(
        '```toml\n[recall]\ntop_k = 8\ndebug = "on"\n```\n',
        encoding="utf-8",
    )
    env = {"MEMO_CONFIG_DIR": str(home)}

    assert flags.flag_int("MEMO_RECALL_TOP_K", env=env) == 8
    assert flags.flag_bool("MEMO_RECALL_DEBUG", env=env) is True


def test_empty_env_var_falls_through_to_markdown_config(tmp_path: Path) -> None:
    """`MEMO_X= memo ...` (empty export) must count as UNSET for the whole
    chain: markdown config > overlay > default. Regression: the empty-string
    guard short-circuited straight to the built-in default, silently masking
    values persisted via `memo config set`."""
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "recall-config.md").write_text("```toml\n[recall]\ntop_k = 8\n```\n", encoding="utf-8")
    env = {"MEMO_CONFIG_DIR": str(home), "MEMO_RECALL_TOP_K": ""}

    assert flags.flag_int("MEMO_RECALL_TOP_K", env=env) == 8


def test_empty_env_var_falls_through_to_overlay(tmp_path: Path) -> None:
    from memo import tuned_overlay as ov

    ov.write_overlay(tmp_path, {"MEMO_RECALL_MIN_SIM": 0.6}, {})
    env = {
        "MEMO_CONFIG_DIR": str(tmp_path / "no-md-config"),
        "MEMO_STATE_DIR": str(tmp_path),
        "MEMO_RECALL_MIN_SIM": "",
    }

    assert flags.flag_float("MEMO_RECALL_MIN_SIM", env=env) == 0.6


def test_empty_env_var_without_config_returns_default(tmp_path: Path) -> None:
    env = {"MEMO_CONFIG_DIR": str(tmp_path / "no-md-config"), "MEMO_RECALL_TOP_K": ""}
    assert flags.flag("MEMO_RECALL_TOP_K", env=env) == 3


def test_empty_env_var_stays_explicit_for_empty_default_str_flags(tmp_path: Path) -> None:
    """A str flag whose default is "" keeps treating an empty env var as an
    explicit empty value (it does NOT fall through to markdown config)."""
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "advanced-config.md").write_text(
        '```toml\n[misc]\nproject_tag = "pinned"\n```\n', encoding="utf-8"
    )
    env = {"MEMO_CONFIG_DIR": str(home), "MEMO_PROJECT_TAG": ""}

    assert flags.flag_str("MEMO_PROJECT_TAG", env=env) == ""


def test_env_flag_overrides_markdown_flag(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "recall-config.md").write_text("```toml\n[recall]\ntop_k = 8\n```\n", encoding="utf-8")

    assert (
        flags.flag_int(
            "MEMO_RECALL_TOP_K",
            env={"MEMO_CONFIG_DIR": str(home), "MEMO_RECALL_TOP_K": "2"},
        )
        == 2
    )


def test_active_flags_remains_env_only_and_active_config_values_reads_markdown(
    tmp_path: Path,
) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "recall-config.md").write_text(
        "```toml\n[recall]\ntop_k = 8\ndebug = true\n```\n", encoding="utf-8"
    )
    env = {"MEMO_CONFIG_DIR": str(home), "MEMO_RECALL_DEBUG": "1"}

    assert flags.active_flags(env=env) == {"MEMO_RECALL_DEBUG": "1"}
    assert flags.active_config_values(env=env) == {
        "MEMO_RECALL_TOP_K": "8",
        "MEMO_RECALL_DEBUG": "on",
    }


def test_validate_reports_markdown_problems(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "recall-config.md").write_text("```toml\n[recall]\ntoppp_k = 8\n```\n", encoding="utf-8")

    problems = flags.validate(env={"MEMO_CONFIG_DIR": str(home)})

    assert any(p["flag"] == "recall.toppp_k" and "unknown" in p["error"] for p in problems)
