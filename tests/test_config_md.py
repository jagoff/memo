from __future__ import annotations

from pathlib import Path

from memo import config_md


def test_config_home_uses_memo_config_dir(tmp_path: Path) -> None:
    env = {"MEMO_CONFIG_DIR": str(tmp_path / "memo-home")}
    assert config_md.config_home(env) == (tmp_path / "memo-home").resolve()
    assert config_md.index_path(env) == (tmp_path / "memo-home" / "memo-config.md").resolve()
    assert config_md.config_dir(env) == (tmp_path / "memo-home" / "config").resolve()


def test_load_values_reads_fenced_toml_blocks(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "recall-config.md").write_text(
        "# Recall\n\n"
        "Human notes are ignored.\n\n"
        "```toml\n"
        "[recall]\n"
        'top_k = 5\n'
        'disable = "off"\n'
        "```\n",
        encoding="utf-8",
    )

    values = config_md.load_values({"MEMO_CONFIG_DIR": str(home)})

    assert values["recall.top_k"].value == 5
    assert values["recall.top_k"].env_name == "MEMO_RECALL_TOP_K"
    assert values["recall.disable"].value == "off"
    assert values["recall.disable"].env_name == "MEMO_RECALL_DISABLE"


def test_index_toml_is_read_and_validated(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    home.mkdir()
    (home / "memo-config.md").write_text(
        "```toml\n[recall]\ntop_k = 7\n```\n", encoding="utf-8"
    )
    env = {"MEMO_CONFIG_DIR": str(home)}

    assert config_md.flag_values(env)["MEMO_RECALL_TOP_K"] == "7"
    assert config_md.validate_markdown_config(env) == []


def test_invalid_index_value_is_validated_before_domain_override(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (home / "memo-config.md").write_text(
        '```toml\n[recall]\ntop_k = "bad"\n```\n', encoding="utf-8"
    )
    (cfg / "recall-config.md").write_text(
        "```toml\n[recall]\ntop_k = 5\n```\n", encoding="utf-8"
    )
    env = {"MEMO_CONFIG_DIR": str(home)}

    problems = config_md.validate_markdown_config(env)

    assert any(
        p.file.endswith("memo-config.md")
        and p.key == "recall.top_k"
        and "invalid literal" in p.error
        for p in problems
    )
    assert config_md.flag_values(env)["MEMO_RECALL_TOP_K"] == "5"


def test_multiple_toml_blocks_merge(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "storage-config.md").write_text(
        "```toml\n"
        "[storage]\n"
        'data_dir = "/tmp/memo-data"\n'
        "```\n\n"
        "```toml\n"
        "[storage]\n"
        'state_dir = "/tmp/memo-state"\n'
        "```\n",
        encoding="utf-8",
    )

    fields = config_md.field_values({"MEMO_CONFIG_DIR": str(home)})

    assert fields == {"data_dir": "/tmp/memo-data", "state_dir": "/tmp/memo-state"}


def test_boolean_spellings_are_normalized_for_flags(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "recall-config.md").write_text(
        "```toml\n"
        "[recall]\n"
        "debug = true\n"
        'disable = "false"\n'
        'metrics = "1"\n'
        'force_mode = "0"\n'
        'associative = "yes"\n'
        'system_message = "no"\n'
        'cite_instruction = "on"\n'
        "```\n",
        encoding="utf-8",
    )

    vals = config_md.flag_values({"MEMO_CONFIG_DIR": str(home)})

    assert vals["MEMO_RECALL_DEBUG"] == "on"
    assert vals["MEMO_RECALL_DISABLE"] == "off"
    assert vals["MEMO_RECALL_METRICS"] == "on"
    assert vals["MEMO_RECALL_FORCE_MODE"] == "off"
    assert vals["MEMO_RECALL_ASSOCIATIVE"] == "on"
    assert vals["MEMO_RECALL_SYSTEM_MESSAGE"] == "off"
    assert vals["MEMO_RECALL_CITE_INSTRUCTION"] == "on"


def test_invalid_boolean_flag_reports_problem(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "recall-config.md").write_text(
        "```toml\n"
        "[recall]\n"
        'debug = "perhaps"\n'
        "```\n",
        encoding="utf-8",
    )

    problems = config_md.validate_markdown_config({"MEMO_CONFIG_DIR": str(home)})

    assert any(
        p.key == "recall.debug" and "expected a boolean" in p.error for p in problems
    )


def test_out_of_bounds_numeric_flag_reports_problem(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "recall-config.md").write_text(
        "```toml\n"
        "[recall]\n"
        "min_sim = -1\n"
        "```\n",
        encoding="utf-8",
    )

    problems = config_md.validate_markdown_config({"MEMO_CONFIG_DIR": str(home)})

    assert any(
        p.key == "recall.min_sim" and "MEMO_RECALL_MIN_SIM must be >= 0.0" in p.error
        for p in problems
    )


def test_invalid_config_field_reports_problem(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "search-config.md").write_text(
        "```toml\n"
        "[search]\n"
        "default_limit = 0\n"
        "```\n",
        encoding="utf-8",
    )

    problems = config_md.validate_markdown_config({"MEMO_CONFIG_DIR": str(home)})

    assert any(
        p.key == "search.default_limit" and "greater than or equal to 1" in p.error
        for p in problems
    )


def test_load_values_accepts_fence_without_final_newline(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "recall-config.md").write_text(
        "```toml\n"
        "[recall]\n"
        "top_k = 5```",
        encoding="utf-8",
    )

    values = config_md.load_values({"MEMO_CONFIG_DIR": str(home)})

    assert values["recall.top_k"].value == 5


def test_unknown_key_reports_problem(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "recall-config.md").write_text(
        "```toml\n"
        "[recall]\n"
        'toppp_k = 5\n'
        "```\n",
        encoding="utf-8",
    )

    problems = config_md.validate_markdown_config({"MEMO_CONFIG_DIR": str(home)})

    assert any(p.key == "recall.toppp_k" and "unknown" in p.error for p in problems)


def test_invalid_toml_reports_problem(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "recall-config.md").write_text("```toml\n[recall\nbroken = \n```\n", encoding="utf-8")

    problems = config_md.validate_markdown_config({"MEMO_CONFIG_DIR": str(home)})

    assert any(p.file.endswith("recall-config.md") and "TOML" in p.error for p in problems)


def test_unknown_config_file_warns(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "custom-config.md").write_text("```toml\n[custom]\nvalue = 1\n```\n", encoding="utf-8")
    (cfg / "notes.md").write_text("ignored notes\n", encoding="utf-8")

    problems = config_md.validate_markdown_config({"MEMO_CONFIG_DIR": str(home)})

    assert any(p.file.endswith("custom-config.md") and "unknown config file" in p.error for p in problems)
    assert not any(p.file.endswith("notes.md") for p in problems)
