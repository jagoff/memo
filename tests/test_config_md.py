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
        'debug = "on"\n'
        'disable = "no"\n'
        "```\n",
        encoding="utf-8",
    )

    vals = config_md.flag_values({"MEMO_CONFIG_DIR": str(home)})

    assert vals["MEMO_RECALL_DEBUG"] == "1"
    assert vals["MEMO_RECALL_DISABLE"] == "0"


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
