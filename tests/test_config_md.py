from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

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
        "top_k = 5\n"
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
    (home / "memo-config.md").write_text("```toml\n[recall]\ntop_k = 7\n```\n", encoding="utf-8")
    env = {"MEMO_CONFIG_DIR": str(home)}

    assert config_md.flag_values(env)["MEMO_RECALL_TOP_K"] == "7"
    assert config_md.validate_markdown_config(env) == []


def test_markdown_flags_do_not_cycle_during_config_import(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "recall-config.md").write_text(
        "```toml\n[recall]\ntop_k = 7\n```\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "MEMO_CONFIG_DIR": str(home),
        "MEMO_CONFIG_FILE": str(tmp_path / "missing.toml"),
    }

    proc = subprocess.run(
        [sys.executable, "-c", "from memo.config import Config; print(Config.__name__)"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "Config"


def test_invalid_index_value_is_validated_before_domain_override(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (home / "memo-config.md").write_text(
        '```toml\n[recall]\ntop_k = "bad"\n```\n', encoding="utf-8"
    )
    (cfg / "recall-config.md").write_text("```toml\n[recall]\ntop_k = 5\n```\n", encoding="utf-8")
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
        '```toml\n[recall]\ndebug = "perhaps"\n```\n',
        encoding="utf-8",
    )

    problems = config_md.validate_markdown_config({"MEMO_CONFIG_DIR": str(home)})

    assert any(p.key == "recall.debug" and "expected a boolean" in p.error for p in problems)


def test_out_of_bounds_numeric_flag_reports_problem(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "recall-config.md").write_text(
        "```toml\n[recall]\nmin_sim = -1\n```\n",
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
        "```toml\n[search]\ndefault_limit = 0\n```\n",
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
        "```toml\n[recall]\ntop_k = 5```",
        encoding="utf-8",
    )

    values = config_md.load_values({"MEMO_CONFIG_DIR": str(home)})

    assert values["recall.top_k"].value == 5


def test_unknown_key_reports_problem(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "recall-config.md").write_text(
        "```toml\n[recall]\ntoppp_k = 5\n```\n",
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

    assert any(
        p.file.endswith("custom-config.md") and "unknown config file" in p.error for p in problems
    )
    assert not any(p.file.endswith("notes.md") for p in problems)


def test_write_default_config_creates_index_and_domain_files(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    written = config_md.write_default_config(
        data_dir=tmp_path / "data",
        vault_path=None,
        env={"MEMO_CONFIG_DIR": str(home)},
    )

    assert home.joinpath("memo-config.md").is_file()
    assert home.joinpath("config", "storage-config.md").is_file()
    assert home.joinpath("config", "models-config.md").is_file()
    assert home.joinpath("config", "recall-config.md").is_file()
    assert any(path.name == "storage-config.md" for path in written)
    assert 'data_dir = "' in home.joinpath("config", "storage-config.md").read_text(
        encoding="utf-8"
    )
    assert 'disable = "off"' in home.joinpath("config", "recall-config.md").read_text(
        encoding="utf-8"
    )


def test_write_default_config_refuses_overwrite_without_force(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    config_md.write_default_config(
        data_dir=tmp_path / "data",
        env={"MEMO_CONFIG_DIR": str(home)},
    )

    try:
        config_md.write_default_config(
            data_dir=tmp_path / "other",
            env={"MEMO_CONFIG_DIR": str(home)},
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected FileExistsError")


def test_set_and_unset_value_rewrite_domain_block(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    config_md.write_default_config(data_dir=tmp_path / "data", env={"MEMO_CONFIG_DIR": str(home)})

    changed = config_md.set_value("recall.top_k", "9", env={"MEMO_CONFIG_DIR": str(home)})
    assert changed == home / "config" / "recall-config.md"
    assert config_md.flag_values({"MEMO_CONFIG_DIR": str(home)})["MEMO_RECALL_TOP_K"] == "9"

    config_md.unset_value("recall.top_k", env={"MEMO_CONFIG_DIR": str(home)})
    assert "MEMO_RECALL_TOP_K" not in config_md.flag_values({"MEMO_CONFIG_DIR": str(home)})


def test_runtime_only_key_is_rejected_in_markdown(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "advanced-config.md").write_text(
        '```toml\n[misc]\nnoninteractive = "on"\n```\n',
        encoding="utf-8",
    )

    problems = config_md.validate_markdown_config({"MEMO_CONFIG_DIR": str(home)})

    assert any("runtime-only" in problem.error for problem in problems)
    assert "misc.noninteractive" not in config_md.load_values({"MEMO_CONFIG_DIR": str(home)})


def test_set_value_refuses_runtime_only_key(tmp_path: Path) -> None:
    env = {"MEMO_CONFIG_DIR": str(tmp_path / "memo-home")}

    with pytest.raises(ValueError, match="runtime-only"):
        config_md.set_value("misc.noninteractive", "on", env)
    with pytest.raises(ValueError, match="runtime-only"):
        config_md.unset_value("misc.noninteractive", env)


def test_configured_values_keeps_source_file(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    path = cfg / "recall-config.md"
    path.write_text("```toml\n[recall]\ntop_k = 7\n```\n", encoding="utf-8")

    value = config_md.configured_values({"MEMO_CONFIG_DIR": str(home)})["recall.top_k"]

    assert value.value == 7
    assert value.file == str(path)


def test_legacy_search_graph_key_loads_as_canonical_graph_key(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    path = cfg / "search-config.md"
    path.write_text(
        "```toml\n[search]\ngraph_signal_enabled = true\n```\n",
        encoding="utf-8",
    )

    values = config_md.load_values({"MEMO_CONFIG_DIR": str(home)})

    assert values["graph.signal_enabled"].value is True
    assert values["graph.signal_enabled"].file == str(path)
    assert config_md.flag_values({"MEMO_CONFIG_DIR": str(home)}) == {
        "MEMO_GRAPH_SIGNAL_ENABLED": "on"
    }


def test_legacy_recall_graph_keys_load_as_curated_signal_keys(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    config_dir = home / "config"
    config_dir.mkdir(parents=True)
    path = config_dir / "recall-config.md"
    path.write_text(
        "```toml\n[recall]\ngraph_proximity = true\ngraph_proximity_weight = 0.25\n```\n",
        encoding="utf-8",
    )

    values = config_md.load_values({"MEMO_CONFIG_DIR": str(home)})

    assert values["graph.signal_enabled"].value is True
    assert values["graph.signal_alpha"].value == 0.25
    assert config_md.flag_values({"MEMO_CONFIG_DIR": str(home)}) == {
        "MEMO_GRAPH_SIGNAL_ALPHA": "0.25",
        "MEMO_GRAPH_SIGNAL_ENABLED": "on",
    }


def test_set_graph_signal_writes_graph_domain_file(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    config_md.write_default_config(
        data_dir=tmp_path / "data",
        env={"MEMO_CONFIG_DIR": str(home)},
    )

    changed = config_md.set_value(
        "graph.signal_enabled",
        "on",
        env={"MEMO_CONFIG_DIR": str(home)},
    )

    assert changed == home / "config" / "graph-config.md"
    assert "signal_enabled = \"on\"" in changed.read_text(encoding="utf-8")


def test_trace_discovery_and_synthesis_switches_route_to_graph_config(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    config_md.write_default_config(
        data_dir=tmp_path / "data",
        env={"MEMO_CONFIG_DIR": str(home)},
    )

    keys = (
        "graph.code_trace_enabled",
        "graph.discovery_enabled",
        "graph.dream_communities_enabled",
        "graph.dream_bridges_enabled",
    )
    paths = {
        config_md.set_value(key, "on", env={"MEMO_CONFIG_DIR": str(home)}) for key in keys
    }

    assert paths == {home / "config" / "graph-config.md"}
    values = config_md.flag_values({"MEMO_CONFIG_DIR": str(home)})
    assert values["MEMO_GRAPH_CODE_TRACE_ENABLED"] == "on"
    assert values["MEMO_GRAPH_DISCOVERY_ENABLED"] == "on"
    assert values["MEMO_DREAM_COMMUNITIES_ENABLED"] == "on"
    assert values["MEMO_DREAM_BRIDGES_ENABLED"] == "on"
