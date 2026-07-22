"""Tuned-params overlay + env > overlay > default flag resolution."""

from __future__ import annotations

from pathlib import Path

from memo import flags
from memo import tuned_overlay as ov


def test_write_then_read_roundtrip(tmp_path: Path):
    ov.write_overlay(tmp_path, {"MEMO_RECALL_MIN_SIM": 0.6}, {"set_by": "dream"})
    doc = ov.read_overlay(tmp_path)
    assert doc["MEMO_RECALL_MIN_SIM"] == 0.6
    assert doc["_meta"]["set_by"] == "dream"


def test_overlay_values_resolves_from_state_dir(tmp_path: Path):
    ov.write_overlay(tmp_path, {"MEMO_RECALL_MIN_SIM": 0.7}, {})
    vals = ov.overlay_values({"MEMO_STATE_DIR": str(tmp_path)})
    assert vals["MEMO_RECALL_MIN_SIM"] == "0.7"


def test_overlay_values_fall_back_to_default_state_dir(tmp_path: Path, monkeypatch):
    """Without MEMO_STATE_DIR exported, the overlay must still resolve through
    Config's fallback chain — daemons/CLI runs used to silently ignore every
    tuner result unless the env var happened to be set. The chain only engages
    for the REAL os.environ; custom env mappings stay hermetic."""
    import os

    import memo.config as config_mod

    ov._state_dir_cache.clear()
    monkeypatch.chdir(tmp_path)  # not a repo checkout
    default_sd = tmp_path / "default-state"
    monkeypatch.setattr(config_mod, "_DEFAULT_STATE_DIR", default_sd)
    monkeypatch.delenv("MEMO_STATE_DIR", raising=False)
    monkeypatch.delenv("MEMO_VAULT_PATH", raising=False)
    monkeypatch.delenv("MEMO_MEMORY_SUBDIR", raising=False)
    monkeypatch.setenv("MEMO_CONFIG_DIR", str(tmp_path / "empty-config"))
    ov.write_overlay(default_sd, {"MEMO_RECALL_MIN_SIM": 0.65}, {"set_by": "dream"})
    try:
        assert ov.overlay_values(os.environ)["MEMO_RECALL_MIN_SIM"] == "0.65"
    finally:
        ov._state_dir_cache.clear()


def test_overlay_values_fall_back_to_repo_state_dir(tmp_path: Path, monkeypatch):
    """A repo-checkout cwd (dev clone) resolves to ./.memo-state, mirroring
    Config.from_env."""
    import os

    ov._state_dir_cache.clear()
    (tmp_path / "src" / "memo").mkdir(parents=True)
    (tmp_path / "src" / "memo" / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MEMO_STATE_DIR", raising=False)
    monkeypatch.delenv("MEMO_VAULT_PATH", raising=False)
    monkeypatch.delenv("MEMO_MEMORY_SUBDIR", raising=False)
    monkeypatch.setenv("MEMO_CONFIG_DIR", str(tmp_path / "empty-config"))
    ov.write_overlay(tmp_path / ".memo-state", {"MEMO_RECALL_MIN_SIM": 0.61}, {})
    try:
        assert ov.overlay_values(os.environ)["MEMO_RECALL_MIN_SIM"] == "0.61"
    finally:
        ov._state_dir_cache.clear()


def test_overlay_repo_cwd_skipped_when_vault_path_exported_empty(tmp_path: Path, monkeypatch):
    """An EXPORTED-but-empty MEMO_VAULT_PATH is key-present, which Config.from_env
    treats as legacy config (opting OUT of repo-cwd mode). The overlay resolver
    must mirror that (key presence, not truthiness) and use the default state
    dir — not ./.memo-state — even inside a repo checkout."""
    import os

    import memo.config as config_mod

    ov._state_dir_cache.clear()
    (tmp_path / "src" / "memo").mkdir(parents=True)
    (tmp_path / "src" / "memo" / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    default_sd = tmp_path / "default-state"
    monkeypatch.setattr(config_mod, "_DEFAULT_STATE_DIR", default_sd)
    monkeypatch.delenv("MEMO_STATE_DIR", raising=False)
    monkeypatch.delenv("MEMO_MEMORY_SUBDIR", raising=False)
    monkeypatch.setenv("MEMO_VAULT_PATH", "")  # exported empty → still legacy
    monkeypatch.setenv("MEMO_CONFIG_DIR", str(tmp_path / "empty-config"))
    # A repo-cwd overlay must be IGNORED; the default-dir one is authoritative.
    ov.write_overlay(tmp_path / ".memo-state", {"MEMO_RECALL_MIN_SIM": 0.42}, {})
    ov.write_overlay(default_sd, {"MEMO_RECALL_MIN_SIM": 0.66}, {"set_by": "dream"})
    try:
        assert ov.overlay_values(os.environ)["MEMO_RECALL_MIN_SIM"] == "0.66"
    finally:
        ov._state_dir_cache.clear()


def test_overlay_repo_cwd_skipped_by_legacy_toml_storage(tmp_path: Path, monkeypatch):
    """A legacy config.toml `[storage]` section is Config.from_env's
    has_storage_config — it opts out of repo-cwd mode AND provides the
    state_dir. The overlay resolver must load the TOML and honor it, not fall
    to ./.memo-state."""
    import os

    ov._state_dir_cache.clear()
    (tmp_path / "src" / "memo").mkdir(parents=True)
    (tmp_path / "src" / "memo" / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    toml_state = tmp_path / "toml-state"
    config_toml = tmp_path / "config.toml"
    config_toml.write_text(f'[storage]\nstate_dir = "{toml_state}"\n', encoding="utf-8")
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(config_toml))
    monkeypatch.delenv("MEMO_STATE_DIR", raising=False)
    monkeypatch.delenv("MEMO_VAULT_PATH", raising=False)
    monkeypatch.delenv("MEMO_MEMORY_SUBDIR", raising=False)
    monkeypatch.setenv("MEMO_CONFIG_DIR", str(tmp_path / "empty-config"))
    ov.write_overlay(tmp_path / ".memo-state", {"MEMO_RECALL_MIN_SIM": 0.42}, {})
    ov.write_overlay(toml_state, {"MEMO_RECALL_MIN_SIM": 0.71}, {"set_by": "dream"})
    try:
        assert ov.overlay_values(os.environ)["MEMO_RECALL_MIN_SIM"] == "0.71"
    finally:
        ov._state_dir_cache.clear()


def test_overlay_values_custom_env_without_state_dir_is_empty():
    """Custom env mappings (hermetic test contract) do NOT engage the
    machine-level fallback chain."""
    assert ov.overlay_values({}) == {}


def test_overlay_values_missing_overlay_file_is_empty(tmp_path: Path):
    assert ov.overlay_values({"MEMO_STATE_DIR": str(tmp_path / "nowhere")}) == {}


def test_corrupt_overlay_is_ignored(tmp_path: Path):
    ov.overlay_path(tmp_path).write_text("{not json", encoding="utf-8")
    assert ov.overlay_values({"MEMO_STATE_DIR": str(tmp_path)}) == {}


def test_write_preserves_prev_then_rollback(tmp_path: Path):
    ov.write_overlay(tmp_path, {"MEMO_RECALL_MIN_SIM": 0.5}, {})
    ov.write_overlay(tmp_path, {"MEMO_RECALL_MIN_SIM": 0.6}, {})
    assert ov.read_overlay(tmp_path)["_meta"]["prev"]["MEMO_RECALL_MIN_SIM"] == 0.5
    restored = ov.rollback_overlay(tmp_path)
    assert restored is not None
    assert restored["MEMO_RECALL_MIN_SIM"] == 0.5


def test_rollback_when_no_prev_returns_none(tmp_path: Path):
    assert ov.rollback_overlay(tmp_path) is None


def test_overlay_carries_bool_and_float_graph_config(tmp_path: Path):
    # The graph tuner writes an atomic bool + float configuration.
    ov.write_overlay(
        tmp_path,
        {"MEMO_GRAPH_SIGNAL_ENABLED": True, "MEMO_GRAPH_SIGNAL_ALPHA": 0.15},
        {"set_by": "dream-curated-graph"},
    )
    vals = ov.overlay_values({"MEMO_STATE_DIR": str(tmp_path)})
    assert vals["MEMO_GRAPH_SIGNAL_ENABLED"] == "1"
    assert vals["MEMO_GRAPH_SIGNAL_ALPHA"] == "0.15"
    env = {"MEMO_STATE_DIR": str(tmp_path)}
    assert flags.flag_bool("MEMO_GRAPH_SIGNAL_ENABLED", env=env) is True
    assert flags.flag_float("MEMO_GRAPH_SIGNAL_ALPHA", env=env) == 0.15


def test_bool_lever_rolls_back_to_bool(tmp_path: Path):
    ov.write_overlay(tmp_path, {"MEMO_GRAPH_SIGNAL_ENABLED": False}, {})
    ov.write_overlay(tmp_path, {"MEMO_GRAPH_SIGNAL_ENABLED": True}, {})
    restored = ov.rollback_overlay(tmp_path)
    assert restored is not None
    assert restored["MEMO_GRAPH_SIGNAL_ENABLED"] is False  # type preserved


def test_mixed_float_and_bool_coexist(tmp_path: Path):
    # A float knob and a bool lever in the same overlay must both surface.
    ov.write_overlay(
        tmp_path,
        {"MEMO_RECALL_MIN_SIM": 0.55, "MEMO_GRAPH_SIGNAL_ENABLED": True},
        {},
    )
    env = {"MEMO_STATE_DIR": str(tmp_path)}
    assert flags.flag_float("MEMO_RECALL_MIN_SIM", env=env) == 0.55
    assert flags.flag_bool("MEMO_GRAPH_SIGNAL_ENABLED", env=env) is True


def test_flag_precedence_env_over_overlay_over_default(tmp_path: Path):
    ov.write_overlay(tmp_path, {"MEMO_RECALL_MIN_SIM": 0.6}, {})
    env = {"MEMO_STATE_DIR": str(tmp_path)}
    # overlay supplies the value when env is unset
    assert flags.flag_float("MEMO_RECALL_MIN_SIM", env=env) == 0.6
    # explicit env var wins over the overlay
    env["MEMO_RECALL_MIN_SIM"] = "0.8"
    assert flags.flag_float("MEMO_RECALL_MIN_SIM", env=env) == 0.8
    # no overlay, no env → registry default (0.5)
    assert flags.flag_float("MEMO_RECALL_MIN_SIM", env={"MEMO_STATE_DIR": "/nonexistent"}) == 0.5


def test_params_version_base_when_no_overlay(tmp_path):
    from memo.tuned_overlay import params_version

    assert params_version(tmp_path) == "base"


def test_params_version_stable_and_order_independent(tmp_path):
    from memo.tuned_overlay import params_version, write_overlay

    write_overlay(
        tmp_path, {"MEMO_RECALL_MIN_SIM": 0.62, "MEMO_RECALL_MODE": "hybrid"}, {"set_by": "test"}
    )
    v1 = params_version(tmp_path)
    # same params, different insertion order → identical hash
    write_overlay(
        tmp_path, {"MEMO_RECALL_MODE": "hybrid", "MEMO_RECALL_MIN_SIM": 0.62}, {"set_by": "test"}
    )
    v2 = params_version(tmp_path)

    assert v1 == v2
    assert v1 != "base"
    assert len(v1) == 12
