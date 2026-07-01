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


def test_overlay_values_missing_state_dir_is_empty():
    assert ov.overlay_values({}) == {}


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


def test_overlay_carries_bool_and_string_levers(tmp_path: Path):
    # The retrieval tuner flips booleans + the recall mode; they must survive
    # the write/read/resolve round-trip (bool -> "1"/"0", str verbatim).
    ov.write_overlay(
        tmp_path,
        {"MEMO_GRAPH_RETRIEVAL_ENABLED": True, "MEMO_RECALL_MODE": "hybrid"},
        {"set_by": "dream-retrieval"},
    )
    vals = ov.overlay_values({"MEMO_STATE_DIR": str(tmp_path)})
    assert vals["MEMO_GRAPH_RETRIEVAL_ENABLED"] == "1"
    assert vals["MEMO_RECALL_MODE"] == "hybrid"
    # and they resolve through flag() per kind
    env = {"MEMO_STATE_DIR": str(tmp_path)}
    assert flags.flag_bool("MEMO_GRAPH_RETRIEVAL_ENABLED", env=env) is True
    assert flags.flag_str("MEMO_RECALL_MODE", env=env) == "hybrid"


def test_bool_lever_rolls_back_to_bool(tmp_path: Path):
    ov.write_overlay(tmp_path, {"MEMO_GRAPH_RETRIEVAL_ENABLED": False}, {})
    ov.write_overlay(tmp_path, {"MEMO_GRAPH_RETRIEVAL_ENABLED": True}, {})
    restored = ov.rollback_overlay(tmp_path)
    assert restored is not None
    assert restored["MEMO_GRAPH_RETRIEVAL_ENABLED"] is False  # type preserved


def test_mixed_float_and_bool_coexist(tmp_path: Path):
    # A float knob and a bool lever in the same overlay must both surface.
    ov.write_overlay(
        tmp_path,
        {"MEMO_RECALL_MIN_SIM": 0.55, "MEMO_GRAPH_EXPANSION_ENABLED": True},
        {},
    )
    env = {"MEMO_STATE_DIR": str(tmp_path)}
    assert flags.flag_float("MEMO_RECALL_MIN_SIM", env=env) == 0.55
    assert flags.flag_bool("MEMO_GRAPH_EXPANSION_ENABLED", env=env) is True


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

    write_overlay(tmp_path, {"MEMO_RECALL_MIN_SIM": 0.62, "MEMO_RECALL_MODE": "hybrid"}, {"set_by": "test"})
    v1 = params_version(tmp_path)
    # same params, different insertion order → identical hash
    write_overlay(tmp_path, {"MEMO_RECALL_MODE": "hybrid", "MEMO_RECALL_MIN_SIM": 0.62}, {"set_by": "test"})
    v2 = params_version(tmp_path)

    assert v1 == v2
    assert v1 != "base"
    assert len(v1) == 12
