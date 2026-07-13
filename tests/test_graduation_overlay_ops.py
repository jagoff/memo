from pathlib import Path

from memo.graduation import overlay_ops
from memo.tuned_overlay import read_overlay, write_overlay


def test_flip_on_sets_bool_and_is_detected(tmp_path: Path):
    overlay_ops.flip_on(tmp_path, "MEMO_GRAPH_SIGNAL_ENABLED", evidence={"streak": 5})
    assert overlay_ops.is_flipped_on(tmp_path, "MEMO_GRAPH_SIGNAL_ENABLED") is True
    doc = read_overlay(tmp_path)
    assert doc["MEMO_GRAPH_SIGNAL_ENABLED"] is True
    assert doc["_meta"]["evidence"] == {"streak": 5}


def test_flip_on_preserves_other_tuned_knobs(tmp_path: Path):
    write_overlay(tmp_path, {"MEMO_RECALL_MIN_SIM": 0.62}, {"set_by": "tuner"})
    overlay_ops.flip_on(tmp_path, "MEMO_GRAPH_SIGNAL_ENABLED", evidence={})
    doc = read_overlay(tmp_path)
    assert doc["MEMO_RECALL_MIN_SIM"] == 0.62      # untouched
    assert doc["MEMO_GRAPH_SIGNAL_ENABLED"] is True


def test_revert_drops_only_that_flag(tmp_path: Path):
    write_overlay(tmp_path, {"MEMO_RECALL_MIN_SIM": 0.62}, {"set_by": "tuner"})
    overlay_ops.flip_on(tmp_path, "MEMO_GRAPH_SIGNAL_ENABLED", evidence={})
    overlay_ops.revert(tmp_path, "MEMO_GRAPH_SIGNAL_ENABLED")
    doc = read_overlay(tmp_path)
    assert "MEMO_GRAPH_SIGNAL_ENABLED" not in doc
    assert doc["MEMO_RECALL_MIN_SIM"] == 0.62
    assert overlay_ops.is_flipped_on(tmp_path, "MEMO_GRAPH_SIGNAL_ENABLED") is False


def test_is_flipped_on_false_when_absent(tmp_path: Path):
    assert overlay_ops.is_flipped_on(tmp_path, "MEMO_GRAPH_SIGNAL_ENABLED") is False
