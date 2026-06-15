"""Tests for the grounding ground-truth eval (memo eval grounding)."""
from __future__ import annotations

import json
from pathlib import Path

from memo import eval_grounding


def _labels_file(tmp_path: Path, labels: list[dict]) -> Path:
    p = tmp_path / "labels.json"
    p.write_text(
        json.dumps({"schema": eval_grounding.LABELS_SCHEMA, "labels": labels}),
        encoding="utf-8",
    )
    return p


def test_load_labels_rejects_wrong_schema(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"schema": "nope", "labels": []}), encoding="utf-8")
    import pytest

    with pytest.raises(ValueError):
        eval_grounding.load_labels(p)


def test_detector_used_decision():
    assert eval_grounding.detector_used({"used_score": 0.85})           # strong
    assert eval_grounding.detector_used({"used_score": 0.5, "specific_score": 0.1})  # paraphrase
    assert eval_grounding.detector_used({"used_score": 0.2, "downstream_action": "opened_file"})
    assert not eval_grounding.detector_used({"used_score": 0.72})       # topical only
    assert not eval_grounding.detector_used({"used_score": 0.5, "specific_score": 0.01})


def test_evaluate_precision_recall(tmp_path: Path):
    rows = [
        {"session_id": "s", "turn": 1, "recall_id": "aaaa1111", "used_score": 0.9},   # pred used
        {"session_id": "s", "turn": 2, "recall_id": "bbbb2222", "used_score": 0.72},  # pred not used
        {"session_id": "s", "turn": 3, "recall_id": "cccc3333", "used_score": 0.5},   # pred not used
    ]
    labels = eval_grounding.load_labels(_labels_file(tmp_path, [
        {"session_id": "s", "turn": 1, "recall_id": "aaaa1111", "used": True},   # tp
        {"session_id": "s", "turn": 2, "recall_id": "bbbb2222", "used": True},   # fn (real use, detector missed)
        {"session_id": "s", "turn": 3, "recall_id": "cccc3333", "used": False},  # tn
    ]))
    r = eval_grounding.evaluate(rows, labels)
    assert r["tp"] == 1 and r["fn"] == 1 and r["tn"] == 1 and r["fp"] == 0
    assert r["precision"] == 1.0
    assert r["recall"] == 0.5
    assert r["false_negatives"] == [("s", 2, "bbbb2222")]


def test_evaluate_counts_missing(tmp_path: Path):
    labels = eval_grounding.load_labels(_labels_file(tmp_path, [
        {"session_id": "s", "turn": 9, "recall_id": "zzzz9999", "used": True},
    ]))
    r = eval_grounding.evaluate([], labels)
    assert r["missing"] == 1 and r["scored"] == 0
