"""TDD tests for Task 9: Lever 3 — precision-gate for low-grounding score bands.

Tests:
  - _band_key: 0.05-wide bucket key
  - suppress_score: predicate
  - learn_precision_bands: flags a zero-grounding band
  - load_precision_bands: reads cached JSON
"""

from __future__ import annotations

import json
from pathlib import Path

from memo.token_meter import (
    _band_key,
    learn_precision_bands,
    load_precision_bands,
    suppress_score,
)

# ---------------------------------------------------------------------------
# _band_key — bucketing
# ---------------------------------------------------------------------------


def test_band_key_buckets_by_0_05():
    assert _band_key(0.63) == "0.60"
    assert _band_key(0.60) == "0.60"
    assert _band_key(0.64) == "0.60"
    assert _band_key(0.65) == "0.65"
    assert _band_key(0.50) == "0.50"
    assert _band_key(0.55) == "0.55"
    assert _band_key(0.99) == "0.95"


# ---------------------------------------------------------------------------
# suppress_score — predicate
# ---------------------------------------------------------------------------


def test_suppress_score_returns_false_for_empty_bands():
    assert suppress_score(0.63, {}) is False


def test_suppress_score_returns_true_when_band_suppressed():
    bands = {"0.60": {"total": 25, "grounded": 0, "suppress": True}}
    assert suppress_score(0.63, bands) is True


def test_suppress_score_returns_false_when_band_not_suppressed():
    bands = {"0.60": {"total": 25, "grounded": 5, "suppress": False}}
    assert suppress_score(0.63, bands) is False


def test_suppress_score_returns_false_for_unknown_band():
    bands = {"0.70": {"total": 25, "grounded": 0, "suppress": True}}
    assert suppress_score(0.63, bands) is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# learn_precision_bands — zero-grounding band detection
# ---------------------------------------------------------------------------


def test_learn_flags_zero_grounding_band(tmp_path):
    """A band with >=min_samples recalls and zero grounding gets suppress=True."""
    from memo.dashboard_logs import grounding_log_path, recall_log_path

    state = tmp_path / "state"
    state.mkdir()

    # 20 recalls in band 0.60 (score=0.62 → bucket "0.60"), none grounded
    recall_entries = [
        {
            "session_id": f"S{i}",
            "turn": 1,
            "hits": [{"id": f"id{i}", "score": 0.62, "title": "t"}],
        }
        for i in range(20)
    ]
    _write_jsonl(recall_log_path(state), recall_entries)
    _write_jsonl(grounding_log_path(state), [])

    bands = learn_precision_bands(state, min_samples=20)

    assert "0.60" in bands
    assert bands["0.60"]["total"] == 20
    assert bands["0.60"]["grounded"] == 0
    assert bands["0.60"]["suppress"] is True


def test_learn_does_not_suppress_grounded_band(tmp_path):
    """A band with any grounding is not suppressed even if large."""
    from memo.dashboard_logs import grounding_log_path, recall_log_path
    from memo.dashboard_metrics import GROUNDED_SCORE

    state = tmp_path / "state"
    state.mkdir()

    recall_entries = [
        {
            "session_id": f"S{i}",
            "turn": 1,
            "hits": [{"id": f"id{i}", "score": 0.72, "title": "t"}],
        }
        for i in range(25)
    ]
    _write_jsonl(recall_log_path(state), recall_entries)
    # One grounding event for session S0, turn 1
    _write_jsonl(
        grounding_log_path(state),
        [
            {
                "session_id": "S0",
                "turn": 1,
                "recall_id": "id0",
                "used_score": GROUNDED_SCORE + 0.1,
            }
        ],
    )

    bands = learn_precision_bands(state, min_samples=20)

    assert "0.70" in bands
    assert bands["0.70"]["suppress"] is False


def test_learn_below_min_samples_not_suppressed(tmp_path):
    """A band with fewer than min_samples is not suppressed."""
    from memo.dashboard_logs import grounding_log_path, recall_log_path

    state = tmp_path / "state"
    state.mkdir()

    # Only 5 recalls in band 0.55 (score=0.57 → "0.55"), below min_samples=20
    recall_entries = [
        {
            "session_id": f"S{i}",
            "turn": 1,
            "hits": [{"id": f"id{i}", "score": 0.57, "title": "t"}],
        }
        for i in range(5)
    ]
    _write_jsonl(recall_log_path(state), recall_entries)
    _write_jsonl(grounding_log_path(state), [])

    bands = learn_precision_bands(state, min_samples=20)

    assert "0.55" in bands
    assert bands["0.55"]["suppress"] is False


def test_learn_skips_entries_without_hits(tmp_path):
    """Bail-type entries (hits=[]) are skipped."""
    from memo.dashboard_logs import grounding_log_path, recall_log_path

    state = tmp_path / "state"
    state.mkdir()

    _write_jsonl(
        recall_log_path(state),
        [
            {"session_id": "S0", "turn": 1, "hits": [], "via": "bail"},
        ],
    )
    _write_jsonl(grounding_log_path(state), [])

    bands = learn_precision_bands(state, min_samples=1)
    # No band should be suppressed since no real hits were logged
    assert all(not b["suppress"] for b in bands.values())


def test_learn_caches_to_precision_bands_json(tmp_path):
    """learn_precision_bands writes precision_bands.json to state_dir."""
    from memo.dashboard_logs import grounding_log_path, recall_log_path

    state = tmp_path / "state"
    state.mkdir()
    _write_jsonl(recall_log_path(state), [])
    _write_jsonl(grounding_log_path(state), [])

    learn_precision_bands(state)
    assert (state / "precision_bands.json").is_file()


# ---------------------------------------------------------------------------
# load_precision_bands
# ---------------------------------------------------------------------------


def test_load_precision_bands_returns_empty_for_missing_file(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    assert load_precision_bands(state) == {}


def test_load_precision_bands_reads_cached(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    bands = {"0.60": {"total": 20, "grounded": 0, "suppress": True}}
    (state / "precision_bands.json").write_text(json.dumps(bands), encoding="utf-8")

    result = load_precision_bands(state)
    assert result == bands


def test_load_precision_bands_returns_empty_on_corrupt_json(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "precision_bands.json").write_text("not-json", encoding="utf-8")
    assert load_precision_bands(state) == {}
