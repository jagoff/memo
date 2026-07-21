"""Distance decay in rerank tests.

Tests for the MEMO_GRAPH_DISTANCE_DECAY feature that applies
inverse-distance weighting to memories during reranking.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from memo.config import Config
from memo.memory import Memory


def test_rerank_applies_distance_decay(tmp_path: Path):
    """When MEMO_GRAPH_DISTANCE_DECAY=True, distant memories score lower."""
    cfg = Config(data_dir=tmp_path / "data", state_dir=tmp_path / "state")
    memory = Memory(cfg)

    # Mock rerank input: both hit with base score 0.9
    hits = [
        {"id": "dec1", "score": 0.9, "body": "close (distance 1)"},
        {"id": "syn1", "score": 0.9, "body": "far (distance 5)"},
    ]

    try:
        # Patch graph.distance_to_nearest_fact
        with patch.object(memory.graph, "distance_to_nearest_fact") as mock_dist:

            def distance_side_effect(mid):
                return 1 if mid == "dec1" else 5

            mock_dist.side_effect = distance_side_effect

            # Patch flags
            with (
                patch("memo.memory.rerank_ops.flag_bool", return_value=True),
                patch("memo.memory.rerank_ops.flag_float", return_value=0.15),
            ):
                result = memory._rerank_logic(
                    hits=hits,
                    query="test",
                    rerank_candidates=2,
                )

        # Close memory should score higher than far memory
        close_final_score = next(h["score"] for h in result if h["id"] == "dec1")
        far_final_score = next(h["score"] for h in result if h["id"] == "syn1")

        assert close_final_score > far_final_score, (
            f"Close {close_final_score} should > Far {far_final_score}"
        )
    finally:
        memory.close()


def test_distance_decay_disabled_no_change(tmp_path: Path):
    """When MEMO_GRAPH_DISTANCE_DECAY=False, scores unchanged."""
    cfg = Config(data_dir=tmp_path / "data", state_dir=tmp_path / "state")
    memory = Memory(cfg)

    hits = [
        {"id": "syn1", "score": 0.9},
        {"id": "dec1", "score": 0.9},
    ]

    try:
        with patch("memo.memory.rerank_ops.flag_bool", return_value=False):
            result = memory._rerank_logic(
                hits=hits,
                query="test",
                rerank_candidates=2,
            )

        # Scores should be identical to input
        assert result[0]["score"] == hits[0]["score"]
        assert result[1]["score"] == hits[1]["score"]
    finally:
        memory.close()
