"""Tests for dream_passes handlers — the extracted phase implementations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from memo.cli_dream_passes import (
    _run_consolidate_dups,
    _run_contradict,
    _run_entities,
    _run_roi_decay,
    _run_roi_reconcile,
    _run_stale,
    _run_synthesis,
)
from memo.memory import Memory


class TestRunContradict:
    """Tests for _run_contradict phase handler."""

    def test_contradict_returns_empty_when_no_pairs(self, mock_memory: Memory) -> None:
        """Empty result when no contradictions found."""
        result = _run_contradict(mock_memory, dry_run=True)
        assert result["superseded"] == []
        assert result["evolved"] == []
        assert result["confidence_penalized"] == 0

    def test_contradict_handles_error_gracefully(self, mock_memory: Memory) -> None:
        """Errors are captured in result dict, not raised."""
        with patch.object(
            mock_memory.contradict_scanner, "scan_corpus", side_effect=RuntimeError("scan failed")
        ):
            result = _run_contradict(mock_memory, dry_run=True)
            assert "error" in result
            assert "scan failed" in result["error"]

    def test_contradict_dry_run_does_not_modify(self, mock_memory: Memory) -> None:
        """Dry-run mode never archives or modifies data."""
        with patch.object(mock_memory.contradict_store, "list_open", return_value=[]):
            with patch.object(mock_memory.lifecycle, "archive_memory") as mock_archive:
                _run_contradict(mock_memory, dry_run=True)
                mock_archive.assert_not_called()


class TestRunConsolidateDups:
    """Tests for _run_consolidate_dups phase handler."""

    def test_consolidate_returns_empty_when_no_dups(self, mock_memory: Memory) -> None:
        """Empty result when no duplicates found."""
        with patch.object(
            mock_memory.consolidator,
            "consolidate_all",
            return_value={"results": []},
        ):
            result = _run_consolidate_dups(mock_memory, dry_run=True)
            assert result["merged"] == []

    def test_consolidate_returns_merged_records(self, mock_memory: Memory) -> None:
        """Returns list of merged cluster records."""
        mock_results = {
            "results": [
                {"merged_id": "aaa", "archived_ids": ["bbb", "ccc"]},
                {"merged_id": "ddd", "archived_ids": ["eee"]},
            ]
        }
        with patch.object(
            mock_memory.consolidator,
            "consolidate_all",
            return_value=mock_results,
        ):
            result = _run_consolidate_dups(mock_memory, dry_run=True)
            assert len(result["merged"]) == 2
            assert result["merged"][0]["merged_id"] == "aaa"
            assert result["merged"][0]["archived_ids"] == ["bbb", "ccc"]

    def test_consolidate_handles_error_gracefully(self, mock_memory: Memory) -> None:
        """Errors are captured in result dict, not raised."""
        with patch.object(
            mock_memory.consolidator,
            "consolidate_all",
            side_effect=RuntimeError("consolidate failed"),
        ):
            result = _run_consolidate_dups(mock_memory, dry_run=True)
            assert "error" in result
            assert "consolidate failed" in result["error"]


class TestRunStale:
    """Tests for _run_stale phase handler."""

    def test_stale_returns_empty_when_no_stale(self, mock_memory: Memory) -> None:
        """Empty result when no stale memories found."""
        with patch.object(mock_memory.temporal, "detect_stale_memories", return_value=[]):
            result = _run_stale(mock_memory, dry_run=True)
            assert result["archived"] == []

    def test_stale_returns_archived_records(self, mock_memory: Memory) -> None:
        """Returns list of archived stale memory records."""
        mock_stale = [
            {"id": "aaa", "days_since_update": 400},
            {"id": "bbb", "days_since_update": 600},
        ]
        with patch.object(mock_memory.temporal, "detect_stale_memories", return_value=mock_stale):
            result = _run_stale(mock_memory, dry_run=True)
            assert len(result["archived"]) == 2
            assert result["archived"][0]["id"] == "aaa"
            assert result["archived"][0]["days"] == 400

    def test_stale_skips_none_ids(self, mock_memory: Memory) -> None:
        """Skips records with missing ids."""
        mock_stale = [
            {"id": "aaa", "days_since_update": 400},
            {"id": None, "days_since_update": 500},  # Should be skipped
        ]
        with patch.object(mock_memory.temporal, "detect_stale_memories", return_value=mock_stale):
            result = _run_stale(mock_memory, dry_run=True)
            assert len(result["archived"]) == 1
            assert result["archived"][0]["id"] == "aaa"

    def test_stale_dry_run_does_not_archive(self, mock_memory: Memory) -> None:
        """Dry-run mode never archives data."""
        mock_stale = [{"id": "aaa", "days_since_update": 400}]
        with patch.object(mock_memory.temporal, "detect_stale_memories", return_value=mock_stale):
            with patch.object(mock_memory.lifecycle, "archive_memory") as mock_archive:
                _run_stale(mock_memory, dry_run=True)
                mock_archive.assert_not_called()


class TestRunSynthesis:
    """Tests for _run_synthesis phase handler."""

    def test_synthesis_returns_empty_when_no_results(self, mock_memory: Memory) -> None:
        """Empty result when no syntheses generated."""
        with patch.object(mock_memory, "synthesize_cross_cluster", return_value=[]):
            result = _run_synthesis(mock_memory, dry_run=True)
            assert result["synthesized"] == []

    def test_synthesis_returns_synthesis_records(self, mock_memory: Memory) -> None:
        """Returns list of synthesis records."""
        mock_synths = [
            {"title": "Pattern A", "confidence": 0.85, "saved": True},
            {"title": "Pattern B", "confidence": 0.72, "saved": False},
        ]
        with patch.object(mock_memory, "synthesize_cross_cluster", return_value=mock_synths):
            result = _run_synthesis(mock_memory, dry_run=True)
            assert len(result["synthesized"]) == 2
            assert result["synthesized"][0]["title"] == "Pattern A"
            assert result["synthesized"][0]["confidence"] == 0.85
            assert result["synthesized"][0]["saved"] is True

    def test_synthesis_handles_error_gracefully(self, mock_memory: Memory) -> None:
        """Errors are captured in result dict, not raised."""
        with patch.object(
            mock_memory,
            "synthesize_cross_cluster",
            side_effect=RuntimeError("synthesis failed"),
        ):
            result = _run_synthesis(mock_memory, dry_run=True)
            assert "error" in result
            assert "synthesis failed" in result["error"]


class TestRunEntities:
    """Tests for _run_entities phase handler."""

    def test_entities_returns_zero_when_none_extracted(self, mock_memory: Memory) -> None:
        """Zero result when no entities extracted."""
        with patch.object(mock_memory, "extract_entities", return_value={"entities_extracted": 0}):
            result = _run_entities(mock_memory, dry_run=True)
            assert result["extracted"] == 0

    def test_entities_returns_count(self, mock_memory: Memory) -> None:
        """Returns count of extracted entities."""
        with patch.object(mock_memory, "extract_entities", return_value={"entities_extracted": 42}):
            result = _run_entities(mock_memory, dry_run=True)
            assert result["extracted"] == 42

    def test_entities_handles_error_gracefully(self, mock_memory: Memory) -> None:
        """Errors are captured in result dict, not raised."""
        with patch.object(
            mock_memory,
            "extract_entities",
            side_effect=RuntimeError("extraction failed"),
        ):
            result = _run_entities(mock_memory, dry_run=True)
            assert "error" in result
            assert "extraction failed" in result["error"]


class TestRunRoiReconcile:
    """Tests for _run_roi_reconcile phase handler."""

    def test_roi_reconcile_returns_reconciled_count(self, mock_memory: Memory, monkeypatch) -> None:
        """Returns count of reconciled ROI records."""
        from unittest.mock import patch as mock_patch

        monkeypatch.setenv("MEMO_OUTCOME_SOURCE_FEEDBACK", "0")
        monkeypatch.setenv("MEMO_OUTCOME_DEAD_MIN_SURFACED", "0")

        with mock_patch("memo.cli_dream_passes.reconcile_roi") as mock_rec:
            with mock_patch("memo.cli_dream_passes.dead_weight") as mock_dw:
                mock_rec.return_value = {"updated": 15}
                mock_dw.return_value = []

                result = _run_roi_reconcile(mock_memory, dry_run=True)
                assert result["reconciled"] == 15
                assert result["dead_archived"] == []

    def test_roi_reconcile_archives_dead_weight(self, mock_memory: Memory, monkeypatch) -> None:
        """Archives dead-weight memories (surfaced but not grounded)."""
        from unittest.mock import patch as mock_patch

        monkeypatch.setenv("MEMO_OUTCOME_SOURCE_FEEDBACK", "0")
        monkeypatch.setenv("MEMO_OUTCOME_DEAD_MIN_SURFACED", "0")

        with mock_patch("memo.cli_dream_passes.reconcile_roi") as mock_rec:
            with mock_patch("memo.cli_dream_passes.dead_weight") as mock_dw:
                with mock_patch.object(mock_memory, "forget", return_value=True):
                    mock_rec.return_value = {"updated": 5}
                    mock_dw.return_value = [
                        {"id": "aaa", "surfaced": 10},
                        {"id": "bbb", "surfaced": 5},
                    ]

                    result = _run_roi_reconcile(mock_memory, dry_run=False)
                    assert len(result["dead_archived"]) == 2
                    assert "aaa" in result["dead_archived"]
                    assert "bbb" in result["dead_archived"]

    def test_roi_reconcile_dry_run_does_not_modify(self, mock_memory: Memory, monkeypatch) -> None:
        """Dry-run mode never archives, but reports what would be archived."""
        from unittest.mock import patch as mock_patch

        monkeypatch.setenv("MEMO_OUTCOME_SOURCE_FEEDBACK", "0")
        monkeypatch.setenv("MEMO_OUTCOME_DEAD_MIN_SURFACED", "0")

        with mock_patch("memo.cli_dream_passes.reconcile_roi") as mock_rec:
            with mock_patch("memo.cli_dream_passes.dead_weight") as mock_dw:
                with mock_patch.object(mock_memory, "forget") as mock_forget:
                    mock_rec.return_value = {"updated": 5}
                    mock_dw.return_value = [{"id": "aaa", "surfaced": 10}]

                    result = _run_roi_reconcile(mock_memory, dry_run=True)
                    mock_forget.assert_not_called()
                    assert "aaa" in result["dead_archived"]  # Still reports

    def test_roi_reconcile_handles_error_gracefully(self, mock_memory: Memory, monkeypatch) -> None:
        """Errors are captured in result dict, not raised."""
        from unittest.mock import patch as mock_patch

        monkeypatch.setenv("MEMO_OUTCOME_SOURCE_FEEDBACK", "0")
        monkeypatch.setenv("MEMO_OUTCOME_DEAD_MIN_SURFACED", "0")

        with mock_patch("memo.cli_dream_passes.reconcile_roi", side_effect=RuntimeError("roi failed")):
            result = _run_roi_reconcile(mock_memory, dry_run=True)
            assert "error" in result
            assert "roi failed" in result["error"]


class TestRunRoiDecay:
    """Tests for _run_roi_decay phase handler."""

    def test_roi_decay_returns_zero_on_dry_run(self, mock_memory: Memory) -> None:
        """Estimates count on dry-run but doesn't modify."""
        with patch.object(
            mock_memory.store, "_conn"
        ) as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = (10,)
            mock_conn.execute.return_value = mock_cursor

            result = _run_roi_decay(mock_memory, dry_run=True)
            assert "decayed" in result
            # In dry-run, we get estimated count
            assert result["decayed"] >= 0

    def test_roi_decay_applies_decay_wet_run(self, mock_memory: Memory) -> None:
        """Applies decay when not in dry-run mode."""
        with patch.object(mock_memory.store, "decay_roi", return_value=25):
            result = _run_roi_decay(mock_memory, dry_run=False)
            assert result["decayed"] == 25

    def test_roi_decay_handles_error_gracefully(self, mock_memory: Memory) -> None:
        """Errors are captured in result dict, not raised."""
        with patch.object(
            mock_memory.store, "decay_roi", side_effect=RuntimeError("decay failed")
        ):
            result = _run_roi_decay(mock_memory, dry_run=False)
            assert "error" in result
            assert "decay failed" in result["error"]


class TestPhaseIntegration:
    """Integration tests for multiple phase handlers."""

    def test_all_phases_handle_dry_run(self, mock_memory: Memory, monkeypatch) -> None:
        """All phases respect dry_run flag."""
        monkeypatch.setenv("MEMO_OUTCOME_SOURCE_FEEDBACK", "0")
        monkeypatch.setenv("MEMO_OUTCOME_DEAD_MIN_SURFACED", "0")

        # Mock all the stores/methods to return empty results
        with patch.object(mock_memory.contradict_scanner, "scan_corpus"):
            with patch.object(mock_memory.contradict_store, "list_open", return_value=[]):
                with patch.object(
                    mock_memory.consolidator, "consolidate_all", return_value={"results": []}
                ):
                    with patch.object(mock_memory.temporal, "detect_stale_memories", return_value=[]):
                        with patch.object(mock_memory, "synthesize_cross_cluster", return_value=[]):
                            with patch.object(
                                mock_memory, "extract_entities", return_value={"entities_extracted": 0}
                            ):
                                from unittest.mock import patch as mock_patch

                                with mock_patch("memo.cli_dream_passes.reconcile_roi") as mock_rec:
                                    with mock_patch(
                                        "memo.cli_dream_passes.dead_weight"
                                    ) as mock_dw:
                                        mock_rec.return_value = {"updated": 0}
                                        mock_dw.return_value = []

                                        # All should complete without error in dry-run
                                        r1 = _run_contradict(mock_memory, dry_run=True)
                                        r2 = _run_consolidate_dups(mock_memory, dry_run=True)
                                        r3 = _run_stale(mock_memory, dry_run=True)
                                        r4 = _run_synthesis(mock_memory, dry_run=True)
                                        r5 = _run_entities(mock_memory, dry_run=True)
                                        r6 = _run_roi_reconcile(mock_memory, dry_run=True)
                                        r7 = _run_roi_decay(mock_memory, dry_run=True)

                                        # None should have errors in this clean path
                                        assert "error" not in r1
                                        assert "error" not in r2
                                        assert "error" not in r3
                                        assert "error" not in r4
                                        assert "error" not in r5
                                        assert "error" not in r6
                                        assert "error" not in r7

    def test_all_phases_return_proper_structure(self, mock_memory: Memory, monkeypatch) -> None:
        """All phases return dicts with consistent keys."""
        monkeypatch.setenv("MEMO_OUTCOME_SOURCE_FEEDBACK", "0")
        monkeypatch.setenv("MEMO_OUTCOME_DEAD_MIN_SURFACED", "0")

        with patch.object(mock_memory.contradict_scanner, "scan_corpus"):
            with patch.object(mock_memory.contradict_store, "list_open", return_value=[]):
                with patch.object(
                    mock_memory.consolidator, "consolidate_all", return_value={"results": []}
                ):
                    with patch.object(mock_memory.temporal, "detect_stale_memories", return_value=[]):
                        with patch.object(mock_memory, "synthesize_cross_cluster", return_value=[]):
                            with patch.object(
                                mock_memory, "extract_entities", return_value={"entities_extracted": 0}
                            ):
                                from unittest.mock import patch as mock_patch

                                with mock_patch("memo.cli_dream_passes.reconcile_roi") as mock_rec:
                                    with mock_patch(
                                        "memo.cli_dream_passes.dead_weight"
                                    ) as mock_dw:
                                        mock_rec.return_value = {"updated": 0}
                                        mock_dw.return_value = []

                                        r1 = _run_contradict(mock_memory)
                                        assert isinstance(r1, dict)
                                        assert "superseded" in r1
                                        assert "evolved" in r1

                                        r2 = _run_consolidate_dups(mock_memory)
                                        assert isinstance(r2, dict)
                                        assert "merged" in r2

                                        r3 = _run_stale(mock_memory)
                                        assert isinstance(r3, dict)
                                        assert "archived" in r3

                                        r4 = _run_synthesis(mock_memory)
                                        assert isinstance(r4, dict)
                                        assert "synthesized" in r4

                                        r5 = _run_entities(mock_memory)
                                        assert isinstance(r5, dict)
                                        assert "extracted" in r5

                                        r6 = _run_roi_reconcile(mock_memory)
                                        assert isinstance(r6, dict)
                                        assert "reconciled" in r6
                                        assert "dead_archived" in r6

                                        r7 = _run_roi_decay(mock_memory)
                                        assert isinstance(r7, dict)
                                        assert "decayed" in r7
