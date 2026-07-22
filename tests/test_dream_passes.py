"""Tests for dream_passes handlers — the extracted phase implementations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from memo.cli_dream_passes import (
    _run_consolidate_dups,
    _run_contradict,
    _run_entities,
    _run_graph_projection,
    _run_roi_decay,
    _run_roi_reconcile,
    _run_stale,
    _run_synthesis,
)
from memo.memory import Memory


class TestRunMandateSyncPhase:
    def test_disabled_phase_leaves_receipt_and_progress_unchanged(
        self, tmp_cfg, mock_memory: Memory, monkeypatch
    ) -> None:
        from memo.cli_dream import _run_mandate_sync_phase

        monkeypatch.setenv("MEMO_DYNAMIC_MANDATE_SYNC_ENABLED", "0")
        receipt = {"errors": []}
        progress = MagicMock()

        _run_mandate_sync_phase(tmp_cfg, mock_memory, receipt, progress, "step")

        assert receipt == {"errors": []}
        progress.update.assert_not_called()

    def test_enabled_phase_records_result_error_and_progress(
        self, tmp_cfg, mock_memory: Memory, monkeypatch
    ) -> None:
        from memo.cli_dream import _run_mandate_sync_phase

        monkeypatch.setenv("MEMO_DYNAMIC_MANDATE_SYNC_ENABLED", "1")
        result = {"synced": ["repo-a", "repo-b"], "error": "repo-c refused update"}
        run_pass = MagicMock(return_value=result)
        monkeypatch.setattr("memo.constitution.run_mandate_sync_pass", run_pass)
        receipt = {"errors": []}
        progress = MagicMock()

        _run_mandate_sync_phase(tmp_cfg, mock_memory, receipt, progress, "step")

        run_pass.assert_called_once_with(tmp_cfg, mock_memory)
        assert receipt["mandate_sync"] is result
        assert receipt["errors"] == ["mandate_sync: repo-c refused update"]
        assert progress.update.call_count == 2
        assert progress.update.call_args_list[0].kwargs["description"].startswith("[mandate-sync]")
        assert "2 repo(s)" in progress.update.call_args_list[1].kwargs["description"]


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

    def test_run_contradict_marks_competing_within_margin(self, mock_memory, monkeypatch) -> None:
        """Belief mode + wide margin -> pair marked competing, neither side archived."""
        monkeypatch.setenv("MEMO_BELIEF_COMPETING", "1")
        monkeypatch.setenv("MEMO_SUPERSEDE_MARGIN", "0.5")  # wide -> force competing

        a = mock_memory.save(content="El dashboard corre en el puerto 8080", title="p-old")
        b = mock_memory.save(content="El dashboard corre en el puerto 8765", title="p-new")
        mock_memory.contradict_store.upsert_open(
            memory_id_a=a.id,
            memory_id_b=b.id,
            relationship="contradiction",
            confidence=0.95,
            rationale="ports differ",
        )

        result = _run_contradict(mock_memory, dry_run=False)

        assert result.get("competing"), "expected a competing pair"
        # neither side archived
        assert mock_memory.get(a.id) is not None
        assert mock_memory.get(b.id) is not None

    def test_run_contradict_nway_marks_all_competing(self, mock_memory, monkeypatch) -> None:
        """3+ mutually-contradicting memories (a connected component) all end competing."""
        monkeypatch.setenv("MEMO_BELIEF_COMPETING", "1")
        monkeypatch.setenv("MEMO_BELIEF_NWAY", "1")

        a = mock_memory.save(content="El release sale el lunes", title="r1")
        b = mock_memory.save(content="El release sale el martes", title="r2")
        c = mock_memory.save(content="El release sale el miércoles", title="r3")
        for x, y in [(a, b), (b, c), (c, a)]:
            mock_memory.contradict_store.upsert_open(
                memory_id_a=x.id,
                memory_id_b=y.id,
                relationship="contradiction",
                confidence=0.95,
                rationale="dates differ",
            )

        result = _run_contradict(mock_memory, dry_run=False)
        assert len(result.get("competing", [])) == 3
        assert mock_memory.get(a.id) and mock_memory.get(b.id) and mock_memory.get(c.id)


class TestDreamRunReceiptCarriesCompeting:
    """`dream run` must copy competing/flagged_for_review from _run_contradict
    into the top-level receipt (cli_dream.py merge seam), same as cli_maintain.py."""

    def test_dream_run_receipt_carries_competing_pair(self, mock_memory, monkeypatch) -> None:
        import json

        from click.testing import CliRunner

        from memo.cli_dream import dream_cmd

        monkeypatch.setenv("MEMO_BELIEF_COMPETING", "1")
        monkeypatch.setenv("MEMO_SUPERSEDE_MARGIN", "0.5")  # wide -> force competing

        a = mock_memory.save(content="El dashboard corre en el puerto 8080", title="p-old")
        b = mock_memory.save(content="El dashboard corre en el puerto 8765", title="p-new")
        mock_memory.contradict_store.upsert_open(
            memory_id_a=a.id,
            memory_id_b=b.id,
            relationship="contradiction",
            confidence=0.95,
            rationale="ports differ",
        )

        with patch("memo.cli_dream._get_memory", return_value=mock_memory):
            res = CliRunner().invoke(
                dream_cmd,
                [
                    "run",
                    "--json",
                    "--skip-orientation",
                    "--skip-signal-gather",
                    "--skip-entities",
                    "--skip-decay",
                    "--skip-prune-floor",
                    "--skip-evict",
                    "--skip-compress",
                    "--skip-prewarm",
                    "--skip-presynthesis",
                ],
            )
        assert res.exit_code == 0, res.output
        out = res.output
        receipt = json.loads(out[out.index("{") :])
        assert receipt.get("competing"), (
            "expected the competing pair to surface in the dream receipt"
        )
        assert "flagged_for_review" in receipt
        # neither side archived
        assert mock_memory.get(a.id) is not None
        assert mock_memory.get(b.id) is not None


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


class TestRunGraphProjection:
    def test_rebuilds_dirty_projection(self, mock_memory: Memory, monkeypatch) -> None:
        monkeypatch.setenv("MEMO_GRAPH_PROJECTION_ENABLED", "1")
        mock_memory.graph.mark_projection_dirty()

        result = _run_graph_projection(mock_memory)

        assert result["status"] == "rebuilt"
        assert mock_memory.graph.projection_dirty() is False

    def test_dry_run_never_mutates(self, mock_memory: Memory, monkeypatch) -> None:
        monkeypatch.setenv("MEMO_GRAPH_PROJECTION_ENABLED", "1")
        mock_memory.graph.mark_projection_dirty()

        result = _run_graph_projection(mock_memory, dry_run=True)

        assert result["status"] == "would_rebuild"
        assert mock_memory.graph.projection_dirty() is True

    def test_disabled_projection_is_a_noop(self, mock_memory: Memory, monkeypatch) -> None:
        monkeypatch.setenv("MEMO_GRAPH_PROJECTION_ENABLED", "0")

        assert _run_graph_projection(mock_memory)["status"] == "disabled"

    def test_dream_run_records_projection_error_and_advances_phase(
        self, tmp_cfg, mock_memory: Memory, monkeypatch
    ) -> None:
        import json

        from click.testing import CliRunner

        from memo.cli_dream import dream_cmd

        monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_cfg.data_dir))
        monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_cfg.state_dir))
        monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
        monkeypatch.setenv("MEMO_GRAPH_PROJECTION_ENABLED", "1")
        monkeypatch.setenv("MEMO_OUTCOME_RANKING_ENABLED", "0")
        monkeypatch.setenv("MEMO_DREAM_EVAL_ENABLED", "0")
        monkeypatch.setenv("MEMO_DYNAMIC_MANDATE_SYNC_ENABLED", "0")
        monkeypatch.setattr("memo.cli_dream._get_memory", lambda _cfg: mock_memory)
        run_projection = MagicMock(
            return_value={"status": "error", "error": "projection validation failed"}
        )
        monkeypatch.setattr("memo.cli_dream._run_graph_projection", run_projection)
        skips = [
            "--skip-entities",
            "--skip-decay",
            "--skip-maintain",
            "--skip-orientation",
            "--skip-signal-gather",
            "--skip-prune-floor",
            "--skip-evict",
            "--skip-compress",
            "--skip-prewarm",
            "--skip-presynthesis",
        ]

        result = CliRunner().invoke(dream_cmd, ["run", "--dry-run", "--json", *skips])

        assert result.exit_code == 0, result.output
        receipt = json.loads(result.output[result.output.index("{") :])
        assert receipt["graph_projection"] == {
            "status": "error",
            "error": "projection validation failed",
        }
        assert "graph_projection: projection validation failed" in receipt["errors"]
        run_projection.assert_called_once_with(mock_memory, dry_run=True)


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

        with mock_patch(
            "memo.cli_dream_passes.reconcile_roi", side_effect=RuntimeError("roi failed")
        ):
            result = _run_roi_reconcile(mock_memory, dry_run=True)
            assert "error" in result
            assert "roi failed" in result["error"]


class TestRunRoiDecay:
    """Tests for _run_roi_decay phase handler."""

    def test_roi_decay_returns_zero_on_dry_run(self, mock_memory: Memory) -> None:
        """Estimates count on dry-run but doesn't modify."""
        with patch.object(mock_memory.store, "_conn") as mock_conn:
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
        with patch.object(mock_memory.store, "decay_roi", side_effect=RuntimeError("decay failed")):
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
                    with patch.object(
                        mock_memory.temporal, "detect_stale_memories", return_value=[]
                    ):
                        with patch.object(mock_memory, "synthesize_cross_cluster", return_value=[]):
                            with patch.object(
                                mock_memory,
                                "extract_entities",
                                return_value={"entities_extracted": 0},
                            ):
                                from unittest.mock import patch as mock_patch

                                with mock_patch("memo.cli_dream_passes.reconcile_roi") as mock_rec:
                                    with mock_patch("memo.cli_dream_passes.dead_weight") as mock_dw:
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
                    with patch.object(
                        mock_memory.temporal, "detect_stale_memories", return_value=[]
                    ):
                        with patch.object(mock_memory, "synthesize_cross_cluster", return_value=[]):
                            with patch.object(
                                mock_memory,
                                "extract_entities",
                                return_value={"entities_extracted": 0},
                            ):
                                from unittest.mock import patch as mock_patch

                                with mock_patch("memo.cli_dream_passes.reconcile_roi") as mock_rec:
                                    with mock_patch("memo.cli_dream_passes.dead_weight") as mock_dw:
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
