"""Tests for Dream Mode initiatives: health scores, graph expansion,
adaptive recall, synthesis default, smarter capture, memo_health_report."""

from __future__ import annotations

import pytest

from memo.flags import flag_bool
from memo.memory.write_ops import _infer_type_from_content

# ---------------------------------------------------------------------------
# Initiative 6 — Smarter Capture: regex type inference
# ---------------------------------------------------------------------------


class TestInferTypeFromContent:
    def test_decision_english(self):
        assert _infer_type_from_content("We decided to use sqlite-vec") == "decision"

    def test_decision_spanish(self):
        assert _infer_type_from_content("Decidimos usar sqlite-vec para el store") == "decision"

    def test_preference_english(self):
        assert _infer_type_from_content("I prefer pytest over unittest") == "preference"

    def test_preference_spanish(self):
        assert _infer_type_from_content("Prefiero usar uv en vez de pip") == "preference"

    def test_bug_english(self):
        assert _infer_type_from_content("Bug: the root cause was a missing null check") == "bug"

    def test_bug_issue(self):
        assert _infer_type_from_content("Issue: found that the connection pool leaks") == "bug"

    def test_fact_english(self):
        assert (
            _infer_type_from_content("Turns out the model needs the instruction prefix") == "fact"
        )

    def test_fact_spanish(self):
        assert (
            _infer_type_from_content("Resulta que el modelo necesita el prefijo de instrucción")
            == "fact"
        )

    def test_no_match_returns_none(self):
        assert _infer_type_from_content("The quick brown fox jumps") is None

    def test_short_content_returns_none(self):
        assert _infer_type_from_content("ok") is None

    def test_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("MEMO_CAPTURE_PATTERN_TYPES", "0")
        assert _infer_type_from_content("We decided to use sqlite-vec") is None

    def test_only_first_600_chars_checked(self):
        # Pattern after 600 chars should not be detected.
        padding = "x " * 300  # 600 chars
        content = padding + "We decided to use sqlite-vec"
        assert _infer_type_from_content(content) is None


# ---------------------------------------------------------------------------
# Initiative 4 — Synthesis default on
# ---------------------------------------------------------------------------


def test_synthesis_enabled_by_default():
    assert flag_bool("MEMO_SYNTHESIS_ENABLED") is True


def test_synthesis_min_cluster_default():
    from memo.flags import flag_int

    assert flag_int("MEMO_SYNTHESIS_MIN_CLUSTER") == 5


def test_synthesis_max_clusters_default():
    from memo.flags import flag_int

    assert flag_int("MEMO_SYNTHESIS_MAX_CLUSTERS") == 10


# ---------------------------------------------------------------------------
# Initiative 2 — Adaptive recall context flag
# ---------------------------------------------------------------------------


def test_adaptive_context_enabled_by_default():
    assert flag_bool("MEMO_RECALL_ADAPTIVE_CONTEXT") is True


# ---------------------------------------------------------------------------
# Initiative 7 — Outcome loop on by default
# ---------------------------------------------------------------------------


def test_outcome_ranking_enabled_by_default():
    assert flag_bool("MEMO_OUTCOME_RANKING_ENABLED") is True


# ---------------------------------------------------------------------------
# Initiative 5 — Memory Health Scores (store layer)
# ---------------------------------------------------------------------------


class TestMemoryHealthStore:
    def test_get_health_batch_empty(self, mock_memory):
        result = mock_memory.store.get_health_batch([])
        assert result == {}

    def test_get_health_batch_missing_ids(self, mock_memory):
        result = mock_memory.store.get_health_batch(["nonexistent"])
        assert result == {}

    def test_boost_roi_creates_row(self, mock_memory):
        mock_memory.store.boost_roi_batch(["abc123"])
        health = mock_memory.store.get_health_batch(["abc123"])
        assert "abc123" in health
        assert health["abc123"]["roi_score"] > 1.0
        assert health["abc123"]["confidence"] == pytest.approx(1.0)

    def test_boost_roi_increments(self, mock_memory):
        mock_memory.store.boost_roi_batch(["abc"])
        mock_memory.store.boost_roi_batch(["abc"])
        health = mock_memory.store.get_health_batch(["abc"])
        # Two boosts of 0.05 each → 1.0 + 0.05 + 0.05 = 1.1
        assert health["abc"]["roi_score"] == pytest.approx(1.1, abs=1e-4)

    def test_boost_roi_capped(self, mock_memory):
        # Boost 100 times — should cap at 1.5
        for _ in range(100):
            mock_memory.store.boost_roi_batch(["abc"])
        health = mock_memory.store.get_health_batch(["abc"])
        assert health["abc"]["roi_score"] <= 1.5 + 1e-6

    def test_penalize_confidence(self, mock_memory):
        mock_memory.store.penalize_confidence_batch(["abc"])
        health = mock_memory.store.get_health_batch(["abc"])
        assert health["abc"]["confidence"] < 1.0
        assert health["abc"]["confidence"] == pytest.approx(1.0 - 0.15, abs=1e-4)

    def test_penalize_confidence_floored(self, mock_memory):
        # Penalize 100 times — should floor at 0.1
        for _ in range(100):
            mock_memory.store.penalize_confidence_batch(["abc"])
        health = mock_memory.store.get_health_batch(["abc"])
        assert health["abc"]["confidence"] >= 0.1 - 1e-6

    def test_decay_roi_updates_rows(self, mock_memory):
        # Insert rows with a past timestamp by directly setting updated_at.
        mock_memory.store.boost_roi_batch(["abc", "def"])
        # Backdate the rows so they match the older_than_days filter.
        with mock_memory.store._tx() as cx:
            cx.execute("UPDATE memory_health SET updated_at = datetime('now', '-60 days')")
        n = mock_memory.store.decay_roi(factor=0.5, older_than_days=30)
        assert n == 2
        health = mock_memory.store.get_health_batch(["abc"])
        # roi_score was 1.05, after 0.5× decay ≈ 0.525
        assert health["abc"]["roi_score"] == pytest.approx(1.05 * 0.5, abs=1e-3)

    def test_get_batch_empty(self, mock_memory):
        assert mock_memory.store.get_batch([]) == []

    def test_get_batch_returns_saved(self, mock_memory):
        rec = mock_memory.save(content="hello world test", title="Batch Test")
        rows = mock_memory.store.get_batch([rec.id])
        assert len(rows) == 1
        assert rows[0]["id"] == rec.id
        assert rows[0]["title"] == "Batch Test"


# ---------------------------------------------------------------------------
# Initiative 5 — Health scores applied in search
# ---------------------------------------------------------------------------


class TestHealthScoresInSearch:
    def test_neutral_health_no_effect(self, mock_memory):
        """Records without health entries → score unchanged."""
        rec = mock_memory.save(content="unique search content xyzzy", title="H1")
        hits = mock_memory.search("unique search content xyzzy", limit=5)
        # Should find the record
        assert any(h.id == rec.id for h in hits)

    def test_high_roi_record_ranks_at_or_above_neutral(self, mock_memory, monkeypatch):
        """A record with boosted roi_score should not rank lower than a neutral one."""
        monkeypatch.delenv("MEMO_HEALTH_SCORES_DISABLED", raising=False)
        rec_a = mock_memory.save(content="knowledge alpha testing", title="Alpha")
        rec_b = mock_memory.save(content="knowledge beta testing", title="Beta")
        # Boost Alpha's ROI significantly
        for _ in range(10):
            mock_memory.store.boost_roi_batch([rec_a.id])
        hits = mock_memory.search("knowledge testing", limit=5)
        ids = [h.id for h in hits]
        if rec_a.id in ids and rec_b.id in ids:
            assert ids.index(rec_a.id) <= ids.index(rec_b.id)

    def test_health_scores_disabled_flag(self, mock_memory, monkeypatch):
        """MEMO_HEALTH_SCORES_DISABLED=1 skips health multiplication."""
        monkeypatch.setenv("MEMO_HEALTH_SCORES_DISABLED", "1")
        rec = mock_memory.save(content="disabled health test content", title="DH")
        hits = mock_memory.search("disabled health test", limit=5)
        assert any(h.id == rec.id for h in hits)


# ---------------------------------------------------------------------------
# Initiative 3 — Dream Mode CLI smoke test
# ---------------------------------------------------------------------------


class TestDreamCli:
    def test_dream_run_dry_run(self, tmp_cfg, monkeypatch):
        """memo dream run --dry-run should exit 0 without modifying state."""
        import json as _json

        from click.testing import CliRunner

        from memo.cli_dream import dream_cmd

        runner = CliRunner()
        result = runner.invoke(
            dream_cmd,
            ["run", "--dry-run", "--json"],
            env={
                "MEMO_NONINTERACTIVE": "1",
                "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
                "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
            },
        )
        assert result.exit_code == 0
        # Rich console output precedes the JSON block — find the opening brace.
        out = result.output
        json_start = out.index("{")
        receipt = _json.loads(out[json_start:])
        assert receipt["dry_run"] is True
        assert "errors" in receipt

    def test_dream_run_dry_run_does_not_persist_contradiction_scan(self, tmp_cfg, monkeypatch):
        from unittest.mock import MagicMock

        from click.testing import CliRunner

        from memo.cli_dream import dream_cmd
        from memo.contradict import ScanResult

        mem = MagicMock()
        mem.lifecycle.enforce_forget_ttl.return_value = []
        mem.contradict_scanner.scan_corpus.return_value = ScanResult(
            scanned_memorias=0,
            pairs_examined=0,
            pairs_inserted=0,
            pairs_refreshed=0,
            pairs_skipped_resolved=0,
            contradictions_found=0,
            evolutions_found=0,
        )
        mem.contradict_store.list_open.return_value = []
        mem.consolidator.consolidate_all.return_value = {"results": []}
        mem.temporal.detect_stale_memorias.return_value = []
        mem.synthesize_cross_cluster.return_value = []
        monkeypatch.setattr("memo.cli_dream._get_memory", lambda _cfg: mem)

        result = CliRunner().invoke(
            dream_cmd,
            [
                "run",
                "--dry-run",
                "--json",
                "--skip-orientation",
                "--skip-entities",
                "--skip-decay",
                "--skip-prune-floor",
                "--skip-evict",
                "--skip-compress",
                "--skip-prewarm",
                "--skip-presynthesis",
            ],
            env={
                "MEMO_NONINTERACTIVE": "1",
                "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
                "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
            },
        )

        assert result.exit_code == 0
        assert mem.contradict_scanner.scan_corpus.call_args.kwargs["persist"] is False

    def test_dream_status_never_run(self, tmp_cfg, monkeypatch):
        from click.testing import CliRunner

        from memo.cli_dream import dream_cmd

        monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_cfg.state_dir))
        runner = CliRunner()
        result = runner.invoke(
            dream_cmd,
            ["status"],
            env={
                "MEMO_NONINTERACTIVE": "1",
                "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
                "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
            },
        )
        assert result.exit_code == 0
        assert "never" in result.output


# ---------------------------------------------------------------------------
# Initiative 5 — memo_health_report MCP tool
# ---------------------------------------------------------------------------


class TestMemoryHealthReport:
    def test_returns_empty_on_fresh_corpus(self, mock_memory):
        from unittest.mock import MagicMock

        from memo.server_health import register

        server = MagicMock()
        captured = {}

        def mock_tool():
            def decorator(fn):
                captured["fn"] = fn
                return fn

            return decorator

        server.tool = mock_tool
        register(server, mock_memory)
        result = captured["fn"]()
        assert "low_confidence" in result
        assert "high_roi" in result
        assert "total_tracked" in result
        assert result["total_tracked"] == 0

    def test_shows_boosted_record(self, mock_memory):
        from unittest.mock import MagicMock

        from memo.server_health import register

        rec = mock_memory.save(content="important recalled fact", title="Top ROI")
        for _ in range(5):
            mock_memory.store.boost_roi_batch([rec.id])

        server = MagicMock()
        captured = {}

        def mock_tool():
            def decorator(fn):
                captured["fn"] = fn
                return fn

            return decorator

        server.tool = mock_tool
        register(server, mock_memory)
        result = captured["fn"]()
        high_ids = [r["id"] for r in result["high_roi"]]
        assert rec.id in high_ids

    def test_shows_penalized_record(self, mock_memory):
        from unittest.mock import MagicMock

        from memo.server_health import register

        rec = mock_memory.save(content="contradicted old belief", title="Low Conf")
        mock_memory.store.penalize_confidence_batch([rec.id])

        server = MagicMock()
        captured = {}

        def mock_tool():
            def decorator(fn):
                captured["fn"] = fn
                return fn

            return decorator

        server.tool = mock_tool
        register(server, mock_memory)
        result = captured["fn"]()
        low_ids = [r["id"] for r in result["low_confidence"]]
        assert rec.id in low_ids
