"""CLI tests for the `memo stats` command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from memo.cli import cli


def test_stats_closes_memory(tmp_path: Path) -> None:
    runner = CliRunner()
    env = {
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
    }

    cfg = MagicMock()
    cfg.state_dir = tmp_path / "state"
    cfg.data_dir = tmp_path / "data"
    cfg.model_profile = "balanced"
    cfg.embedder_model = "org/embedder"
    cfg.llm_model = "org/llm"

    mock_memory = MagicMock()
    mock_memory.cfg = cfg
    mock_memory.store.count.return_value = 3
    mock_memory.store._conn.execute.return_value.fetchone.return_value = (0,)

    with (
        patch("memo.cli_stats.Config.from_env", return_value=cfg),
        patch("memo.cli_stats.Memory", return_value=mock_memory),
        patch("memo.cli_stats.read_context_cost_log", return_value=[]),
        patch("memo.cli_stats.read_grounding_log", return_value=[]),
        patch(
            "memo.cli_stats.recall_health",
            return_value={
                "fired": 1,
                "hit_rate": 1.0,
                "composite_score_threshold": 0.85,
                "top_composite_score_rate": 1.0,
                "p50_latency_ms": 12,
            },
        ),
        patch("memo.cli_stats.consult_breakdown", return_value={"consumers": []}),
    ):
        result = runner.invoke(cli, ["stats"], env=env)

    assert result.exit_code == 0, result.output
    assert "top composite 100% (final ranking score >0.85)" in result.output
    assert "strong hits" not in result.output
    mock_memory.close.assert_called_once_with()


def test_stats_json_emits_stable_report(tmp_path: Path) -> None:
    runner = CliRunner()
    env = {
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
    }
    cfg = MagicMock(
        state_dir=tmp_path / "state",
        data_dir=tmp_path / "data",
        model_profile="balanced",
        llm_model="org/llm",
    )
    mock_memory = MagicMock(cfg=cfg)
    mock_memory.store.count.return_value = 3
    mock_memory.store.embedder_model = "org/embedder"
    with (
        patch("memo.cli_stats.Config.from_env", return_value=cfg),
        patch("memo.cli_stats.Memory", return_value=mock_memory),
        patch("memo.cli_stats.read_context_cost_log", return_value=[{"tokens_est": 12}]),
        patch("memo.dashboard.read_usage_log", return_value=[{"id": "mem-1"}]),
        patch("memo.cli_stats.recall_health", return_value={"fired": 0}),
        patch("memo.cli_stats.consult_breakdown", return_value={"consumers": []}),
    ):
        result = runner.invoke(cli, ["stats", "--json"], env=env)

    assert result.exit_code == 0, result.output
    assert '"schema": "memo.stats.v2"' in result.output
    assert '"total": 3' in result.output
    assert '"context_tokens_injected": 12' in result.output
    assert '"memories_surfaced": 1' in result.output
    assert '"tokens_saved"' not in result.output
    mock_memory.close.assert_called_once_with()
