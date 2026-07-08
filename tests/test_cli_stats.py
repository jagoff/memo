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
        patch("memo.cli_stats.recall_health", return_value={}),
        patch("memo.cli_stats.consult_breakdown", return_value={"consumers": []}),
    ):
        result = runner.invoke(cli, ["stats"], env=env)

    assert result.exit_code == 0, result.output
    mock_memory.close.assert_called_once_with()
