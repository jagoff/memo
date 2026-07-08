"""CLI tests for the outcome loop surface."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from memo.cli import cli


def test_outcome_dead_min_surfaced_zero_disables_dead_weight(tmp_path) -> None:
    runner = CliRunner()
    env = {
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
        "MEMO_OUTCOME_DEAD_MIN_SURFACED": "0",
    }

    with (
        patch("memo.cli_outcome.Config") as mock_cfg_cls,
        patch("memo.memory.Memory") as mock_memory_cls,
        patch("memo.outcome.compute_utilities") as mock_compute,
        patch("memo.outcome.dead_weight") as mock_dead_weight,
    ):
        mock_cfg = MagicMock()
        mock_cfg.state_dir = tmp_path / "state"
        mock_cfg_cls.from_env.return_value = mock_cfg
        mock_memory_cls.return_value = MagicMock()
        mock_compute.return_value = {"by_prefix": {}, "prior_mean": 0.0}
        mock_dead_weight.return_value = []

        result = runner.invoke(cli, ["outcome", "--json"], env=env)

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["dead_weight"] == []
    mock_dead_weight.assert_called_once()
    assert mock_dead_weight.call_args.kwargs["min_surfaced"] == 0
    mock_memory_cls.return_value.close.assert_called_once_with()
