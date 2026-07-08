"""CLI tests for relevance feedback commands."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from memo.cli import cli


def test_feedback_list_json_closes_memory(tmp_path) -> None:
    runner = CliRunner()
    env = {
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
    }

    with (
        patch("memo.cli_feedback.Config") as mock_cfg_cls,
        patch("memo.memory.Memory") as mock_memory_cls,
    ):
        mock_cfg_cls.from_env.return_value = MagicMock()
        mock_memory = MagicMock()
        mock_memory.feedback_list.return_value = [{"source_id": "abc123", "rating": 1}]
        mock_memory_cls.return_value = mock_memory

        result = runner.invoke(cli, ["feedback", "list", "--as-json"], env=env)

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)[0]["source_id"] == "abc123"
    mock_memory.feedback_list.assert_called_once_with(source_id=None, limit=50)
    mock_memory.close.assert_called_once_with()
