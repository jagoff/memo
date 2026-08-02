"""CLI tests for corpus diff and per-record history commands."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from memo.cli import cli


def test_diff_json_closes_memory(tmp_path) -> None:
    runner = CliRunner()
    env = {
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
    }
    diff_result = SimpleNamespace(
        from_ts=datetime(2026, 1, 1, tzinfo=UTC),
        to_ts=datetime(2026, 1, 2, tzinfo=UTC),
        added=[],
        removed=[],
        updated=[],
    )
    mock_memory = MagicMock()

    with (
        patch("memo.memory.Memory", return_value=mock_memory),
        patch("memo.time_machine.diff", return_value=diff_result),
    ):
        result = runner.invoke(cli, ["diff", "--from", "2026-01-01", "--json"], env=env)

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["added"] == []
    mock_memory.close.assert_called_once_with()


def test_diff_plain_text_renders_letter_leading_ids(tmp_path) -> None:
    """Regression: Rich swallowed a bracketed id whose text starts with a-f.

    `[ad957d80]` printed raw is parsed as an (invalid) style tag and silently
    dropped; escape() must wrap the whole bracketed substring, not just its
    contents, for the id to actually render.
    """
    runner = CliRunner()
    env = {
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
    }
    added = SimpleNamespace(id="ad957d80aaaa", title="Added memory", type="note")
    removed = SimpleNamespace(id="d81cb7e6bbbb", title="Removed memory", type="fact")
    diff_result = SimpleNamespace(
        from_ts=datetime(2026, 1, 1, tzinfo=UTC),
        to_ts=datetime(2026, 1, 2, tzinfo=UTC),
        added=[added],
        removed=[removed],
        updated=[{"id": "beef0000cccc", "title": "Updated memory", "changed_fields": ["title"]}],
        summary=lambda: "1 added, 1 removed, 1 updated",
    )
    mock_memory = MagicMock()

    with (
        patch("memo.memory.Memory", return_value=mock_memory),
        patch("memo.time_machine.diff", return_value=diff_result),
    ):
        result = runner.invoke(cli, ["diff", "--from", "2026-01-01"], env=env)

    assert result.exit_code == 0, result.output
    assert "ad957d80" in result.output
    assert "d81cb7e6" in result.output
    assert "beef0000" in result.output


def test_record_history_missing_record_closes_memory(tmp_path) -> None:
    runner = CliRunner()
    env = {
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
    }
    mock_memory = MagicMock()
    mock_memory.resolve_id.return_value = None

    with patch("memo.memory.Memory", return_value=mock_memory):
        result = runner.invoke(cli, ["record-history", "missing"], env=env)

    assert result.exit_code == 1
    mock_memory.close.assert_called_once_with()
