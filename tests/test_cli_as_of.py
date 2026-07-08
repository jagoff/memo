"""CLI tests for `memo as-of` commands."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from memo.cli import cli


def _isolated_env(tmp_path) -> dict[str, str]:
    return {
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
    }


def test_as_of_search_json_closes_memory(tmp_path) -> None:
    runner = CliRunner()
    mock_memory = MagicMock()
    hit = MagicMock(type="fact")
    hit.to_dict.return_value = {"id": "abc123", "type": "fact"}

    class SearchSnapshot:
        as_of = datetime(2026, 1, 1, tzinfo=UTC)

        def __len__(self) -> int:
            return 1

        def search(self, *_args, **_kwargs):
            assert not mock_memory.close.called
            return [hit]

    snap = SearchSnapshot()

    with (
        patch("memo.memory.Memory", return_value=mock_memory),
        patch("memo.time_machine.reconstruct", return_value=snap),
    ):
        result = runner.invoke(
            cli,
            ["as-of", "search", "query", "--date", "2026-01-01", "--json"],
            env=_isolated_env(tmp_path),
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["results"] == [{"id": "abc123", "type": "fact"}]
    mock_memory.close.assert_called_once_with()


def test_as_of_ask_json_closes_memory(tmp_path) -> None:
    runner = CliRunner()
    mock_memory = MagicMock()

    class AskSnapshot:
        def ask(self, *_args, **_kwargs):
            assert not mock_memory.close.called
            return {"answer": "known", "sources": []}

    snap = AskSnapshot()

    with (
        patch("memo.memory.Memory", return_value=mock_memory),
        patch("memo.time_machine.reconstruct", return_value=snap),
    ):
        result = runner.invoke(
            cli,
            ["as-of", "ask", "question", "--date", "2026-01-01", "--json"],
            env=_isolated_env(tmp_path),
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["answer"] == "known"
    mock_memory.close.assert_called_once_with()


def test_as_of_list_json_closes_memory(tmp_path) -> None:
    runner = CliRunner()
    mock_memory = MagicMock()
    snap = MagicMock()
    snap.as_of = datetime(2026, 1, 1, tzinfo=UTC)
    snap.list.return_value = [
        SimpleNamespace(
            id="abc123",
            title="Title",
            type="note",
            tags=["x"],
            updated="2026-01-01T00:00:00+00:00",
        )
    ]
    snap.__len__.return_value = 1

    with (
        patch("memo.memory.Memory", return_value=mock_memory),
        patch("memo.time_machine.reconstruct", return_value=snap),
    ):
        result = runner.invoke(
            cli,
            ["as-of", "list", "--date", "2026-01-01", "--json"],
            env=_isolated_env(tmp_path),
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["records"][0]["id"] == "abc123"
    mock_memory.close.assert_called_once_with()
