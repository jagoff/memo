"""CLI tests for `memo repo` command resource cleanup."""

from __future__ import annotations

import json
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


def _repo_index_result() -> dict[str, object]:
    return {
        "repo_id": "repo1",
        "name": "sample",
        "url": "https://example.test/sample.git",
        "ref": "main",
        "commit_sha": "abcdef1234567890",
        "clone_path": "/tmp/sample",
        "checked_files": 1,
        "indexed_files": 1,
        "unchanged_files": 0,
        "deleted_files": 0,
        "indexed_chunks": 1,
        "indexed_lines": 10,
        "embedded_chunks": 1,
        "pending_chunks": 0,
        "errors": 0,
        "semantic_status": "ready",
    }


def test_repo_index_json_closes_memory(tmp_path) -> None:
    mock_memory = MagicMock()
    mock_memory.repo_index.return_value = _repo_index_result()

    with patch("memo.memory.Memory", return_value=mock_memory):
        result = CliRunner().invoke(
            cli,
            ["repo", "index", "https://example.test/sample.git", "--json"],
            env=_isolated_env(tmp_path),
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["repo_id"] == "repo1"
    mock_memory.close.assert_called_once_with()


def test_repo_embed_json_closes_memory(tmp_path) -> None:
    mock_memory = MagicMock()
    mock_memory.repo_embed.return_value = {
        "name": "sample",
        "embedded_chunks": 1,
        "model_chunks": 1,
        "cached_chunks": 0,
        "total_chunks": 1,
        "pending_chunks": 0,
        "semantic_status": "ready",
    }

    with patch("memo.memory.Memory", return_value=mock_memory):
        result = CliRunner().invoke(
            cli,
            ["repo", "embed", "sample", "--json"],
            env=_isolated_env(tmp_path),
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["name"] == "sample"
    mock_memory.close.assert_called_once_with()


def test_repo_status_missing_closes_memory(tmp_path) -> None:
    mock_memory = MagicMock()
    mock_memory.repo_status.return_value = None

    with patch("memo.memory.Memory", return_value=mock_memory):
        result = CliRunner().invoke(
            cli,
            ["repo", "status", "missing"],
            env=_isolated_env(tmp_path),
        )

    assert result.exit_code == 1
    mock_memory.close.assert_called_once_with()


def test_repo_list_json_closes_memory(tmp_path) -> None:
    mock_memory = MagicMock()
    mock_memory.repo_list.return_value = [{"name": "sample"}]

    with patch("memo.memory.Memory", return_value=mock_memory):
        result = CliRunner().invoke(
            cli,
            ["repo", "list", "--json"],
            env=_isolated_env(tmp_path),
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)[0]["name"] == "sample"
    mock_memory.close.assert_called_once_with()


def test_repo_search_json_closes_memory(tmp_path) -> None:
    mock_memory = MagicMock()
    hit = MagicMock()
    hit.to_dict.return_value = {"path": "src/app.py"}
    mock_memory.repo_search.return_value = [hit]

    with patch("memo.memory.Memory", return_value=mock_memory):
        result = CliRunner().invoke(
            cli,
            ["repo", "search", "needle", "--json"],
            env=_isolated_env(tmp_path),
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)[0]["path"] == "src/app.py"
    mock_memory.close.assert_called_once_with()


def test_repo_get_missing_closes_memory(tmp_path) -> None:
    mock_memory = MagicMock()
    mock_memory.repo_get_file.return_value = None

    with patch("memo.memory.Memory", return_value=mock_memory):
        result = CliRunner().invoke(
            cli,
            ["repo", "get", "sample", "missing.py"],
            env=_isolated_env(tmp_path),
        )

    assert result.exit_code == 1
    mock_memory.close.assert_called_once_with()


def test_repo_delete_yes_closes_memory(tmp_path) -> None:
    mock_memory = MagicMock()
    mock_memory.repo_delete.return_value = True

    with patch("memo.memory.Memory", return_value=mock_memory):
        result = CliRunner().invoke(
            cli,
            ["repo", "delete", "sample", "--yes"],
            env=_isolated_env(tmp_path),
        )

    assert result.exit_code == 0, result.output
    mock_memory.close.assert_called_once_with()
