"""CLI-plumbing tests for `memo entity`'s plain-text output."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from memo.cli import cli


def _env(tmp_path) -> dict[str, str]:
    return {
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
    }


def test_entity_plain_text_renders_letter_leading_id(tmp_path) -> None:
    """Regression: `[{escape(mid[:8])}]` still had unescaped literal brackets
    around the escaped id, so a letter-leading id was still swallowed by
    Rich's markup parser.
    """
    mock_memory = MagicMock()
    mock_memory.graph.entity_memories.return_value = ["ad957d80cccc"]
    mock_memory.store.get.return_value = {
        "title": "A memory with a [bracketed] title",
        "updated": "2026-01-01T00:00:00",
    }

    with patch("memo.memory.Memory", return_value=mock_memory):
        result = CliRunner().invoke(cli, ["entity", "postgres"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "ad957d80" in result.output
    assert "[bracketed]" in result.output


def test_entity_no_matches(tmp_path) -> None:
    mock_memory = MagicMock()
    mock_memory.graph.entity_memories.return_value = []

    with patch("memo.memory.Memory", return_value=mock_memory):
        result = CliRunner().invoke(cli, ["entity", "nonexistent"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "no memories mention" in result.output
