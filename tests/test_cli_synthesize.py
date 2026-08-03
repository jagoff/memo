"""CLI-plumbing tests for `memo synthesize`'s plain-text output.

Business logic (`Memory.synthesize_cross_cluster`) is covered end-to-end in
test_synthesize.py; these tests only exercise the CLI's rendering of results
it's handed, using a mocked Memory.
"""

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


def test_synthesize_plain_text_renders_letter_leading_id_and_rationale(tmp_path) -> None:
    """Regression: a bracketed id/confidence starting with a letter (or any
    bracket-shaped content in the title/rationale) was silently dropped by
    Rich's markup parser when the surrounding brackets went unescaped.
    """
    mock_memory = MagicMock()
    mock_memory.synthesize_cross_cluster.return_value = [
        {
            "id": "ad957d80cccc",
            "saved": True,
            "title": "Insight with a [bracketed] title",
            "confidence": "high",
            "sources": ["s1", "s2"],
            "rationale": "Rationale mentioning [scope] explicitly.",
        }
    ]

    with patch("memo.memory.Memory", return_value=mock_memory):
        result = CliRunner().invoke(cli, ["synthesize"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "ad957d80" in result.output
    assert "[bracketed]" in result.output
    assert "[scope]" in result.output


def test_synthesize_plain_text_no_candidates(tmp_path) -> None:
    mock_memory = MagicMock()
    mock_memory.synthesize_cross_cluster.return_value = []

    with patch("memo.memory.Memory", return_value=mock_memory):
        result = CliRunner().invoke(cli, ["synthesize"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "no synthesis candidates found" in result.output
