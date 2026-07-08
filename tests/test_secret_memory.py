"""Tests for secret storage Memory integration."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from memo.cli import cli


# Memory mixin tests (placeholder)
# Full tests require working Memory + conftest fixtures
# Smoke test: verify Memory has secret methods
def test_memory_has_secret_methods():
    """Memory should have secret operation methods."""
    from memo.memory import Memory

    assert hasattr(Memory, "save_secret")
    assert hasattr(Memory, "get_secret")
    assert hasattr(Memory, "list_secrets")
    assert hasattr(Memory, "forget_secret")


def test_secret_list_closes_memory(tmp_path):
    mock_memory = MagicMock()
    mock_memory.list_secrets.return_value = []

    with (
        patch("memo.cli_secret.Config.from_env", return_value=MagicMock()),
        patch("memo.cli_secret.Memory", return_value=mock_memory),
    ):
        result = CliRunner().invoke(cli, ["secret", "list"], env={"MEMO_NONINTERACTIVE": "1"})

    assert result.exit_code == 0, result.output
    mock_memory.close.assert_called_once_with()


def test_secret_get_closes_memory_on_error(tmp_path):
    mock_memory = MagicMock()
    mock_memory.get_secret.side_effect = RuntimeError("missing")

    with (
        patch("memo.cli_secret.Config.from_env", return_value=MagicMock()),
        patch("memo.cli_secret.Memory", return_value=mock_memory),
    ):
        result = CliRunner().invoke(
            cli,
            ["secret", "get", "--name", "missing"],
            env={"MEMO_NONINTERACTIVE": "1"},
        )

    assert result.exit_code != 0
    mock_memory.close.assert_called_once_with()
