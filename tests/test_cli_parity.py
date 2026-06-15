"""CLI parity tests — memo chat ask, memo search --rerank, memo import csv/json.

Task 10: Add CLI commands for features that previously only had MCP equivalents.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from memo.cli import cli

# ---------------------------------------------------------------------------
# Shared env fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def runner_env(tmp_path: Path) -> dict[str, str]:
    """Isolated CliRunner environment."""
    data = tmp_path / "data"
    data.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    vault = tmp_path / "vault"
    vault.mkdir()
    return {
        "MEMO_DATA_DIR": str(data),
        "MEMO_STATE_DIR": str(state),
        "MEMO_VAULT_PATH": str(vault),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
        # Disable reranker by default in tests (no MLX needed)
        "MEMO_RERANKER_ENABLED": "0",
    }


# ---------------------------------------------------------------------------
# 10a — memo chat ask
# ---------------------------------------------------------------------------


class TestChatAskCommand:
    """memo chat ask <question> [--stream] [--json]"""

    def test_chat_group_exists(self, runner_env: dict[str, str]) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["chat", "--help"], env=runner_env)
        assert result.exit_code == 0
        assert "ask" in result.output

    def test_chat_ask_help(self, runner_env: dict[str, str]) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["chat", "ask", "--help"], env=runner_env)
        assert result.exit_code == 0
        assert "--stream" in result.output
        assert "--json" in result.output
        assert "QUESTION" in result.output

    def test_chat_ask_calls_memory_chat_ask(
        self, runner_env: dict[str, str], tmp_path: Path
    ) -> None:
        """chat ask delegates to memory.chat_ask and prints the answer."""
        fake_envelope = {
            "answer": "MLX is Apple's ML framework.",
            "sources": [],
            "question": "what do I know about MLX?",
        }

        runner = CliRunner()
        with patch("memo.cli_chat.Config") as mock_cfg_cls, patch(
            "memo.cli_chat._get_memory"
        ) as mock_get_mem:
            mock_cfg = MagicMock()
            mock_cfg_cls.from_env.return_value = mock_cfg
            mock_mem = MagicMock()
            mock_mem.chat_ask.return_value = fake_envelope
            mock_get_mem.return_value = mock_mem

            result = runner.invoke(
                cli,
                ["chat", "ask", "what do I know about MLX?"],
                env=runner_env,
            )

        assert result.exit_code == 0, result.output
        mock_mem.chat_ask.assert_called_once()
        call_kwargs = mock_mem.chat_ask.call_args
        assert call_kwargs[0][0] == "what do I know about MLX?"

    def test_chat_ask_json_output(self, runner_env: dict[str, str]) -> None:
        """--json flag emits JSON envelope."""
        fake_envelope = {
            "answer": "The answer.",
            "sources": [],
            "question": "test?",
        }

        runner = CliRunner()
        with patch("memo.cli_chat.Config") as mock_cfg_cls, patch(
            "memo.cli_chat._get_memory"
        ) as mock_get_mem:
            mock_cfg_cls.from_env.return_value = MagicMock()
            mock_mem = MagicMock()
            mock_mem.chat_ask.return_value = fake_envelope
            mock_get_mem.return_value = mock_mem

            result = runner.invoke(
                cli,
                ["chat", "ask", "--json", "test?"],
                env=runner_env,
            )

        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert parsed["answer"] == "The answer."

    def test_chat_ask_stream_calls_chat_ask_stream(
        self, runner_env: dict[str, str]
    ) -> None:
        """--stream calls chat_ask_stream and emits NDJSON events."""
        events = [
            {"type": "context", "sources": []},
            {"type": "token", "delta": "Hello "},
            {"type": "token", "delta": "world."},
            {"type": "done", "answer": "Hello world."},
        ]

        runner = CliRunner()
        with patch("memo.cli_chat.Config") as mock_cfg_cls, patch(
            "memo.cli_chat._get_memory"
        ) as mock_get_mem:
            mock_cfg_cls.from_env.return_value = MagicMock()
            mock_mem = MagicMock()
            mock_mem.chat_ask_stream.return_value = iter(events)
            mock_get_mem.return_value = mock_mem

            result = runner.invoke(
                cli,
                ["chat", "ask", "--stream", "explain my setup"],
                env=runner_env,
            )

        assert result.exit_code == 0, result.output
        lines = [ln for ln in result.output.splitlines() if ln.strip()]
        assert len(lines) == len(events)
        # Each line is valid JSON
        for i, line in enumerate(lines):
            obj = json.loads(line)
            assert obj["type"] == events[i]["type"]

    def test_chat_group_distinct_from_flat_chat_ask(
        self, runner_env: dict[str, str]
    ) -> None:
        """Both memo chat-ask and memo chat ask are reachable."""
        runner = CliRunner()
        # flat hyphenated form (backwards compat)
        result_flat = runner.invoke(cli, ["chat-ask", "--help"], env=runner_env)
        assert result_flat.exit_code == 0

        # group form (new)
        result_group = runner.invoke(cli, ["chat", "ask", "--help"], env=runner_env)
        assert result_group.exit_code == 0


# ---------------------------------------------------------------------------
# 10b — memo search --rerank / --no-rerank
# ---------------------------------------------------------------------------


class TestSearchRerankFlag:
    """memo search <query> [--rerank | --no-rerank]"""

    def test_search_rerank_flag_in_help(self, runner_env: dict[str, str]) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "--help"], env=runner_env)
        assert result.exit_code == 0
        assert "--rerank" in result.output
        assert "--no-rerank" in result.output

    def test_search_no_rerank_passes_disable_reranker_true(
        self, runner_env: dict[str, str]
    ) -> None:
        """--no-rerank must pass disable_reranker=True to memory.search."""
        runner = CliRunner()
        with patch("memo.cli_search.Config") as mock_cfg_cls, patch(
            "memo.cli_search._get_memory"
        ) as mock_get_mem:
            mock_cfg_cls.from_env.return_value = MagicMock()
            mock_mem = MagicMock()
            mock_mem.search.return_value = []
            mock_get_mem.return_value = mock_mem

            result = runner.invoke(
                cli,
                ["search", "--no-rerank", "MLX"],
                env=runner_env,
            )

        assert result.exit_code == 0, result.output
        call_kwargs = mock_mem.search.call_args[1]
        assert call_kwargs.get("disable_reranker") is True

    def test_search_rerank_sets_env_and_passes_disable_false(
        self, runner_env: dict[str, str]
    ) -> None:
        """--rerank must set MEMO_RERANKER_ENABLED=1 and pass disable_reranker=False."""
        runner = CliRunner()
        captured_env: dict = {}

        def _fake_from_env():
            captured_env["MEMO_RERANKER_ENABLED"] = os.environ.get("MEMO_RERANKER_ENABLED")
            cfg = MagicMock()
            cfg.reranker_enabled = True
            return cfg

        with patch("memo.cli_search.Config") as mock_cfg_cls, patch(
            "memo.cli_search._get_memory"
        ) as mock_get_mem:
            mock_cfg_cls.from_env.side_effect = _fake_from_env
            mock_mem = MagicMock()
            mock_mem.search.return_value = []
            mock_get_mem.return_value = mock_mem

            result = runner.invoke(
                cli,
                ["search", "--rerank", "MLX"],
                env={**runner_env, "MEMO_RERANKER_ENABLED": "0"},
            )

        assert result.exit_code == 0, result.output
        assert captured_env.get("MEMO_RERANKER_ENABLED") == "1"
        call_kwargs = mock_mem.search.call_args[1]
        assert call_kwargs.get("disable_reranker") is False

    def test_search_default_no_rerank_override(self, runner_env: dict[str, str]) -> None:
        """When neither --rerank nor --no-rerank is given, disable_reranker is False."""
        runner = CliRunner()
        with patch("memo.cli_search.Config") as mock_cfg_cls, patch(
            "memo.cli_search._get_memory"
        ) as mock_get_mem:
            mock_cfg_cls.from_env.return_value = MagicMock()
            mock_mem = MagicMock()
            mock_mem.search.return_value = []
            mock_get_mem.return_value = mock_mem

            result = runner.invoke(
                cli,
                ["search", "MLX"],
                env=runner_env,
            )

        assert result.exit_code == 0, result.output
        call_kwargs = mock_mem.search.call_args[1]
        # disable_reranker defaults to False when use_rerank is None
        assert call_kwargs.get("disable_reranker") is False


# ---------------------------------------------------------------------------
# 10c — memo import csv / json (already implemented, smoke test)
# ---------------------------------------------------------------------------


class TestImportCommands:
    """memo import csv <file> and memo import json <file>."""

    def test_import_csv_help(self, runner_env: dict[str, str]) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["import", "csv", "--help"], env=runner_env)
        assert result.exit_code == 0
        assert "INPUT_PATH" in result.output

    def test_import_json_help(self, runner_env: dict[str, str]) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["import", "json", "--help"], env=runner_env)
        assert result.exit_code == 0
        assert "INPUT_PATH" in result.output

    def test_import_csv_calls_import_from(
        self, runner_env: dict[str, str], tmp_path: Path
    ) -> None:
        """import csv delegates to memory.import_export.import_from."""
        csv_file = tmp_path / "test.csv"
        # Write a minimal CSV
        with csv_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["title", "content", "type", "tags"])
            writer.writeheader()
            writer.writerow({"title": "T1", "content": "body1", "type": "note", "tags": ""})

        from memo.import_export import ImportResult

        fake_result = ImportResult(imported_count=1, skipped_count=0, errors=[])

        runner = CliRunner()
        with patch("memo.cli_import.Config") as mock_cfg_cls, patch(
            "memo.cli_import._get_memory"
        ) as mock_get_mem:
            mock_cfg_cls.from_env.return_value = MagicMock()
            mock_mem = MagicMock()
            mock_mem.import_export.import_from.return_value = fake_result
            mock_get_mem.return_value = mock_mem

            result = runner.invoke(
                cli,
                ["import", "csv", str(csv_file)],
                env=runner_env,
            )

        assert result.exit_code == 0, result.output
        assert "Import complete" in result.output
        mock_mem.import_export.import_from.assert_called_once()
        call_args = mock_mem.import_export.import_from.call_args
        assert call_args[0][1] == "csv"

    def test_import_json_calls_import_from(
        self, runner_env: dict[str, str], tmp_path: Path
    ) -> None:
        """import json delegates to memory.import_export.import_from."""
        json_file = tmp_path / "test.json"
        json_file.write_text(
            json.dumps([{"title": "T1", "content": "body1", "type": "note", "tags": []}]),
            encoding="utf-8",
        )

        from memo.import_export import ImportResult

        fake_result = ImportResult(imported_count=1, skipped_count=0, errors=[])

        runner = CliRunner()
        with patch("memo.cli_import.Config") as mock_cfg_cls, patch(
            "memo.cli_import._get_memory"
        ) as mock_get_mem:
            mock_cfg_cls.from_env.return_value = MagicMock()
            mock_mem = MagicMock()
            mock_mem.import_export.import_from.return_value = fake_result
            mock_get_mem.return_value = mock_mem

            result = runner.invoke(
                cli,
                ["import", "json", str(json_file)],
                env=runner_env,
            )

        assert result.exit_code == 0, result.output
        assert "Import complete" in result.output
        mock_mem.import_export.import_from.assert_called_once()
        call_args = mock_mem.import_export.import_from.call_args
        assert call_args[0][1] == "json"
