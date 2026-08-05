"""Regression: the CLI must reject the inputs the MCP surface already rejects.

Found running memo as an end user: `memo search q --limit 0` and
`memo list --limit -5` exited 0 and printed an empty table, and `memo ask ""`
printed an empty answer panel and exited 0. The MCP tools clamp `limit` to
1..500 and require a non-empty question, so the same mistake was caught there
and silently accepted here.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from memo.cli import cli


@pytest.fixture
def qa_env(tmp_path):
    return {
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
    }


@pytest.mark.parametrize(
    "argv",
    [
        ["search", "anything", "--limit", "0"],
        ["search", "anything", "--limit", "-5"],
        ["search", "anything", "--limit", "501"],
        ["list", "--limit", "0"],
        ["list", "--limit", "-5"],
        ["ask", "anything", "--k", "0"],
    ],
)
def test_out_of_range_limits_are_rejected(qa_env, argv) -> None:
    result = CliRunner().invoke(cli, argv, env=qa_env)

    assert result.exit_code != 0, result.output
    assert "range" in result.output.lower() or "invalid value" in result.output.lower()


def test_in_range_limits_still_work(qa_env) -> None:
    listed = CliRunner().invoke(cli, ["list", "--limit", "1"], env=qa_env)
    assert listed.exit_code == 0, listed.output


@pytest.mark.parametrize("question", ["", "   "])
def test_ask_requires_a_question(qa_env, question) -> None:
    result = CliRunner().invoke(cli, ["ask", question], env=qa_env)

    assert result.exit_code != 0, result.output
    assert "non-empty" in result.output.lower()
