"""Regression: `memo edit` must offer the same edit shapes as `memo_update`.

Found running memo as an end user: the MCP tool takes `append` and a surgical
`replace_old`/`replace_new` pair, while the CLI only had `--content`, a full
body replace. Editing one line of a long memory from the terminal meant
retyping the whole body — and `Memory.update` already supported both shapes.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from memo.cli import cli

BODY = "First line.\n\nSecond line mentions Postgres 16.\n\nThird line."


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


def _save(runner, env) -> str:
    saved = runner.invoke(cli, ["save", BODY, "--title", "db choice", "--json"], env=env)
    assert saved.exit_code == 0, saved.output
    return str(json.loads(saved.output)["id"])


def _body(runner, env, id_: str) -> str:
    got = runner.invoke(cli, ["get", id_, "--json"], env=env)
    assert got.exit_code == 0, got.output
    return str(json.loads(got.output)["body"])


def test_append_adds_a_paragraph_and_keeps_the_body(qa_env) -> None:
    runner = CliRunner()
    id_ = _save(runner, qa_env)

    edited = runner.invoke(cli, ["edit", id_, "--append", "Fourth line."], env=qa_env)
    assert edited.exit_code == 0, edited.output

    body = _body(runner, qa_env, id_)
    assert body.startswith(BODY)
    assert body.rstrip().endswith("Fourth line.")


def test_replace_is_surgical(qa_env) -> None:
    runner = CliRunner()
    id_ = _save(runner, qa_env)

    edited = runner.invoke(
        cli,
        ["edit", id_, "--replace-old", "Postgres 16", "--replace-new", "Postgres 17"],
        env=qa_env,
    )
    assert edited.exit_code == 0, edited.output

    body = _body(runner, qa_env, id_)
    assert "Postgres 17" in body
    assert "Postgres 16" not in body
    # Everything else stays byte-identical.
    assert body == BODY.replace("Postgres 16", "Postgres 17")


def test_replace_needs_both_halves(qa_env) -> None:
    runner = CliRunner()
    id_ = _save(runner, qa_env)

    result = runner.invoke(cli, ["edit", id_, "--replace-old", "Postgres 16"], env=qa_env)

    assert result.exit_code != 0
    assert "--replace-new" in result.output


def test_edit_shapes_are_mutually_exclusive(qa_env) -> None:
    runner = CliRunner()
    id_ = _save(runner, qa_env)

    result = runner.invoke(
        cli, ["edit", id_, "--content", "new body", "--append", "tail"], env=qa_env
    )

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower()


def test_replace_rejects_text_that_is_not_unique(qa_env) -> None:
    runner = CliRunner()
    saved = runner.invoke(cli, ["save", "repeat repeat", "--title", "twice", "--json"], env=qa_env)
    id_ = str(json.loads(saved.output)["id"])

    result = runner.invoke(
        cli, ["edit", id_, "--replace-old", "repeat", "--replace-new", "once"], env=qa_env
    )

    assert result.exit_code != 0, result.output
