"""Regression: memory content is data, not Rich markup.

Found running memo as an end user: a body of
``Ver la doc en [Context7](url) y el issue [#42]. Config: array[index]``
rendered as ``Ver la doc en [Context7](url) y el issue . Config: array`` —
Rich parsed the square brackets as style tags and swallowed them. The same
bug hid citation ids: ``[fae467f3]`` disappeared while ``[36ee3a85]``
survived, because only ids starting with a letter look like a style tag. The
markdown on disk was always correct; only what the user reads was wrong.

Every surface that renders a record's own text must escape it.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from memo.cli import cli

# A title/body pair that exercises the three failure shapes: a markdown link,
# a bracketed token that looks like an unknown style tag, and a real Rich tag.
MARKUP_TITLE = "Doc [Context7] y [b]negrita[/b]"
MARKUP_BODY = "Ver [Context7](https://context7.com), issue [#42], array[index], [bold] literal."


@pytest.fixture
def qa_env(tmp_path, monkeypatch):
    env = {
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
        # Rich strips ANSI when not a tty, but still parses markup — which is
        # exactly the bug. Keep the width wide so nothing wraps mid-token.
        "COLUMNS": "200",
    }
    return env


def _save(runner, env, *, title=MARKUP_TITLE, body=MARKUP_BODY) -> str:
    saved = runner.invoke(
        cli, ["save", body, "--title", title, "--type", "note", "--json"], env=env
    )
    assert saved.exit_code == 0, saved.output
    return str(json.loads(saved.output)["id"])


def test_get_renders_body_and_title_literally(qa_env) -> None:
    runner = CliRunner()
    id_ = _save(runner, qa_env)

    shown = runner.invoke(cli, ["get", id_], env=qa_env)
    assert shown.exit_code == 0, shown.output
    # Nothing may be swallowed: every bracketed token survives verbatim.
    for token in ("[Context7]", "[#42]", "array[index]", "[bold]", "[b]negrita[/b]"):
        assert token in shown.output, f"{token!r} was eaten by markup parsing"


def test_list_table_renders_title_literally(qa_env) -> None:
    runner = CliRunner()
    _save(runner, qa_env)

    listed = runner.invoke(cli, ["list", "--limit", "5"], env=qa_env)
    assert listed.exit_code == 0, listed.output
    assert "[Context7]" in listed.output
    assert "[b]negrita[/b]" in listed.output


def test_save_receipt_renders_title_literally(qa_env) -> None:
    runner = CliRunner()
    saved = runner.invoke(
        cli, ["save", MARKUP_BODY, "--title", MARKUP_TITLE, "--type", "note"], env=qa_env
    )
    assert saved.exit_code == 0, saved.output
    assert "[Context7]" in saved.output


def test_digest_command_renders_nudge_titles_literally(qa_env, monkeypatch) -> None:
    """`memo digest` prints memory-derived nudge titles through Rich."""
    # cli_proactive imports these lazily inside the command body.
    from memo.proactive import engine, surfaces

    monkeypatch.setattr(surfaces, "render_digest", lambda _routed: MARKUP_TITLE)
    monkeypatch.setattr(engine, "compute_routed", lambda *_a, **_k: object())

    env = {**qa_env, "MEMO_PROACTIVE_ENABLED": "1"}
    shown = CliRunner().invoke(cli, ["digest"], env=env)

    assert shown.exit_code == 0, shown.output
    assert "[b]negrita[/b]" in shown.output


def test_as_of_list_renders_title_literally(qa_env) -> None:
    runner = CliRunner()
    _save(runner, qa_env)

    listed = runner.invoke(cli, ["as-of", "list", "--date", "2099-01-01"], env=qa_env)
    assert listed.exit_code == 0, listed.output
    assert "[b]negrita[/b]" in listed.output


def test_update_and_rename_receipts_render_title_literally(qa_env) -> None:
    runner = CliRunner()
    id_ = _save(runner, qa_env, title="plain title")

    edited = runner.invoke(cli, ["edit", id_, "--title", MARKUP_TITLE], env=qa_env)
    assert edited.exit_code == 0, edited.output
    assert "[Context7]" in edited.output

    renamed = runner.invoke(cli, ["rename", MARKUP_TITLE, id_], env=qa_env)
    assert renamed.exit_code == 0, renamed.output
    assert "[Context7]" in renamed.output
