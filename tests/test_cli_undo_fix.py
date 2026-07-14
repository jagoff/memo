"""`memo undo` / `memo fix` — thin verbs over Memory.delete/update for
correcting an auto-captured memory named in a capture receipt."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from memo.cli import cli


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def test_undo_removes_a_saved_memory(tmp_path: Path) -> None:
    r = CliRunner()
    env = _env(tmp_path)
    save = r.invoke(
        cli, ["save", "--type", "note", "--defer-embed", "--json", "throwaway capture"], env=env
    )
    assert save.exit_code == 0, save.output
    mid = json.loads(save.output)["id"]

    out = r.invoke(cli, ["undo", mid], env=env)
    assert out.exit_code == 0, out.output
    assert "removed" in out.output.lower()

    get = r.invoke(cli, ["get", mid], env=env)
    assert get.exit_code == 1
    assert "not found" in get.output.lower()


def test_fix_updates_title(tmp_path: Path) -> None:
    r = CliRunner()
    env = _env(tmp_path)
    save = r.invoke(cli, ["save", "--type", "note", "--defer-embed", "--json", "old body"], env=env)
    assert save.exit_code == 0, save.output
    mid = json.loads(save.output)["id"]

    out = r.invoke(cli, ["fix", mid, "--title", "corrected title"], env=env)
    assert out.exit_code == 0, out.output

    get = r.invoke(cli, ["get", mid, "--json"], env=env)
    assert get.exit_code == 0, get.output
    assert json.loads(get.output)["title"] == "corrected title"
