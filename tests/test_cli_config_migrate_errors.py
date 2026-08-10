"""`memo config migrate` reports a pre-existing config instead of crashing.

`config init` and `config migrate` call the same `write_default_config`, which
raises `FileExistsError` rather than clobber a Markdown config. `init` wrapped
it in a `ClickException`; `migrate` did not, so on 2026-08-09 a plain
`memo config migrate` on a machine that already had a config printed a raw
traceback — and, because the legacy-config snapshot was taken FIRST, left a
`pre-md-config` backup behind for a migration that never ran.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from memo.cli import cli


@pytest.fixture
def config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "xdg"
    (home / "memo").mkdir(parents=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(home / "memo" / "config.toml"))
    # Per-test Markdown config home; the session default in tests/conftest.py is
    # one shared tmp path, so two migrations in one run would see each other's.
    monkeypatch.setenv("MEMO_CONFIG_DIR", str(home / "memo"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    return home


def _legacy_config(config_home: Path, data_dir: Path) -> None:
    (config_home / "memo" / "config.toml").write_text(
        f'[storage]\ndata_dir = "{data_dir}"\n', encoding="utf-8"
    )


def test_migrate_reports_an_existing_config_without_a_traceback(
    config_home: Path, tmp_path: Path
) -> None:
    data_dir = tmp_path / "memorias"
    data_dir.mkdir()
    _legacy_config(config_home, data_dir)
    runner = CliRunner()

    first = runner.invoke(cli, ["config", "migrate"])
    assert first.exit_code == 0, first.output

    second = runner.invoke(cli, ["config", "migrate"])

    assert second.exit_code != 0
    assert "Traceback" not in second.output
    assert "already exists" in second.output
    assert "--force" in second.output


def test_a_refused_migration_leaves_no_backup(config_home: Path, tmp_path: Path) -> None:
    """The legacy snapshot is a record of a migration that happened."""
    data_dir = tmp_path / "memorias"
    data_dir.mkdir()
    _legacy_config(config_home, data_dir)
    runner = CliRunner()
    first = runner.invoke(cli, ["config", "migrate"])
    assert first.exit_code == 0, first.output

    before = sorted(p.name for p in (config_home / "memo").glob("*pre-md-config*"))
    runner.invoke(cli, ["config", "migrate"])
    after = sorted(p.name for p in (config_home / "memo").glob("*pre-md-config*"))

    assert after == before
