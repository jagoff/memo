from __future__ import annotations

import sqlite3
import zipfile
from contextlib import closing
from pathlib import Path

from click.testing import CliRunner

from memo.cli_memory import restore


def _restore_env(data_dir: Path, state_dir: Path) -> dict[str, str]:
    return {
        "MEMO_DATA_DIR": str(data_dir),
        "MEMO_STATE_DIR": str(state_dir),
        "MEMO_NONINTERACTIVE": "1",
    }


def test_restore_portable_backup_writes_to_memory_dir_when_memories_live_in_vault(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    vault = tmp_path / "vault"
    memory_dir = vault / "Obsidian" / "AI" / "memory"
    data_dir.mkdir()
    state_dir.mkdir()
    memory_dir.mkdir(parents=True)

    backup = tmp_path / "memo-backup.zip"
    with zipfile.ZipFile(backup, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", "{}")
        zf.writestr("memory/project/example.md", "restored body")

    env = {
        "MEMO_DATA_DIR": str(data_dir),
        "MEMO_STATE_DIR": str(state_dir),
        "MEMO_VAULT_PATH": str(vault),
        "MEMO_MEMORIES_IN_VAULT": "1",
        "MEMO_NONINTERACTIVE": "1",
    }

    result = CliRunner().invoke(restore, [str(backup), "--yes"], env=env)

    assert result.exit_code == 0, result.output
    assert (memory_dir / "project" / "example.md").read_text(encoding="utf-8") == "restored body"
    assert not (data_dir / "project" / "example.md").exists()


def test_restore_rejects_state_token_overwrite(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    data_dir.mkdir()
    state_dir.mkdir()
    token = state_dir / "http-api-token"
    token.write_text("SAFE-TOKEN", encoding="utf-8")
    backup = tmp_path / "malicious-token.zip"
    with zipfile.ZipFile(backup, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", "{}")
        zf.writestr("state/http-api-token", "ATTACKER-TOKEN")

    result = CliRunner().invoke(
        restore,
        [str(backup), "--yes"],
        env=_restore_env(data_dir, state_dir),
    )

    assert result.exit_code != 0
    assert "unexpected state archive member" in result.output
    assert token.read_text(encoding="utf-8") == "SAFE-TOKEN"


def test_restore_rejects_unrelated_state_file_without_writing_it(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    data_dir.mkdir()
    state_dir.mkdir()
    backup = tmp_path / "malicious-unrelated.zip"
    with zipfile.ZipFile(backup, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", "{}")
        zf.writestr("state/unrelated/owned.txt", "OWNED")

    result = CliRunner().invoke(
        restore,
        [str(backup), "--yes"],
        env=_restore_env(data_dir, state_dir),
    )

    assert result.exit_code != 0
    assert "unexpected state archive member" in result.output
    assert not (state_dir / "unrelated" / "owned.txt").exists()


def test_restore_preflights_every_member_before_writing_valid_entries(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    data_dir.mkdir()
    state_dir.mkdir()
    backup = tmp_path / "valid-then-malicious.zip"
    with zipfile.ZipFile(backup, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", "{}")
        zf.writestr("memory/project/valid.md", "must not be partially restored")
        zf.writestr("state/unrelated.txt", "invalid trailing member")

    result = CliRunner().invoke(
        restore,
        [str(backup), "--yes"],
        env=_restore_env(data_dir, state_dir),
    )

    assert result.exit_code != 0
    assert "unexpected state archive member" in result.output
    assert not (data_dir / "project" / "valid.md").exists()


def test_restore_rejects_suspicious_zip_compression_ratio_before_writing(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    data_dir.mkdir()
    state_dir.mkdir()
    backup = tmp_path / "compression-bomb.zip"
    with zipfile.ZipFile(backup, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", "{}")
        zf.writestr("memory/bomb.md", b"0" * (8 * 1024 * 1024))

    result = CliRunner().invoke(
        restore,
        [str(backup), "--yes"],
        env=_restore_env(data_dir, state_dir),
    )

    assert result.exit_code != 0
    assert "suspicious compression ratio" in result.output
    assert not (data_dir / "bomb.md").exists()


def test_restore_rejects_symlinked_destination_component(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    outside = tmp_path / "outside"
    data_dir.mkdir()
    state_dir.mkdir()
    outside.mkdir()
    (data_dir / "project").symlink_to(outside, target_is_directory=True)
    backup = tmp_path / "symlink-destination.zip"
    with zipfile.ZipFile(backup, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", "{}")
        zf.writestr("memory/project/example.md", "ATTACKER")

    result = CliRunner().invoke(
        restore,
        [str(backup), "--yes"],
        env=_restore_env(data_dir, state_dir),
    )

    assert result.exit_code != 0
    assert "symlinked restore destination" in result.output
    assert not (outside / "example.md").exists()


def test_restore_rejects_non_object_manifest_before_writing(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    data_dir.mkdir()
    state_dir.mkdir()
    backup = tmp_path / "bad-manifest.zip"
    with zipfile.ZipFile(backup, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", "[1]")
        zf.writestr("memory/example.md", "must not be written")

    result = CliRunner().invoke(
        restore,
        [str(backup), "--yes"],
        env=_restore_env(data_dir, state_dir),
    )

    assert result.exit_code != 0
    assert "manifest must be a JSON object" in result.output
    assert not (data_dir / "example.md").exists()


def test_restore_rejects_invalid_sqlite_without_replacing_existing_db(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    data_dir.mkdir()
    state_dir.mkdir()
    existing = state_dir / "memvec.db"
    with closing(sqlite3.connect(existing)) as connection:
        connection.execute("CREATE TABLE preserved (value TEXT)")
        connection.execute("INSERT INTO preserved VALUES ('safe')")
        connection.commit()

    backup = tmp_path / "invalid-db.zip"
    with zipfile.ZipFile(backup, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", "{}")
        zf.writestr("state/memvec.db", b"not sqlite")

    result = CliRunner().invoke(
        restore,
        [str(backup), "--yes"],
        env=_restore_env(data_dir, state_dir),
    )

    assert result.exit_code != 0
    assert "invalid SQLite database" in result.output
    with closing(sqlite3.connect(existing)) as connection:
        assert connection.execute("SELECT value FROM preserved").fetchone() == ("safe",)


def test_cli_type_choices_track_the_durable_registry() -> None:
    """The CLI must not drift from `tiers.DURABLE_TYPES`.

    The list was hardcoded in three places and silently lacked `procedure` and
    `failure_pattern` — `memo save --type failure_pattern` was rejected for a
    type the store accepts and CLAUDE.md documents.
    """
    from memo.cli_memory import _USER_SETTABLE_TYPES
    from memo.tiers import DURABLE_TYPES

    assert set(_USER_SETTABLE_TYPES) == DURABLE_TYPES - {"synthesis"}
    assert "failure_pattern" in _USER_SETTABLE_TYPES
    assert "procedure" in _USER_SETTABLE_TYPES
