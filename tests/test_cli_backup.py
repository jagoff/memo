from __future__ import annotations

import json
import sqlite3
import stat
import zipfile
from contextlib import closing
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from memo import cli_backup
from memo.cli_memory import restore
from memo.config import Config


def _create_wal_database(path: Path) -> tuple[sqlite3.Connection, bytes]:
    marker = b"MEMO_SECRET_BACKUP_MARKER_7f21" * 16
    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    connection.execute("PRAGMA wal_autocheckpoint=0")
    connection.executescript(
        "CREATE TABLE visible (value TEXT NOT NULL);"
        "CREATE TABLE secret_store (encrypted_blob BLOB NOT NULL);"
        "CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "CREATE TABLE meta (id TEXT PRIMARY KEY);"
    )
    connection.execute("INSERT INTO secret_store VALUES (?)", (marker,))
    connection.commit()
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.execute("INSERT INTO visible VALUES ('committed-in-wal')")
    connection.commit()
    assert path.with_name(f"{path.name}-wal").stat().st_size > 0
    return connection, marker


def _assert_sanitized_snapshot(path: Path, marker: bytes) -> None:
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert connection.execute("SELECT COUNT(*) FROM secret_store").fetchone() == (0,)
        assert connection.execute("SELECT value FROM visible").fetchall() == [("committed-in-wal",)]
    assert marker not in path.read_bytes()


def test_portable_backup_snapshots_wal_sanitizes_secrets_and_restores(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_cfg = Config(
        data_dir=tmp_path / "source-data",
        state_dir=tmp_path / "source-state",
        reranker_enabled=False,
    )
    source_cfg.ensure_dirs()
    writer, marker = _create_wal_database(source_cfg.db_path)
    archive = tmp_path / "portable.zip"
    monkeypatch.setattr(
        cli_backup.Config,
        "from_env",
        classmethod(lambda _cls: source_cfg),
    )

    try:
        cli_backup._portable_backup(str(archive))
    finally:
        writer.close()

    with zipfile.ZipFile(archive) as zipped:
        snapshot_bytes = zipped.read(f"state/{source_cfg.db_path.name}")
    extracted = tmp_path / "extracted.db"
    extracted.write_bytes(snapshot_bytes)
    _assert_sanitized_snapshot(extracted, marker)

    restored_cfg = Config(
        data_dir=tmp_path / "restored-data",
        state_dir=tmp_path / "restored-state",
        reranker_enabled=False,
    )
    monkeypatch.setattr(
        cli_backup.Config,
        "from_env",
        classmethod(lambda _cls: restored_cfg),
    )

    result = CliRunner().invoke(restore, [str(archive), "--yes"])

    assert result.exit_code == 0, result.output
    _assert_sanitized_snapshot(restored_cfg.state_dir / source_cfg.db_path.name, marker)


def test_portable_backup_skips_markdown_symlink_outside_memory_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg = Config(
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        reranker_enabled=False,
    )
    cfg.ensure_dirs()
    safe = cfg.memory_dir / "safe.md"
    safe.write_text("safe memory", encoding="utf-8")
    outside = tmp_path / "outside-secret.md"
    outside.write_text("BACKUP_SYMLINK_EXFILTRATION_MARKER", encoding="utf-8")
    (cfg.memory_dir / "leak.md").symlink_to(outside)
    archive = tmp_path / "portable.zip"
    monkeypatch.setattr(
        cli_backup.Config,
        "from_env",
        classmethod(lambda _cls: cfg),
    )

    cli_backup._portable_backup(str(archive))

    with zipfile.ZipFile(archive) as zipped:
        assert "memory/safe.md" in zipped.namelist()
        assert "memory/leak.md" not in zipped.namelist()
        assert json.loads(zipped.read("manifest.json"))["n_md"] == 1
    assert b"BACKUP_SYMLINK_EXFILTRATION_MARKER" not in archive.read_bytes()


def test_portable_backup_rejects_symlinked_state_database(monkeypatch, tmp_path: Path) -> None:
    cfg = Config(
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        reranker_enabled=False,
    )
    cfg.ensure_dirs()
    outside = tmp_path / "outside.db"
    with closing(sqlite3.connect(outside)) as connection:
        connection.execute("CREATE TABLE external_secret(marker TEXT)")
        connection.execute("INSERT INTO external_secret VALUES ('DB_EXFIL_MARKER')")
        connection.commit()
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.db_path.symlink_to(outside)
    monkeypatch.setattr(
        cli_backup.Config,
        "from_env",
        classmethod(lambda cls: cfg),
    )

    archive = tmp_path / "unsafe.zip"
    with pytest.raises(click.ClickException, match="symlinked state database"):
        cli_backup._portable_backup(str(archive))

    assert not archive.exists()


def test_portable_backup_archive_is_private(monkeypatch, tmp_path: Path) -> None:
    cfg = Config(
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        reranker_enabled=False,
    )
    cfg.ensure_dirs()
    (cfg.memory_dir / "private.md").write_text("private memory", encoding="utf-8")
    archive = tmp_path / "portable.zip"
    monkeypatch.setattr(
        cli_backup.Config,
        "from_env",
        classmethod(lambda _cls: cfg),
    )

    cli_backup._portable_backup(str(archive))

    assert stat.S_IMODE(archive.stat().st_mode) == 0o600


def test_portable_backup_removes_partial_archive_on_failure(monkeypatch, tmp_path: Path) -> None:
    cfg = Config(
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        reranker_enabled=False,
    )
    cfg.ensure_dirs()
    with closing(sqlite3.connect(cfg.db_path)) as connection:
        connection.execute("CREATE TABLE example (value TEXT)")
        connection.commit()
    archive = tmp_path / "partial.zip"
    monkeypatch.setattr(
        cli_backup.Config,
        "from_env",
        classmethod(lambda _cls: cfg),
    )
    monkeypatch.setattr(
        "memo.sqlite_snapshot.snapshot_sqlite_database",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("snapshot failed")),
    )

    with pytest.raises(OSError, match="snapshot failed"):
        cli_backup._portable_backup(str(archive))

    assert not archive.exists()
