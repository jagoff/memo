"""WAL-consistent, secret-sanitized SQLite snapshots for backups."""

from __future__ import annotations

import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path


def _backup_database(source: Path, destination: Path) -> None:
    if source.is_symlink():
        raise ValueError(f"refusing symlinked SQLite source: {source}")
    source_uri = f"{source.resolve(strict=True).as_uri()}?mode=ro"
    with (
        closing(sqlite3.connect(source_uri, uri=True)) as source_connection,
        closing(sqlite3.connect(destination)) as destination_connection,
    ):
        source_connection.backup(destination_connection)


def _sanitize_secret_store(database: Path) -> None:
    with closing(sqlite3.connect(database)) as connection:
        # Keep rollback material out of a persistent WAL/journal while the
        # sensitive table is purged from this private scratch snapshot.
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA secure_delete=ON")
        has_secret_store = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='secret_store'"
        ).fetchone()
        if has_secret_store:
            connection.execute("DELETE FROM secret_store")
            connection.commit()
            # Rebuild every page so deleted ciphertext/name cells and freelist
            # pages cannot survive into the published snapshot.
            connection.execute("VACUUM")
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise sqlite3.DatabaseError(f"SQLite snapshot failed integrity_check: {database}")


def snapshot_sqlite_database(source: Path, destination: Path) -> None:
    """Publish a consistent SQLite snapshot with an empty ``secret_store``.

    The first backup API call includes committed WAL frames without copying a
    live ``.db`` file. Secrets are securely deleted and vacuumed only in a
    private scratch database. The second backup call publishes that sanitized
    logical database, so the destination never receives the original secret
    pages and remains directly restorable as a normal SQLite file.
    """
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"SQLite snapshot destination already exists: {destination}")

    try:
        with tempfile.TemporaryDirectory(
            prefix=".sqlite-snapshot-",
            dir=destination.parent,
        ) as scratch_dir:
            sanitized = Path(scratch_dir) / "sanitized.db"
            _backup_database(source, sanitized)
            _sanitize_secret_store(sanitized)
            _backup_database(sanitized, destination)
    except Exception:
        destination.unlink(missing_ok=True)
        destination.with_name(f"{destination.name}-wal").unlink(missing_ok=True)
        destination.with_name(f"{destination.name}-shm").unlink(missing_ok=True)
        raise

    with closing(sqlite3.connect(destination)) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            destination.unlink(missing_ok=True)
            raise sqlite3.DatabaseError(f"SQLite snapshot failed integrity_check: {destination}")


__all__ = ["snapshot_sqlite_database"]
