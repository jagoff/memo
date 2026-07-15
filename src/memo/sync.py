"""Production-grade sync via audit log replay.

Enables robust, conflict-free synchronization between multiple Macs by
replaying missing events from a remote `history.db`.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import hmac
import json
import logging
import os
import shutil
import sqlite3
import stat
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from memo.atomic_io import authority_write_lock
from memo.history import HistoryStore
from memo.sqlite_snapshot import snapshot_sqlite_database

_log = logging.getLogger(__name__)

_ARCHIVE_MAX_MEMBERS = 20_000
_ARCHIVE_MAX_METADATA_BYTES = 1024 * 1024
_ARCHIVE_MAX_MEMORY_BYTES = 64 * 1024 * 1024
_ARCHIVE_MAX_DB_BYTES = 8 * 1024 * 1024 * 1024
_ARCHIVE_MAX_TOTAL_BYTES = 12 * 1024 * 1024 * 1024
_ARCHIVE_RATIO_MIN_BYTES = 1024 * 1024
_ARCHIVE_MAX_COMPRESSION_RATIO = 100


@dataclass
class SyncDiff:
    """Result of computing sync diff between vaults."""

    applied: int
    conflicts: int
    errors: int


@dataclass
class BackupMetadata:
    """Metadata for a backup archive."""

    timestamp: str
    memory_count: int
    checksum: str
    compressed_size: int
    original_size: int
    name: str = ""


class BackupManager:
    """Manages backup creation and restoration."""

    def __init__(self, memory_dir: Path, db_dir: Path, backup_dir: Path) -> None:
        self.memory_dir = memory_dir
        self.db_dir = db_dir
        self.backup_dir = backup_dir
        self._lock_file = backup_dir / ".backup.lock"
        self._lock_file.parent.mkdir(parents=True, exist_ok=True)

    def _acquire_lock(self, fh):
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)

    def _release_lock(self, fh):
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def _open_lock_file(self):
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self._lock_file, flags, 0o600)
        except OSError as exc:
            raise RuntimeError(f"Could not open backup lock safely: {exc}") from exc

        try:
            descriptor_stat = os.fstat(descriptor)
            path_stat = os.stat(self._lock_file, follow_symlinks=False)
            if (
                not stat.S_ISREG(descriptor_stat.st_mode)
                or not stat.S_ISREG(path_stat.st_mode)
                or (descriptor_stat.st_dev, descriptor_stat.st_ino)
                != (path_stat.st_dev, path_stat.st_ino)
            ):
                raise RuntimeError("Could not open backup lock safely: unsafe lock file")
            os.fchmod(descriptor, 0o600)
            return os.fdopen(descriptor, "r+b")
        except Exception:
            os.close(descriptor)
            raise

    @staticmethod
    def _validate_backup_name(name: str) -> str:
        """Return a safe single-component backup name or reject it."""
        if (
            not name
            or len(name) > 128
            or name in {".", ".."}
            or name != name.strip()
            or "/" in name
            or "\\" in name
            or "\x00" in name
            or Path(name).is_absolute()
        ):
            raise ValueError("Invalid backup name: use one filename component")
        return name

    @classmethod
    def _logical_backup_name(cls, name: str) -> str:
        """Normalize an optional archive suffix to the logical backup name."""
        return cls._validate_backup_name(name.removesuffix(".tar.gz"))

    def _backup_path(self, name: str) -> Path:
        """Resolve a validated name and prove it remains below backup_dir."""
        safe_name = self._validate_backup_name(name)
        root = self.backup_dir.resolve()
        path = (root / safe_name).resolve()
        if path.parent != root:
            raise ValueError("Invalid backup name: path escapes backup directory")
        return path

    @staticmethod
    def _safe_regular_files(root: Path, pattern: str) -> list[Path]:
        """Return regular in-root files without following any symlink component."""
        root_resolved = root.resolve()
        safe: list[Path] = []
        for candidate in sorted(root.rglob(pattern)):
            relative = candidate.relative_to(root)
            cursor = root
            if any((cursor := cursor / part).is_symlink() for part in relative.parts):
                continue
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root_resolved)
            except (OSError, ValueError):
                continue
            if resolved.is_file():
                safe.append(candidate)
        return safe

    @staticmethod
    def _safe_restore_destination(root: Path, relative: Path) -> Path:
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe restore path: {relative}")
        cursor = root
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                raise ValueError(f"symlinked restore destination: {cursor}")
        destination = root / relative
        if not destination.resolve(strict=False).is_relative_to(root.resolve()):
            raise ValueError(f"Unsafe restore path: {relative}")
        return destination

    @staticmethod
    def _validate_sqlite_backup(path: Path) -> None:
        try:
            uri = f"{path.resolve().as_uri()}?mode=ro"
            with contextlib.closing(sqlite3.connect(uri, uri=True)) as connection:
                result = connection.execute("PRAGMA integrity_check").fetchone()
            if result != ("ok",):
                raise sqlite3.DatabaseError(str(result[0]) if result else "no result")
        except sqlite3.DatabaseError as exc:
            raise ValueError(f"Invalid SQLite backup {path.name}: {exc}") from exc

    @staticmethod
    def _copy_sqlite_database(source: Path, destination: Path) -> None:
        """Copy a logical SQLite snapshot without replacing the destination inode."""
        if source.is_symlink() or destination.is_symlink():
            raise ValueError("refusing symlinked SQLite restore path")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_uri = f"{source.resolve(strict=True).as_uri()}?mode=ro"
        with (
            contextlib.closing(sqlite3.connect(source_uri, uri=True)) as source_connection,
            contextlib.closing(
                sqlite3.connect(destination, timeout=10.0)
            ) as destination_connection,
        ):
            destination_connection.execute("PRAGMA busy_timeout = 10000")
            source_connection.backup(destination_connection)
            result = destination_connection.execute("PRAGMA integrity_check").fetchone()
            if result != ("ok",):
                raise sqlite3.DatabaseError(f"SQLite restore failed integrity_check: {destination}")
        os.chmod(destination, 0o600)

    def _restore_sqlite_in_place(self, source: Path, destination: Path) -> None:
        self._copy_sqlite_database(source, destination)

    @staticmethod
    def _atomic_copy(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".restore-tmp",
        )
        temporary = Path(temporary_name)
        try:
            with source.open("rb") as input_file, os.fdopen(descriptor, "wb") as output_file:
                descriptor = -1
                shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
                output_file.flush()
                os.fsync(output_file.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, destination)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _move_aside(path: Path) -> Path | None:
        if path.is_symlink():
            raise ValueError(f"symlinked restore destination: {path}")
        if not path.exists():
            return None
        descriptor, rollback_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".restore-rollback",
        )
        os.close(descriptor)
        rollback = Path(rollback_name)
        rollback.unlink()
        os.replace(path, rollback)
        return rollback

    def create_backup(
        self,
        compress: bool = True,
        name: str | None = None,
    ) -> BackupMetadata:
        fh = self._open_lock_file()
        try:
            self._acquire_lock(fh)
        except OSError as exc:
            fh.close()
            raise RuntimeError(f"Could not acquire backup lock: {exc}") from exc
        try:
            # Share the Markdown authority lock with CRUD/reindex so files and
            # logical SQLite snapshots represent one coherent checkpoint.
            with authority_write_lock(self.memory_dir):
                return self._create_backup_inner(compress, name)
        finally:
            self._release_lock(fh)
            fh.close()

    def _create_backup_inner(
        self,
        compress: bool = True,
        name: str | None = None,
    ) -> BackupMetadata:
        timestamp = datetime.now(UTC).isoformat()
        backup_name = (
            self._logical_backup_name(name)
            if name is not None
            else f"backup_{timestamp.replace(':', '-')}"
        )
        backup_path = self._backup_path(backup_name)
        archive_path = self._backup_path(f"{backup_name}.tar.gz")
        for destination in (backup_path, archive_path):
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Backup already exists: {destination.name}")

        scratch = Path(tempfile.mkdtemp(prefix=".backup-", dir=self.backup_dir))
        try:
            metadata = self._populate_backup(scratch, timestamp, backup_name)

            if compress:
                import tarfile

                descriptor = os.open(
                    archive_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
                try:
                    with os.fdopen(descriptor, "wb") as archive_file:
                        descriptor = -1
                        with tarfile.open(fileobj=archive_file, mode="w:gz") as tar:
                            tar.add(scratch, arcname=backup_name)
                except Exception:
                    archive_path.unlink(missing_ok=True)
                    raise
                finally:
                    if descriptor >= 0:
                        os.close(descriptor)
                metadata.compressed_size = archive_path.stat().st_size
            else:
                scratch.rename(backup_path)
            return metadata
        finally:
            if scratch.exists():
                shutil.rmtree(scratch)

    def _populate_backup(
        self,
        backup_path: Path,
        timestamp: str,
        backup_name: str,
    ) -> BackupMetadata:
        """Populate a private scratch directory and return its metadata."""

        memory_backup = backup_path / "memories"
        memory_backup.mkdir()
        memory_files = [
            path
            for path in self._safe_regular_files(self.memory_dir, "*.md")
            if path.relative_to(self.memory_dir).parts[:1] != ("secrets",)
        ]
        for f in memory_files:
            # Preserve the per-project bucket layout (memory_dir/<project>/...)
            # so restore recreates it and same-named files in different buckets
            # don't collide. rglob handles both flat and bucketed layouts.
            dest = memory_backup / f.relative_to(self.memory_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)

        db_backup = backup_path / "db"
        db_backup.mkdir(exist_ok=True)
        for db_file in self._safe_regular_files(self.db_dir, "*.db"):
            if db_file.parent != self.db_dir:
                continue
            snapshot_sqlite_database(db_file, db_backup / db_file.name)

        checksum = self._compute_checksum(backup_path)
        memory_count = len(memory_files)
        original_size = sum(f.stat().st_size for f in backup_path.rglob("*"))
        metadata = BackupMetadata(
            timestamp=timestamp,
            memory_count=memory_count,
            checksum=checksum,
            compressed_size=original_size,
            original_size=original_size,
            name=backup_name,
        )
        (backup_path / "metadata.json").write_text(
            json.dumps(metadata.__dict__, indent=2),
            encoding="utf-8",
        )

        return metadata

    def _compute_checksum(self, path: Path) -> str:
        sha = hashlib.sha256()
        for f in sorted(path.rglob("*")):
            if f.name == "metadata.json":
                continue
            if f.is_symlink():
                raise ValueError(f"Backup checksum refuses symlink: {f}")
            if f.is_file():
                relative = f.relative_to(path).as_posix().encode("utf-8")
                sha.update(len(relative).to_bytes(8, "big"))
                sha.update(relative)
                sha.update(f.stat().st_size.to_bytes(8, "big"))
                with f.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        sha.update(chunk)
        return sha.hexdigest()

    @staticmethod
    def _validated_archive_members(
        tar: Any,
        *,
        expected_root: str,
        compressed_size: int,
    ) -> list[Any]:
        """Preflight a named tar backup before extracting a single byte."""
        members: list[Any] = []
        seen: set[str] = set()
        total_size = 0
        allowed_trees = {"memories", "memorias", "db"}

        for member in tar:
            if len(members) >= _ARCHIVE_MAX_MEMBERS:
                raise ValueError("Backup archive contains too many members")
            name = member.name
            if not name or "\\" in name or "\x00" in name:
                raise ValueError(f"Unsafe path in backup archive: {name!r}")
            path = PurePosixPath(name)
            canonical = path.as_posix()
            if (
                path.is_absolute()
                or ".." in path.parts
                or canonical != name.rstrip("/")
                or not path.parts
                or path.parts[0] != expected_root
            ):
                raise ValueError(f"Unsafe path in backup archive: {name!r}")
            if canonical in seen:
                raise ValueError(f"Duplicate backup archive member: {name!r}")
            seen.add(canonical)

            relative_parts = path.parts[1:]
            if not (member.isdir() or member.isreg()):
                raise ValueError(f"Unsupported archive member type: {name!r}")
            if member.isdir():
                if relative_parts and relative_parts[0] not in allowed_trees:
                    raise ValueError(f"Unexpected backup archive member: {name!r}")
                members.append(member)
                continue

            if member.size < 0:
                raise ValueError(f"Invalid backup archive member size: {name!r}")
            if relative_parts == ("metadata.json",):
                member_limit = _ARCHIVE_MAX_METADATA_BYTES
            elif (
                len(relative_parts) >= 2
                and relative_parts[0] in {"memories", "memorias"}
                and path.suffix == ".md"
            ):
                member_limit = _ARCHIVE_MAX_MEMORY_BYTES
            elif len(relative_parts) == 2 and relative_parts[0] == "db" and path.suffix == ".db":
                member_limit = _ARCHIVE_MAX_DB_BYTES
            else:
                raise ValueError(f"Unexpected backup archive member: {name!r}")
            if member.size > member_limit:
                raise ValueError(f"Backup archive member is too large: {name!r}")
            total_size += member.size
            if total_size > _ARCHIVE_MAX_TOTAL_BYTES:
                raise ValueError("Backup archive is too large")
            members.append(member)

        if (
            total_size >= _ARCHIVE_RATIO_MIN_BYTES
            and total_size / max(compressed_size, 1) > _ARCHIVE_MAX_COMPRESSION_RATIO
        ):
            raise ValueError("Backup archive has a suspicious compression ratio")
        return members

    def list_backups(self) -> list[BackupMetadata]:
        backups = []
        for archive in self.backup_dir.glob("*.tar.gz"):
            backups.append(self._read_archive_metadata(archive))
        for p in self.backup_dir.iterdir():
            if p.is_dir() and (p / "metadata.json").is_file():
                backups.append(self._read_directory_metadata(p))
        return sorted(backups, key=lambda b: b.timestamp, reverse=True)

    def _read_directory_metadata(self, backup_path: Path) -> BackupMetadata:
        data = json.loads((backup_path / "metadata.json").read_text(encoding="utf-8"))
        meta = BackupMetadata(**data)
        meta.name = backup_path.name
        return meta

    def _read_archive_metadata(self, archive: Path) -> BackupMetadata:
        import tarfile

        try:
            with tarfile.open(archive, "r:gz") as tar:
                members = self._validated_archive_members(
                    tar,
                    expected_root=archive.name.removesuffix(".tar.gz"),
                    compressed_size=archive.stat().st_size,
                )
                for member in members:
                    if member.name.endswith("/metadata.json"):
                        extracted = tar.extractfile(member)
                        if extracted:
                            data = json.loads(extracted.read().decode("utf-8"))
                            data["compressed_size"] = archive.stat().st_size
                            meta = BackupMetadata(**data)
                            meta.name = archive.name.removesuffix(".tar.gz")
                            return meta
        except Exception:  # noqa: S110
            pass
        meta = BackupMetadata(datetime.now(UTC).isoformat(), 0, "", archive.stat().st_size, 0)
        meta.name = archive.name.removesuffix(".tar.gz")
        return meta

    def restore_backup(
        self,
        backup_name: str,
        restore_memories: bool = True,
        restore_dbs: bool = True,
    ) -> bool:
        """Restore memory files and/or databases from a backup."""
        base_name = self._logical_backup_name(backup_name)
        archive_name = f"{base_name}.tar.gz"
        archive_path = self._backup_path(archive_name)
        directory_path = self._backup_path(base_name)
        if not archive_path.is_file() and not directory_path.is_dir():
            return False

        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        if directory_path.is_dir():
            return self._restore_from_directory(
                directory_path,
                restore_memories=restore_memories,
                restore_dbs=restore_dbs,
            )

        import tarfile
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir).resolve()
            with tarfile.open(archive_path, "r:gz") as tar:
                members = self._validated_archive_members(
                    tar,
                    expected_root=base_name,
                    compressed_size=archive_path.stat().st_size,
                )
                tar.extractall(tmp_root, members=members, filter="data")

            extracted_path = tmp_root / base_name
            if not extracted_path.is_dir():
                return False

            return self._restore_from_directory(
                extracted_path,
                restore_memories=restore_memories,
                restore_dbs=restore_dbs,
            )

    def _restore_from_directory(
        self,
        backup_path: Path,
        *,
        restore_memories: bool,
        restore_dbs: bool,
    ) -> bool:
        with tempfile.TemporaryDirectory(prefix="memo-backup-restore-") as tmpdir:
            staged_path = Path(tmpdir) / "backup"
            shutil.copytree(backup_path, staged_path, symlinks=True)
            return self._restore_from_staged_directory(
                staged_path,
                restore_memories=restore_memories,
                restore_dbs=restore_dbs,
            )

    def _restore_from_staged_directory(
        self,
        backup_path: Path,
        *,
        restore_memories: bool,
        restore_dbs: bool,
    ) -> bool:
        metadata_path = backup_path / "metadata.json"
        if not metadata_path.is_file() or metadata_path.is_symlink():
            raise ValueError("Backup metadata is missing or unsafe")
        try:
            metadata = BackupMetadata(**json.loads(metadata_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"Backup metadata is invalid: {exc}") from exc
        restore_plan: list[tuple[Path, Path, bool]] = []
        if restore_memories:
            memory_backup = backup_path / "memories"
            if not memory_backup.is_dir():
                memory_backup = backup_path / "memorias"  # legacy backup layout
            if memory_backup.is_dir():
                candidates = list(memory_backup.rglob("*.md"))
                safe_files = self._safe_regular_files(memory_backup, "*.md")
                if len(candidates) != len(safe_files):
                    raise ValueError("Backup contains an unsafe markdown symlink or path")
                for source in safe_files:
                    relative = source.relative_to(memory_backup)
                    destination = self._safe_restore_destination(self.memory_dir, relative)
                    restore_plan.append((source, destination, False))

        if restore_dbs:
            db_backup = backup_path / "db"
            if db_backup.is_dir():
                candidates = list(db_backup.glob("*.db"))
                safe_files = [
                    path
                    for path in self._safe_regular_files(db_backup, "*.db")
                    if path.parent == db_backup
                ]
                if len(candidates) != len(safe_files):
                    raise ValueError("Backup contains an unsafe SQLite symlink or path")
                for source in safe_files:
                    self._validate_sqlite_backup(source)
                    destination = self._safe_restore_destination(self.db_dir, Path(source.name))
                    restore_plan.append((source, destination, True))

        checksum = self._compute_checksum(backup_path)
        if not metadata.checksum or not hmac.compare_digest(checksum, metadata.checksum):
            raise ValueError("Backup checksum mismatch; refusing restore")

        # Publish only after the staged copy has passed path, DB and checksum
        # validation. The authority lock coordinates with Memory writers.
        # Markdown files use sibling rollback files; SQLite databases use the
        # backup API in both directions so live connections keep their inode.
        file_plan = [entry for entry in restore_plan if not entry[2]]
        database_plan = [entry for entry in restore_plan if entry[2]]
        with authority_write_lock(self.memory_dir):  # noqa: SIM117 - lock outlives temp rollback
            with tempfile.TemporaryDirectory(prefix="memo-sqlite-rollback-") as tmpdir:
                rollback_root = Path(tmpdir)
                database_journal: list[tuple[Path, Path | None]] = []
                for index, (_source, destination, _is_database) in enumerate(database_plan):
                    relative = destination.relative_to(self.db_dir)
                    destination = self._safe_restore_destination(self.db_dir, relative)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination = self._safe_restore_destination(self.db_dir, relative)
                    rollback = None
                    if destination.exists():
                        rollback = rollback_root / f"{index}-{destination.name}"
                        self._copy_sqlite_database(destination, rollback)
                    database_journal.append((destination, rollback))

                file_journal: list[tuple[Path, Path | None]] = []
                attempted_databases: list[tuple[Path, Path | None]] = []
                try:
                    for source, destination, _is_database in file_plan:
                        relative = destination.relative_to(self.memory_dir)
                        destination = self._safe_restore_destination(self.memory_dir, relative)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination = self._safe_restore_destination(self.memory_dir, relative)
                        rollback = self._move_aside(destination)
                        file_journal.append((destination, rollback))
                        self._atomic_copy(source, destination)

                    for (source, _destination, _is_database), (
                        destination,
                        rollback,
                    ) in zip(database_plan, database_journal, strict=True):
                        attempted_databases.append((destination, rollback))
                        self._restore_sqlite_in_place(source, destination)
                except Exception as restore_error:
                    rollback_error: Exception | None = None
                    for destination, rollback in reversed(attempted_databases):
                        try:
                            if rollback is not None:
                                self._restore_sqlite_in_place(rollback, destination)
                            else:
                                destination.unlink(missing_ok=True)
                                for suffix in ("-wal", "-shm", "-journal"):
                                    destination.with_name(f"{destination.name}{suffix}").unlink(
                                        missing_ok=True
                                    )
                        except Exception as exc:  # pragma: no cover - catastrophic I/O
                            rollback_error = rollback_error or exc
                    for destination, rollback in reversed(file_journal):
                        destination.unlink(missing_ok=True)
                        if rollback is not None and rollback.exists():
                            os.replace(rollback, destination)
                    if rollback_error is not None:
                        raise RuntimeError(
                            f"Restore failed and SQLite rollback also failed: {rollback_error}"
                        ) from restore_error
                    raise
                else:
                    for _destination, rollback in file_journal:
                        if rollback is not None:
                            rollback.unlink(missing_ok=True)

        return True


class SyncManager:
    """Manages incremental sync via audit log replay."""

    def __init__(self, mem: Any) -> None:
        self.mem = mem

    def sync_from_remote(self, remote_history_db: Path) -> SyncDiff:
        """Apply missing events from a remote history database."""
        if not remote_history_db.is_file():
            _log.warning("sync: remote history.db not found at %s", remote_history_db)
            return SyncDiff(0, 0, 1)

        # 1. Open remote history
        remote_store = None
        try:
            remote_store = HistoryStore(remote_history_db)
            remote_device_id = remote_store.device_id
            if remote_device_id == "unknown":
                sample = remote_store.list_recent(limit=1)
                if sample:
                    remote_device_id = sample[0].get("device_id") or "unknown"
            if remote_device_id == "unknown":
                _log.info("sync: remote history has no attributable device events")
                return SyncDiff(0, 0, 0)
            if remote_device_id == self.mem.history.device_id:
                _log.info("sync: remote device id matches local, skipping loopback sync")
                return SyncDiff(0, 0, 0)

            last_lsn = self.mem.history.get_sync_state(remote_device_id)
            new_events = remote_store.list_recent(
                after_lsn=last_lsn, device_id=remote_device_id, limit=1000
            )
            new_events.reverse()

            applied = 0
            conflicts = 0
            errors = 0
            reindexed = False

            for ev in new_events:
                op = ev["op"]
                record_id = ev["record_id"]
                try:
                    if op == "delete":
                        self.mem.delete(record_id)
                        applied += 1
                        self.mem.history.update_sync_state(remote_device_id, ev["id"])
                        continue

                    if not reindexed:
                        self.mem.reindex()
                        reindexed = True

                    if self.mem.get(record_id) is not None:
                        applied += 1
                        self.mem.history.update_sync_state(remote_device_id, ev["id"])
                    else:
                        _log.info(
                            "sync: record %s not present locally yet; deferring to next pass",
                            record_id,
                        )
                        break
                except Exception as exc:
                    _log.error("sync: failed to apply event %s: %s", ev["id"], exc)
                    errors += 1
                    break

            return SyncDiff(applied, conflicts, errors)
        except Exception as exc:
            _log.error("sync: global failure: %s", exc)
            return SyncDiff(0, 0, 1)
        finally:
            if remote_store is not None:
                with contextlib.suppress(Exception):
                    remote_store.close()


__all__ = [
    "BackupManager",
    "BackupMetadata",
    "SyncDiff",
    "SyncManager",
]
