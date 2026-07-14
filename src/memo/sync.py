"""Production-grade sync via audit log replay.

Enables robust, conflict-free synchronization between multiple Macs by
replaying missing events from a remote `history.db`.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memo.history import HistoryStore

_log = logging.getLogger(__name__)


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

    def _backup_path(self, name: str) -> Path:
        """Resolve a validated name and prove it remains below backup_dir."""
        safe_name = self._validate_backup_name(name)
        root = self.backup_dir.resolve()
        path = (root / safe_name).resolve()
        if path.parent != root:
            raise ValueError("Invalid backup name: path escapes backup directory")
        return path

    def create_backup(
        self,
        compress: bool = True,
        name: str | None = None,
    ) -> BackupMetadata:
        fh = self._lock_file.open("w")  # type: ignore[call-overload]
        try:
            self._acquire_lock(fh)
        except OSError as exc:
            fh.close()
            raise RuntimeError(f"Could not acquire backup lock: {exc}") from exc
        try:
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
            self._validate_backup_name(name)
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

                with tarfile.open(archive_path, "x:gz") as tar:
                    tar.add(scratch, arcname=backup_name)
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
        memory_files = list(self.memory_dir.rglob("*.md"))
        for f in memory_files:
            # Preserve the per-project bucket layout (memory_dir/<project>/...)
            # so restore recreates it and same-named files in different buckets
            # don't collide. rglob handles both flat and bucketed layouts.
            dest = memory_backup / f.relative_to(self.memory_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)

        db_backup = backup_path / "db"
        db_backup.mkdir(exist_ok=True)
        for db_file in self.db_dir.glob("*.db"):
            shutil.copy2(db_file, db_backup / db_file.name)

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
            if f.is_file():
                sha.update(f.read_bytes())
        return sha.hexdigest()

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
                for member in tar.getmembers():
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
        archive_name = backup_name
        if backup_name.endswith(".tar.gz"):
            base_name = backup_name.removesuffix(".tar.gz")
        else:
            base_name = backup_name
            archive_name = f"{backup_name}.tar.gz"
        self._validate_backup_name(base_name)
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
                for member in tar.getmembers():
                    target = (tmp_root / member.name).resolve()
                    if not target.is_relative_to(tmp_root):
                        raise ValueError(f"Unsafe path in backup archive: {member.name}")
                tar.extractall(tmp_root, filter="data")

            extracted_path = tmp_root / archive_path.name.removesuffix(".tar.gz")
            if not extracted_path.is_dir():
                candidates = [p for p in tmp_root.iterdir() if p.is_dir()]
                if not candidates:
                    return False
                extracted_path = candidates[0]

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
        if restore_memories:
            memory_backup = backup_path / "memories"
            if not memory_backup.is_dir():
                memory_backup = backup_path / "memorias"  # legacy backup layout
            if memory_backup.is_dir():
                for f in memory_backup.rglob("*.md"):
                    dest = self.memory_dir / f.relative_to(memory_backup)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest)

        if restore_dbs:
            db_backup = backup_path / "db"
            if db_backup.is_dir():
                for db_file in db_backup.glob("*.db"):
                    shutil.copy2(db_file, self.db_dir / db_file.name)

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
