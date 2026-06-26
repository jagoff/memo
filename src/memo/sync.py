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
    memoria_count: int
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
        backup_name = name or f"backup_{timestamp.replace(':', '-')}"
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)

        memoria_backup = backup_path / "memorias"
        memoria_backup.mkdir(exist_ok=True)
        memoria_files = list(self.memory_dir.glob("*.md"))
        for f in memoria_files:
            shutil.copy2(f, memoria_backup / f.name)

        db_backup = backup_path / "db"
        db_backup.mkdir(exist_ok=True)
        for db_file in self.db_dir.glob("*.db"):
            shutil.copy2(db_file, db_backup / db_file.name)

        checksum = self._compute_checksum(backup_path)
        memoria_count = len(memoria_files)
        original_size = sum(f.stat().st_size for f in backup_path.rglob("*"))
        metadata = BackupMetadata(
            timestamp=timestamp,
            memoria_count=memoria_count,
            checksum=checksum,
            compressed_size=original_size,
            original_size=original_size,
        )
        (backup_path / "metadata.json").write_text(
            json.dumps(metadata.__dict__, indent=2),
            encoding="utf-8",
        )

        if compress:
            import tarfile
            archive_path = self.backup_dir / f"{backup_name}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(backup_path, arcname=backup_name)
            shutil.rmtree(backup_path)
            compressed_size = archive_path.stat().st_size
        else:
            compressed_size = original_size

        metadata.compressed_size = compressed_size
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
                            meta.name = archive.stem
                            return meta
        except Exception:  # noqa: S110
            pass
        meta = BackupMetadata(datetime.now(UTC).isoformat(), 0, "", archive.stat().st_size, 0)
        meta.name = archive.name.replace(".tar.gz", "")
        return meta

    def restore_backup(
        self,
        backup_name: str,
        restore_memorias: bool = True,
        restore_dbs: bool = True,
    ) -> bool:
        """Restore memoria files and/or databases from a backup."""
        archive_path = self.backup_dir / (
            backup_name if backup_name.endswith(".tar.gz") else f"{backup_name}.tar.gz"
        )
        directory_path = self.backup_dir / backup_name
        if not archive_path.is_file() and not directory_path.is_dir():
            return False

        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        if directory_path.is_dir():
            return self._restore_from_directory(
                directory_path,
                restore_memorias=restore_memorias,
                restore_dbs=restore_dbs,
            )

        import tarfile
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir).resolve()
            with tarfile.open(archive_path, "r:gz") as tar:
                for member in tar.getmembers():
                    target = (tmp_root / member.name).resolve()
                    if not str(target).startswith(str(tmp_root)):
                        raise ValueError(f"Unsafe path in backup archive: {member.name}")
                tar.extractall(tmp_root)  # noqa: S202 — member paths validated in the loop above

            extracted_path = tmp_root / archive_path.name.removesuffix(".tar.gz")
            if not extracted_path.is_dir():
                candidates = [p for p in tmp_root.iterdir() if p.is_dir()]
                if not candidates:
                    return False
                extracted_path = candidates[0]

            return self._restore_from_directory(
                extracted_path,
                restore_memorias=restore_memorias,
                restore_dbs=restore_dbs,
            )

    def _restore_from_directory(
        self,
        backup_path: Path,
        *,
        restore_memorias: bool,
        restore_dbs: bool,
    ) -> bool:
        if restore_memorias:
            memoria_backup = backup_path / "memorias"
            if memoria_backup.is_dir():
                for f in memoria_backup.glob("*.md"):
                    shutil.copy2(f, self.memory_dir / f.name)

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
