"""Production-grade sync via audit log replay.

Enables robust, conflict-free synchronization between multiple Macs by
replaying missing events from a remote `history.db`.
"""

from __future__ import annotations

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


class BackupManager:
    """Manages backup creation and restoration."""

    def __init__(self, memory_dir: Path, db_dir: Path, backup_dir: Path) -> None:
        self.memory_dir = memory_dir
        self.db_dir = db_dir
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(
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
        return BackupMetadata(**data)

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
                            return BackupMetadata(**data)
        except Exception:
            pass
        return BackupMetadata(datetime.now(UTC).isoformat(), 0, "", archive.stat().st_size, 0)


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
        try:
            remote_store = HistoryStore(remote_history_db)
            remote_device_id = remote_store.device_id
            if remote_device_id == self.mem.history.device_id:
                _log.info("sync: remote device id matches local, skipping loopback sync")
                return SyncDiff(0, 0, 0)

            # 2. Get last LSN seen from this device
            last_lsn = self.mem.history.get_sync_state(remote_device_id)

            # 3. Fetch missing events (limit 1000 per pass)
            new_events = remote_store.list_recent(
                after_lsn=last_lsn, device_id=remote_device_id, limit=1000
            )
            # list_recent returns DESC (newest first), we want ASC for replay
            new_events.reverse()

            applied = 0
            conflicts = 0
            errors = 0

            for ev in new_events:
                try:
                    success = self._apply_event(ev)
                    if success:
                        applied += 1
                        self.mem.history.update_sync_state(remote_device_id, ev["id"])
                    else:
                        conflicts += 1
                except Exception as exc:
                    _log.error("sync: failed to apply event %s: %s", ev["id"], exc)
                    errors += 1

            return SyncDiff(applied, conflicts, errors)
        except Exception as exc:
            _log.error("sync: global failure: %s", exc)
            return SyncDiff(0, 0, 1)

    def _apply_event(self, ev: dict[str, Any]) -> bool:
        """Apply a single history event to the local state."""
        op = ev["op"]
        record_id = ev["record_id"]

        # If it's a delete, just apply it
        if op == "delete":
            self.mem.delete(record_id)
            return True

        # For save/update, we need the actual file content.
        # We assume the file is already synced (e.g. via iCloud).
        # We resolve the record_id to a path and re-index it.
        try:
            # We can't use mem.get() because it only works if it's already in the store.
            # But for a remote 'save', it's NOT in our store yet.
            # However, the record_id IS the filename (id.md).
            path = self.mem.cfg.data_dir / f"{record_id}.md"
            if not path.is_file():
                _log.warning("sync: file for record %s not found at %s", record_id, path)
                return False

            # Read and index
            content = path.read_text(encoding="utf-8")
            # save() with auto_derive=False is fast and idempotent if title/tags match.
            # Using save() ensures the vector index is updated.
            self.mem.save(
                content=content,
                title=ev.get("title") or "",
                type_=ev.get("type") or "note",
                # Note: we don't have the tags in the history.db 'save' event normally,
                # but if we use reindex() logic it's safer.
            )
            return True
        except Exception as exc:
            _log.error("sync: apply_event failed for %s: %s", record_id, exc)
            return False


__all__ = [
    "BackupManager",
    "BackupMetadata",
    "SyncDiff",
    "SyncManager",
]
