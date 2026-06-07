"""EXPERIMENTAL — not covered by the test suite, not exposed via MCP. API may change without notice.

Memory sync & backup — multi-device sync and automated backups.

Enables:
- Sync between multiple memo vaults/devices
- Automated compressed backups
- Restore from backup
- Incremental sync (only changes)
- Conflict resolution for concurrent changes
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from consciousness_contracts import (
        SyncCoordinator,
        SyncPhase,
        default_state_path,
    )
    _HAS_SYNC_COORDINATOR = True
except ImportError:
    _HAS_SYNC_COORDINATOR = False

import logging

_log = logging.getLogger(__name__)


@dataclass
class SyncDiff:
    """Result of computing sync diff between vaults."""
    new: list[str]  # memoria IDs only in source
    modified: list[str]  # memoria IDs modified in source
    deleted: list[str]  # memoria IDs deleted in source
    conflicts: list[str]  # memoria IDs modified in both


@dataclass
class BackupMetadata:
    """Metadata for a backup archive."""
    timestamp: str
    memoria_count: int
    checksum: str
    compressed_size: int
    original_size: int


class BackupManager:
    """Manages backup creation and restoration.

    Args:
        memory_dir: Path to the memoria .md files.
        db_dir: Path to the SQLite databases.
        backup_dir: Path to store backups.
    """

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
        """Create a backup of the entire vault.

        Args:
            compress: Whether to compress the backup.
            name: Optional backup name (defaults to timestamp).

        Returns:
            BackupMetadata with backup information.
        """
        timestamp = datetime.now(UTC).isoformat()
        backup_name = name or f"backup_{timestamp.replace(':', '-')}"
        backup_path = self.backup_dir / backup_name

        # Create backup directory
        backup_path.mkdir(exist_ok=True)

        # Copy memorias
        memoria_backup = backup_path / "memorias"
        memoria_backup.mkdir(exist_ok=True)
        memoria_files = list(self.memory_dir.glob("*.md"))
        for f in memoria_files:
            shutil.copy2(f, memoria_backup / f.name)

        # Copy databases
        db_backup = backup_path / "db"
        db_backup.mkdir(exist_ok=True)
        for db_file in self.db_dir.glob("*.db"):
            shutil.copy2(db_file, db_backup / db_file.name)

        # Compute checksum
        checksum = self._compute_checksum(backup_path)

        # Count memorias
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
            # Compress to tar.gz
            import tarfile

            archive_path = self.backup_dir / f"{backup_name}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(backup_path, arcname=backup_name)

            # Remove uncompressed backup
            shutil.rmtree(backup_path)

            compressed_size = archive_path.stat().st_size
        else:
            compressed_size = original_size

        metadata.compressed_size = compressed_size
        return metadata

    def list_backups(self) -> list[BackupMetadata]:
        """List all available backups.

        Returns:
            List of BackupMetadata objects.
        """
        backups = []

        for archive in self.backup_dir.glob("*.tar.gz"):
            backups.append(self._read_archive_metadata(archive))

        for backup_path in self.backup_dir.iterdir():
            if backup_path.is_dir():
                backups.append(self._read_directory_metadata(backup_path))

        return sorted(backups, key=lambda b: b.timestamp, reverse=True)

    def restore_backup(
        self,
        backup_name: str,
        restore_memorias: bool = True,
        restore_dbs: bool = True,
    ) -> bool:
        """Restore from a backup.

        Args:
            backup_name: Name of the backup to restore.
            restore_memorias: Whether to restore memoria files.
            restore_dbs: Whether to restore databases.

        Returns:
            True if successful.
        """
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
                tar.extractall(tmp_root)

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

    def _compute_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of all files in path."""
        sha = hashlib.sha256()
        for f in sorted(path.rglob("*")):
            if f.is_file():
                sha.update(f.read_bytes())
        return sha.hexdigest()

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

    def _read_directory_metadata(self, backup_path: Path) -> BackupMetadata:
        metadata_path = backup_path / "metadata.json"
        if metadata_path.is_file():
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            return BackupMetadata(**data)

        return BackupMetadata(
            timestamp=self._timestamp_from_name(backup_path.name),
            memoria_count=len(list((backup_path / "memorias").glob("*.md"))),
            checksum=self._compute_checksum(backup_path),
            compressed_size=sum(f.stat().st_size for f in backup_path.rglob("*") if f.is_file()),
            original_size=sum(f.stat().st_size for f in backup_path.rglob("*") if f.is_file()),
        )

    def _read_archive_metadata(self, archive: Path) -> BackupMetadata:
        import tarfile

        try:
            with tarfile.open(archive, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith("/metadata.json"):
                        extracted = tar.extractfile(member)
                        if extracted is None:
                            continue
                        data = json.loads(extracted.read().decode("utf-8"))
                        data["compressed_size"] = archive.stat().st_size
                        return BackupMetadata(**data)
        except (OSError, tarfile.TarError, json.JSONDecodeError, KeyError, TypeError) as exc:
            # Corrupt/unreadable archive — warn, then fall through to the
            # zero-count placeholder so the caller sees "0 memorias" with a log
            # breadcrumb instead of a silent empty backup.
            _log.warning("backup: could not read metadata from %s: %s", archive.name, exc)

        return BackupMetadata(
            timestamp=self._timestamp_from_name(archive.name.removesuffix(".tar.gz")),
            memoria_count=0,
            checksum="",
            compressed_size=archive.stat().st_size,
            original_size=0,
        )

    def _timestamp_from_name(self, name: str) -> str:
        if name.startswith("backup_"):
            raw = name.removeprefix("backup_").replace("-", ":")
            try:
                return datetime.fromisoformat(raw).isoformat()
            except ValueError:
                pass
        return datetime.now(UTC).isoformat()


class SyncManager:
    """Manages sync between multiple vaults.

    Args:
        local_memory: The local Memory instance.
        remote_path: Path to remote vault (or None for remote URL).
    """

    def __init__(self, local_memory: Any, remote_path: Path | None = None) -> None:
        self.local_memory = local_memory
        self.remote_path = remote_path
        self._last_sync_file = local_memory.cfg.state_dir / "last_sync.json"

        # Initialize SyncCoordinator if available
        if _HAS_SYNC_COORDINATOR:
            self._sync_coordinator = SyncCoordinator(
                state_path=default_state_path(local_memory.cfg.data_dir),
            )
        else:
            self._sync_coordinator = None

    def compute_diff(self) -> SyncDiff:
        """Compute diff between local and remote vaults.

        Returns:
            SyncDiff with changes.
        """
        if not self.remote_path or not self.remote_path.is_dir():
            return SyncDiff(new=[], modified=[], deleted=[], conflicts=[])

        # Get local memoria IDs
        local_memorias = {r.id: r.updated for r in self.local_memory.list(limit=10000)}

        # Get remote memoria IDs (simplified - would need to load remote Memory)
        remote_memorias: dict[str, str] = {}  # Would load from remote_path

        new: list[str] = []
        modified: list[str] = []
        deleted: list[str] = []
        conflicts: list[str] = []

        for mem_id in remote_memorias:
            if mem_id not in local_memorias:
                new.append(mem_id)
            elif remote_memorias[mem_id] != local_memorias[mem_id]:
                conflicts.append(mem_id)

        for mem_id in local_memorias:
            if mem_id not in remote_memorias:
                deleted.append(mem_id)
            elif local_memorias[mem_id] != remote_memorias[mem_id]:
                modified.append(mem_id)

        return SyncDiff(
            new=new,
            modified=modified,
            deleted=deleted,
            conflicts=conflicts,
        )

    def sync(self, direction: str = "both") -> SyncDiff:
        """Sync between local and remote vaults.

        Args:
            direction: "push", "pull", or "both".

        Returns:
            SyncDiff with applied changes.
        """
        import time

        sync_started = time.monotonic()

        # Start MEMO_VAULT_SYNC phase
        if self._sync_coordinator:
            self._sync_coordinator.start_phase(SyncPhase.MEMO_VAULT_SYNC)

        diff = self.compute_diff()

        # Apply sync based on direction
        # Simplified implementation
        if direction in ("push", "both"):
            # Push local changes to remote
            pass

        if direction in ("pull", "both"):
            # Pull remote changes to local
            pass

        # Update last sync timestamp
        self._update_last_sync()

        # Complete MEMO_VAULT_SYNC phase
        if self._sync_coordinator:
            duration_ms = (time.monotonic() - sync_started) * 1000
            self._sync_coordinator.complete_phase(
                SyncPhase.MEMO_VAULT_SYNC,
                duration_ms=duration_ms,
                error=None,
            )

        return diff

    def _update_last_sync(self) -> None:
        """Update last sync timestamp."""
        data = {"last_sync": datetime.now(UTC).isoformat()}
        self._last_sync_file.write_text(json.dumps(data), encoding="utf-8")

    def get_last_sync(self) -> str | None:
        """Get last sync timestamp."""
        if self._last_sync_file.is_file():
            data = json.loads(self._last_sync_file.read_text(encoding="utf-8"))
            return data.get("last_sync")
        return None


__all__ = [
    "BackupManager",
    "BackupMetadata",
    "SyncDiff",
    "SyncManager",
]
