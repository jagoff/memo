"""Tests for sync & backup module."""

import pytest

from memo.sync import (
    BackupManager,
    BackupMetadata,
    SyncDiff,
    SyncManager,
)


@pytest.fixture
def backup_manager(tmp_cfg):
    """Fixture providing BackupManager instance."""
    return BackupManager(
        memory_dir=tmp_cfg.memory_dir,
        db_dir=tmp_cfg.state_dir,
        backup_dir=tmp_cfg.state_dir / "backups",
    )


@pytest.fixture
def sync_manager(mock_memory):
    """Fixture providing SyncManager instance."""
    return SyncManager(mock_memory)


def test_backup_manager_init(backup_manager):
    """Test BackupManager initialization."""
    assert backup_manager.memory_dir.is_dir()
    assert backup_manager.backup_dir.is_dir()


def test_backup_manager_create_backup(backup_manager, mock_memory):
    """Test creating a backup."""
    # Create a test memoria
    mock_memory.save(
        content="Test content",
        title="Test",
        tags=["test"],
    )

    metadata = backup_manager.create_backup(compress=False)

    assert metadata.memoria_count > 0
    assert metadata.checksum
    assert metadata.original_size > 0


def test_backup_manager_create_compressed_backup(backup_manager, mock_memory):
    """Test creating a compressed backup."""
    mock_memory.save(
        content="Test content",
        title="Test",
        tags=["test"],
    )

    metadata = backup_manager.create_backup(compress=True)

    assert metadata.compressed_size > 0
    assert metadata.compressed_size < metadata.original_size


def test_backup_manager_list_backups(backup_manager):
    """Test listing backups."""
    # Create a backup
    backup_manager.create_backup(compress=True, name="test_backup")

    backups = backup_manager.list_backups()

    assert len(backups) >= 1
    assert all(isinstance(b, BackupMetadata) for b in backups)


def test_sync_manager_init(sync_manager, mock_memory):
    """SyncManager holds the local Memory it replays events into."""
    assert sync_manager.mem is mock_memory


def test_sync_from_remote_missing_file_reports_error(sync_manager, tmp_path):
    """A missing remote history.db is an error, not a crash."""
    diff = sync_manager.sync_from_remote(tmp_path / "does-not-exist.db")
    assert isinstance(diff, SyncDiff)
    assert diff.applied == 0
    assert diff.errors == 1


def test_sync_from_remote_loopback_is_noop(sync_manager, mock_memory):
    """Syncing from this device's own history is a no-op (same device id)."""
    # Force the history.db into existence with at least one event.
    mock_memory.save(content="seed", title="Seed", tags=["x"])
    diff = sync_manager.sync_from_remote(mock_memory.cfg.history_db)
    assert diff == SyncDiff(0, 0, 0)


def test_sync_diff_dataclass():
    """SyncDiff carries applied/conflicts/errors counters."""
    diff = SyncDiff(applied=3, conflicts=1, errors=2)
    assert diff.applied == 3
    assert diff.conflicts == 1
    assert diff.errors == 2


def test_backup_metadata_dataclass():
    """Test BackupMetadata dataclass structure."""
    metadata = BackupMetadata(
        timestamp="2026-01-01T00:00:00Z",
        memoria_count=10,
        checksum="abc123",
        compressed_size=1000,
        original_size=5000,
    )
    assert metadata.memoria_count == 10
    assert metadata.checksum == "abc123"
    assert metadata.compressed_size == 1000


def test_backup_manager_compute_checksum(backup_manager, tmp_path):
    """Test checksum computation."""
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content", encoding="utf-8")

    checksum = backup_manager._compute_checksum(tmp_path)

    assert checksum
    assert len(checksum) == 64  # SHA256 hex length
