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


def test_backup_manager_restore_backup(backup_manager, mock_memory):
    """Test restoring from backup."""
    # Create a memoria and backup
    rec = mock_memory.save(
        content="Original content",
        title="Original",
        tags=["test"],
    )

    backup_manager.create_backup(compress=False, name="restore_test")

    # Delete the memoria
    mock_memory.delete(rec.id)
    assert mock_memory.get(rec.id) is None

    # Restore
    success = backup_manager.restore_backup("restore_test")

    assert success is True


def test_backup_manager_restore_nonexistent_backup(backup_manager):
    """Test restoring from non-existent backup."""
    success = backup_manager.restore_backup("nonexistent")
    assert success is False


def test_sync_manager_init(sync_manager):
    """Test SyncManager initialization."""
    assert sync_manager.local_memory is not None


def test_sync_manager_compute_diff_no_remote(sync_manager):
    """Test diff with no remote configured."""
    diff = sync_manager.compute_diff()

    assert diff.new == []
    assert diff.modified == []
    assert diff.deleted == []
    assert diff.conflicts == []


def test_sync_manager_update_last_sync(sync_manager):
    """Test updating last sync timestamp."""
    sync_manager._update_last_sync()

    last_sync = sync_manager.get_last_sync()
    assert last_sync is not None


def test_sync_manager_get_last_sync_none(sync_manager):
    """Test getting last sync when never synced."""
    last_sync = sync_manager.get_last_sync()
    assert last_sync is None


def test_sync_diff_dataclass():
    """Test SyncDiff dataclass structure."""
    diff = SyncDiff(
        new=["id1"],
        modified=["id2"],
        deleted=["id3"],
        conflicts=["id4"],
    )
    assert len(diff.new) == 1
    assert len(diff.modified) == 1
    assert len(diff.deleted) == 1
    assert len(diff.conflicts) == 1


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
