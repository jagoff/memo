"""Tests for sync & backup module."""

import hashlib
import shutil

import pytest

from memo.config import Config
from memo.memory import Memory
from memo.sync import (
    BackupManager,
    BackupMetadata,
    SyncDiff,
    SyncManager,
)


def _build_remote_memory(root) -> Memory:
    """An isolated `Memory` (own data/state dir → distinct device id) with a
    deterministic stub embedder, standing in for another machine."""
    data = root / "data"
    state = root / "state"
    data.mkdir(parents=True)
    state.mkdir(parents=True)
    cfg = Config(data_dir=data, state_dir=state, reranker_enabled=False)
    mem = Memory(cfg)

    def _fake_embedding(text: str) -> list[float]:
        digest = hashlib.sha256((text or "").encode("utf-8")).digest()
        values = [((digest[i % len(digest)] / 255.0) * 2.0) - 1.0 for i in range(cfg.embedder_dims)]
        norm = sum(v * v for v in values) ** 0.5
        return [v / norm for v in values]

    mem.embedder.embed = lambda inputs: [_fake_embedding(t) for t in inputs]
    mem.embedder.embed_query = lambda q: _fake_embedding(q)
    return mem


def _copy_md(remote: Memory, local: Memory, rel_path: str) -> None:
    """Copy a memoria's .md from remote → local memory dir, preserving subdirs
    (stands in for the file-sync layer, e.g. iCloud)."""
    src = remote.cfg.memory_dir / rel_path
    dst = local.cfg.memory_dir / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def test_sync_replay_save_preserves_record_id_no_duplicate(sync_manager, mock_memory, tmp_path):
    """Replaying a remote `save` must index the existing `{id}.md` under its
    own id — not mint a fresh uuid (which forks a duplicate memoria). B1."""
    remote = _build_remote_memory(tmp_path / "remote")
    try:
        rec = remote.save(content="A durable fact about the trinity.", title="Fact", tags=["t"])
        # Simulate the .md having already synced to the local machine (iCloud).
        _copy_md(remote, mock_memory, rec.path)

        diff = sync_manager.sync_from_remote(remote.cfg.history_db)

        assert diff.applied == 1
        assert mock_memory.get(rec.id) is not None, "synced record must keep its id"
        rows = mock_memory.store.list_recent(limit=100)
        assert len(rows) == 1, "must not create a duplicate under a new uuid"
        assert rows[0]["id"] == rec.id
    finally:
        remote.close()


def test_sync_cursor_not_wedged_by_missing_then_present_file(sync_manager, mock_memory, tmp_path):
    """A remote save whose file hasn't synced yet must defer (cursor stays put,
    order preserved) and apply on a later pass — never permanently skipped. B2."""
    remote = _build_remote_memory(tmp_path / "remote")
    try:
        rec = remote.save(content="Deferred fact.", title="Deferred", tags=["t"])

        # File not present locally yet → first pass applies nothing, advances nothing.
        diff1 = sync_manager.sync_from_remote(remote.cfg.history_db)
        assert diff1.applied == 0
        assert mock_memory.get(rec.id) is None

        # File arrives; a second pass must still see and apply the event.
        _copy_md(remote, mock_memory, rec.path)
        diff2 = sync_manager.sync_from_remote(remote.cfg.history_db)
        assert diff2.applied == 1
        assert mock_memory.get(rec.id) is not None
    finally:
        remote.close()


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

    assert metadata.memory_count > 0
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
        memory_count=10,
        checksum="abc123",
        compressed_size=1000,
        original_size=5000,
    )
    assert metadata.memory_count == 10
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
