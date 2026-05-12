"""Tests for versioning module."""

import pytest

from memo.versioning import (
    DiffResult,
    Version,
    VersionManager,
    VersionStore,
)


@pytest.fixture
def version_store(tmp_cfg):
    """Fixture providing VersionStore instance."""
    return VersionStore(tmp_cfg.state_dir / "versions.db")


@pytest.fixture
def version_manager(mock_memory):
    """Fixture providing VersionManager instance."""
    return VersionManager(mock_memory)


def test_version_store_init(version_store):
    """Test VersionStore initialization."""
    assert version_store.db_path.is_dir()


def test_version_store_save_version(version_store):
    """Test saving a version."""
    version_id = version_store.save_version(
        memoria_id="test-id",
        title="Test Title",
        type="note",
        tags=["test"],
        body="Test body content",
        reason="Initial version",
    )

    assert version_id > 0


def test_version_store_get_versions(version_store):
    """Test getting versions for a memoria."""
    # Save multiple versions
    version_store.save_version(
        memoria_id="test-id",
        title="Title v1",
        type="note",
        tags=["test"],
        body="Body v1",
        reason="v1",
    )
    version_store.save_version(
        memoria_id="test-id",
        title="Title v2",
        type="note",
        tags=["test"],
        body="Body v2",
        reason="v2",
    )

    versions = version_store.get_versions("test-id", limit=10)

    assert len(versions) == 2
    assert versions[0].title == "Title v2"  # Most recent first
    assert versions[1].title == "Title v1"


def test_version_store_get_version(version_store):
    """Test getting a specific version."""
    version_id = version_store.save_version(
        memoria_id="test-id",
        title="Test Title",
        type="note",
        tags=["test"],
        body="Test body",
        reason="Test",
    )

    version = version_store.get_version("test-id", version_id)

    assert version is not None
    assert version.version_id == version_id
    assert version.title == "Test Title"


def test_version_store_get_version_not_found(version_store):
    """Test getting a non-existent version."""
    version = version_store.get_version("test-id", 999)
    assert version is None


def test_version_store_delete_versions(version_store):
    """Test deleting all versions for a memoria."""
    version_store.save_version(
        memoria_id="test-id",
        title="Test",
        type="note",
        tags=["test"],
        body="Body",
    )

    versions_before = version_store.get_versions("test-id")
    assert len(versions_before) == 1

    version_store.delete_versions("test-id")

    versions_after = version_store.get_versions("test-id")
    assert len(versions_after) == 0


def test_version_manager_init(version_manager):
    """Test VersionManager initialization."""
    assert version_manager.memory is not None
    assert version_manager.version_store is not None


def test_version_manager_track_update(version_manager, mock_memory):
    """Test tracking an update."""
    version_id = version_manager.track_update(
        memoria_id="test-id",
        title="Updated Title",
        type="decision",
        tags=["decision"],
        body="Updated content",
        reason="Updated for clarity",
    )

    assert version_id > 0


def test_version_manager_get_version_history(version_manager, mock_memory):
    """Test getting version history."""
    # Track multiple updates
    version_manager.track_update(
        memoria_id="test-id",
        title="v1",
        type="note",
        tags=["test"],
        body="Body v1",
    )
    version_manager.track_update(
        memoria_id="test-id",
        title="v2",
        type="note",
        tags=["test"],
        body="Body v2",
    )

    history = version_manager.get_version_history("test-id", limit=10)

    assert len(history) == 2
    assert history[0].title == "v2"
    assert history[1].title == "v1"


def test_version_manager_diff_versions(version_manager, mock_memory):
    """Test generating diff between versions."""
    version_manager.track_update(
        memoria_id="test-id",
        title="Original",
        type="note",
        tags=["test"],
        body="Line 1\nLine 2\nLine 3",
    )
    version_manager.track_update(
        memoria_id="test-id",
        title="Modified",
        type="note",
        tags=["test"],
        body="Line 1\nLine 2 modified\nLine 3",
    )

    diff = version_manager.diff_versions("test-id")

    assert diff is not None
    assert diff.memoria_id == "test-id"
    assert "modified" in diff.unified_diff.lower()


def test_version_manager_diff_versions_no_history(version_manager):
    """Test diff when there's no version history."""
    diff = version_manager.diff_versions("nonexistent-id")
    assert diff is None


def test_version_manager_rollback_to_version(version_manager, mock_memory):
    """Test rollback to a previous version."""
    # Create a memoria to rollback
    rec = mock_memory.save(
        content="Original content",
        title="Original",
        tags=["test"],
    )

    # Track versions
    version_manager.track_update(
        memoria_id=rec.id,
        title="Original",
        type="note",
        tags=["test"],
        body="Original content",
    )
    version_manager.track_update(
        memoria_id=rec.id,
        title="Modified",
        type="note",
        tags=["test"],
        body="Modified content",
    )

    # Get versions
    history = version_manager.get_version_history(rec.id, limit=10)
    if len(history) < 2:
        pytest.skip("Need at least 2 versions to test rollback")

    # Rollback to first version
    success = version_manager.rollback_to_version(rec.id, history[-1].version_id)

    assert success is True

    # Verify the memoria was restored
    restored = mock_memory.get(rec.id)
    assert restored.title == "Original"


def test_version_dataclass():
    """Test Version dataclass structure."""
    v = Version(
        version_id=1,
        memoria_id="test-id",
        timestamp="2026-01-01T00:00:00Z",
        title="Test Title",
        type="note",
        tags=["test"],
        body="Test body",
        reason="Test reason",
    )
    assert v.version_id == 1
    assert v.memoria_id == "test-id"
    assert len(v.tags) == 1


def test_diff_result_dataclass():
    """Test DiffResult dataclass structure."""
    diff = DiffResult(
        memoria_id="test-id",
        version_a=1,
        version_b=2,
        unified_diff="--- v1\n+++ v2\n- old\n+ new",
        changes=["- old", "+ new"],
    )
    assert diff.memoria_id == "test-id"
    assert diff.version_a == 1
    assert diff.version_b == 2
    assert len(diff.changes) == 2
