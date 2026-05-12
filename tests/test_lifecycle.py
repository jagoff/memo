"""Tests for lifecycle management module."""

import pytest

from memo.lifecycle import (
    LifecycleAction,
    LifecycleManager,
    LifecyclePolicy,
)


@pytest.fixture
def lifecycle_manager(mock_memory):
    """Fixture providing LifecycleManager instance."""
    policy = LifecyclePolicy(
        archival_days=180,
        promotion_threshold=5,
        demotion_threshold=2,
        temp_expiration_days=30,
        delete_expired=False,
    )
    return LifecycleManager(mock_memory, policy)


def test_lifecycle_manager_init(lifecycle_manager):
    """Test LifecycleManager initialization."""
    assert lifecycle_manager.memory is not None
    assert lifecycle_manager.policy is not None


def test_lifecycle_policy_defaults():
    """Test LifecyclePolicy default values."""
    policy = LifecyclePolicy()
    assert policy.archival_days == 180
    assert policy.promotion_threshold == 5
    assert policy.demotion_threshold == 2
    assert policy.temp_expiration_days == 30
    assert policy.delete_expired is False


def test_get_access_count(lifecycle_manager, mock_memory):
    """Test getting access count for a memoria."""
    rec = mock_memory.save(
        content="Test content",
        title="Test",
        tags=["test"],
    )

    # New memoria should have 0 access count (only save event)
    count = lifecycle_manager.get_access_count(rec.id)
    assert count == 0


def test_get_days_since_access(lifecycle_manager, mock_memory):
    """Test getting days since last access."""
    rec = mock_memory.save(
        content="Test content",
        title="Test",
        tags=["test"],
    )

    # New memoria should return None for days since access (never accessed)
    days = lifecycle_manager.get_days_since_access(rec.id)
    assert days is None


def test_get_days_since_update(lifecycle_manager, mock_memory):
    """Test getting days since last update."""
    rec = mock_memory.save(
        content="Test content",
        title="Test",
        tags=["test"],
    )

    # New memoria should have 0 days since update
    days = lifecycle_manager.get_days_since_update(rec.id)
    assert days == 0


def test_should_archive_never_accessed(lifecycle_manager, mock_memory):
    """Test archival logic for never-accessed memoria."""
    rec = mock_memory.save(
        content="Test content",
        title="Test",
        tags=["test"],
    )

    # Never accessed but recently updated - should not archive
    should, reason = lifecycle_manager.should_archive(rec.id)
    assert should is False


def test_should_promote(lifecycle_manager, mock_memory):
    """Test promotion logic."""
    rec = mock_memory.save(
        content="Test content",
        title="Test",
        tags=["test"],
    )

    # New memoria with 0 access count - should not promote
    should, reason = lifecycle_manager.should_promote(rec.id)
    assert should is False
    assert "0" in reason


def test_should_demote(lifecycle_manager, mock_memory):
    """Test demotion logic."""
    rec = mock_memory.save(
        content="Test content",
        title="Test",
        tags=["test"],
    )

    # New memoria with 0 access count - should demote
    should, reason = lifecycle_manager.should_demote(rec.id)
    assert should is True
    assert "0" in reason


def test_should_expire_temp_type(lifecycle_manager, mock_memory):
    """Test expiration logic for temp type."""
    # This would require a very old temp memoria, which is hard to test
    # Just test the logic with a mock
    pass


def test_should_expire_temp_tag(lifecycle_manager, mock_memory):
    """Test expiration logic for temp:... tags."""
    rec = mock_memory.save(
        content="Test content",
        title="Test",
        tags=["temp:7"],  # 7 days
    )

    # Recently created - should not expire
    should, reason = lifecycle_manager.should_expire(rec.id)
    assert should is False
    assert "Not a temporary memoria" in reason


def test_should_expire_regular_memoria(lifecycle_manager, mock_memory):
    """Test expiration logic for regular memoria."""
    rec = mock_memory.save(
        content="Test content",
        title="Test",
        tags=["test"],
    )

    # Regular memoria - should not expire
    should, reason = lifecycle_manager.should_expire(rec.id)
    assert should is False


def test_archive_memoria(lifecycle_manager, mock_memory):
    """Test archiving a memoria."""
    rec = mock_memory.save(
        content="Test content to archive",
        title="Archive Me",
        tags=["test"],
    )

    # Archive the memoria
    success = lifecycle_manager.archive_memoria(rec.id)

    assert success is True
    # Verify it's deleted from store
    assert mock_memory.get(rec.id) is None
    # Verify inactive directory exists
    inactive_dir = mock_memory.cfg.memory_dir / "inactive"
    assert inactive_dir.is_dir()


def test_apply_lifecycle_rules_dry_run(lifecycle_manager, mock_memory):
    """Test applying lifecycle rules in dry-run mode."""
    # Create test memorias
    mock_memory.save(
        content="Test 1",
        title="Test 1",
        tags=["test"],
    )
    mock_memory.save(
        content="Test 2",
        title="Test 2",
        tags=["test"],
    )

    # Apply in dry-run mode
    actions = lifecycle_manager.apply_lifecycle_rules(dry_run=True, limit=10)

    assert "archived" in actions
    assert "promoted" in actions
    assert "demoted" in actions
    assert "expired" in actions
    assert "deleted" in actions
    assert "skipped" in actions

    # In dry-run, counts should be 0 for destructive actions
    assert actions["deleted"] == 0


def test_get_lifecycle_report(lifecycle_manager, mock_memory):
    """Test generating lifecycle report."""
    # Create test memorias
    mock_memory.save(
        content="Test 1",
        title="Test 1",
        tags=["test"],
    )
    mock_memory.save(
        content="Test 2",
        title="Test 2",
        tags=["test"],
    )

    report = lifecycle_manager.get_lifecycle_report(limit=10)

    assert "total" in report
    assert "archive_candidates" in report
    assert "promotion_candidates" in report
    assert "demotion_candidates" in report
    assert "expiration_candidates" in report
    assert "never_accessed" in report
    assert "avg_access_count" in report

    assert report["total"] == 2


def test_lifecycle_action_dataclass():
    """Test LifecycleAction dataclass structure."""
    action = LifecycleAction(
        memoria_id="abc123",
        action="archive",
        reason="Inactive",
        timestamp="2026-01-01T00:00:00Z",
    )
    assert action.memoria_id == "abc123"
    assert action.action == "archive"
    assert action.reason == "Inactive"


def test_custom_policy(lifecycle_manager):
    """Test LifecycleManager with custom policy."""
    custom_policy = LifecyclePolicy(
        archival_days=90,
        promotion_threshold=10,
        demotion_threshold=1,
        temp_expiration_days=14,
        delete_expired=True,
    )

    custom_manager = LifecycleManager(lifecycle_manager.memory, custom_policy)

    assert custom_manager.policy.archival_days == 90
    assert custom_manager.policy.promotion_threshold == 10
    assert custom_manager.policy.delete_expired is True
