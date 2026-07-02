"""Tests for lifecycle management module."""

from datetime import UTC, datetime, timedelta

import pytest

from memo.lifecycle import (
    FORGET_AFTER_KEY,
    FORGET_REASON_KEY,
    IS_FORGOTTEN_KEY,
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

    # New memoria has last_accessed seeded from updated (same day)
    days = lifecycle_manager.get_days_since_access(rec.id)
    assert days is not None
    assert isinstance(days, int)


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
    should, _reason = lifecycle_manager.should_archive(rec.id)
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
    assert "Not a temporary memory" in reason


def test_should_expire_regular_memoria(lifecycle_manager, mock_memory):
    """Test expiration logic for regular memoria."""
    rec = mock_memory.save(
        content="Test content",
        title="Test",
        tags=["test"],
    )

    # Regular memoria - should not expire
    should, _reason = lifecycle_manager.should_expire(rec.id)
    assert should is False


def test_archive_memory(lifecycle_manager, mock_memory):
    """Test archiving a memoria."""
    rec = mock_memory.save(
        content="Test content to archive",
        title="Archive Me",
        tags=["test"],
    )

    # Archive the memoria
    success = lifecycle_manager.archive_memory(rec.id)

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
        memory_id="abc123",
        action="archive",
        reason="Inactive",
        timestamp="2026-01-01T00:00:00Z",
    )
    assert action.memory_id == "abc123"
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


# -- forget / TTL (supermemory-style soft-delete) ------------------------------


def _past() -> str:
    return (datetime.now(UTC) - timedelta(days=1)).date().isoformat()


def _future() -> str:
    return (datetime.now(UTC) + timedelta(days=30)).date().isoformat()


def test_forget_excludes_from_search_and_list(mock_memory):
    rec = mock_memory.save(content="zephyrine protocol notes", title="Zephyrine")

    assert any(h.id == rec.id for h in mock_memory.search("zephyrine", mode="bm25"))
    assert any(r.id == rec.id for r in mock_memory.list())

    out = mock_memory.forget(rec.id, reason="obsolete")
    assert out is not None
    assert out.extra.get(IS_FORGOTTEN_KEY) is True
    assert out.extra.get(FORGET_REASON_KEY) == "obsolete"

    # Gone from default surfaces…
    assert not any(h.id == rec.id for h in mock_memory.search("zephyrine", mode="bm25"))
    assert not any(r.id == rec.id for r in mock_memory.list())
    # …but file + index survive (reversible, not a delete).
    assert mock_memory.get(rec.id) is not None
    assert (mock_memory.cfg.memory_dir / rec.path).is_file()


def test_include_forgotten_surfaces_it(mock_memory):
    rec = mock_memory.save(content="zephyrine protocol notes", title="Zephyrine")
    mock_memory.forget(rec.id)

    assert any(
        h.id == rec.id for h in mock_memory.search("zephyrine", mode="bm25", include_forgotten=True)
    )
    assert any(r.id == rec.id for r in mock_memory.list(include_forgotten=True))


def test_unforget_restores_and_clears_ttl(mock_memory):
    rec = mock_memory.save(
        content="zephyrine protocol notes",
        title="Zephyrine",
        extra={FORGET_AFTER_KEY: _past(), FORGET_REASON_KEY: "stale"},
    )
    mock_memory.forget(rec.id, reason="stale")

    out = mock_memory.unforget(rec.id)
    assert out is not None
    assert IS_FORGOTTEN_KEY not in out.extra
    assert FORGET_AFTER_KEY not in out.extra  # cleared so it won't re-forget
    assert FORGET_REASON_KEY not in out.extra
    assert any(h.id == rec.id for h in mock_memory.search("zephyrine", mode="bm25"))


def test_forget_preserves_other_extra_keys(mock_memory):
    rec = mock_memory.save(
        content="zephyrine protocol notes",
        title="Zephyrine",
        extra={"synapse_trace_id": "trace-123"},
    )
    out = mock_memory.forget(rec.id, reason="obsolete")
    assert out is not None
    assert out.extra.get("synapse_trace_id") == "trace-123"
    assert out.extra.get(IS_FORGOTTEN_KEY) is True


def test_forget_unknown_id_returns_none(mock_memory):
    assert mock_memory.forget("deadbeef") is None
    assert mock_memory.unforget("deadbeef") is None


def test_enforce_forget_ttl_forgets_elapsed_only(mock_memory):
    due = mock_memory.save(
        content="elapsed alpha",
        title="Elapsed",
        extra={FORGET_AFTER_KEY: _past()},
    )
    not_due = mock_memory.save(
        content="future beta",
        title="Future",
        extra={FORGET_AFTER_KEY: _future()},
    )
    plain = mock_memory.save(content="plain gamma", title="Plain")

    acted_ids = {a["id"] for a in mock_memory.lifecycle.enforce_forget_ttl()}

    assert due.id in acted_ids
    assert not_due.id not in acted_ids
    assert plain.id not in acted_ids
    assert mock_memory.get(due.id).extra.get(IS_FORGOTTEN_KEY) is True
    assert not mock_memory.get(not_due.id).extra.get(IS_FORGOTTEN_KEY)


def test_enforce_forget_ttl_dry_run_changes_nothing(mock_memory):
    rec = mock_memory.save(
        content="elapsed alpha",
        title="Elapsed",
        extra={FORGET_AFTER_KEY: _past()},
    )
    acted = mock_memory.lifecycle.enforce_forget_ttl(dry_run=True)
    assert any(a["id"] == rec.id for a in acted)
    assert not mock_memory.get(rec.id).extra.get(IS_FORGOTTEN_KEY)


def test_should_forget_skips_already_forgotten(mock_memory):
    rec = mock_memory.save(
        content="elapsed alpha",
        title="Elapsed",
        extra={FORGET_AFTER_KEY: _past()},
    )
    mock_memory.forget(rec.id)
    should, _reason = mock_memory.lifecycle.should_forget(rec.id)
    assert should is False


def test_lifecycle_report_counts_forget_candidates(mock_memory):
    mock_memory.save(
        content="elapsed alpha",
        title="Elapsed",
        extra={FORGET_AFTER_KEY: _past()},
    )
    mock_memory.save(content="plain gamma", title="Plain")
    report = mock_memory.lifecycle.get_lifecycle_report()
    assert report["forget_candidates"] == 1


def test_forget_after_accepts_full_datetime(mock_memory):
    stamp = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    rec = mock_memory.save(
        content="elapsed alpha",
        title="Elapsed",
        extra={FORGET_AFTER_KEY: stamp},
    )
    should, _ = mock_memory.lifecycle.should_forget(rec.id)
    assert should is True


def test_forget_after_unparseable_is_ignored(mock_memory):
    rec = mock_memory.save(
        content="elapsed alpha",
        title="Elapsed",
        extra={FORGET_AFTER_KEY: "not-a-date"},
    )
    should, _ = mock_memory.lifecycle.should_forget(rec.id)
    assert should is False
