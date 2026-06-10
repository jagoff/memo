"""Tests for sharing module."""

import pytest

from memo.sharing import (
    Comment,
    Permission,
    Share,
    ShareLink,
    ShareManager,
    ShareStore,
)


@pytest.fixture
def share_store(tmp_cfg):
    """Fixture providing ShareStore instance."""
    return ShareStore(tmp_cfg.state_dir)


@pytest.fixture
def share_manager(share_store):
    """Fixture providing ShareManager instance."""
    return ShareManager(share_store)


def test_share_store_init(share_store):
    """Test ShareStore initialization."""
    assert share_store.state_dir.is_dir()


def test_share_store_add_share(share_store):
    """Test adding a share."""
    share_store.add_share(
        memoria_id="test-id",
        shared_with="user@example.com",
        permission="read",
    )

    shares = share_store.get_shares("test-id")
    assert len(shares) == 1
    assert shares[0].shared_with == "user@example.com"


def test_share_store_get_shares(share_store):
    """Test getting shares for a memoria."""
    share_store.add_share("test-id", "user1@example.com", "read")
    share_store.add_share("test-id", "user2@example.com", "comment")

    shares = share_store.get_shares("test-id")
    assert len(shares) == 2


def test_share_store_remove_share(share_store):
    """Test removing a share."""
    share_store.add_share("test-id", "user@example.com", "read")

    success = share_store.remove_share("test-id", "user@example.com")
    assert success is True

    shares = share_store.get_shares("test-id")
    assert len(shares) == 0


def test_share_store_create_share_link(share_store):
    """Test creating a share link."""
    link = share_store.create_share_link(
        memoria_id="test-id",
        permission="read",
        expires_hours=24,
    )

    assert link.memoria_id == "test-id"
    assert link.link_token
    assert link.permission == "read"
    assert link.access_count == 0


def test_share_store_get_share_link(share_store):
    """Test getting a share link."""
    link = share_store.create_share_link("test-id", "read", 24)

    retrieved = share_store.get_share_link(link.link_token)
    assert retrieved is not None
    assert retrieved.memoria_id == "test-id"


def test_share_store_get_share_link_not_found(share_store):
    """Test getting a non-existent share link."""
    link = share_store.get_share_link("nonexistent")
    assert link is None


def test_share_store_increment_link_access(share_store):
    """Test incrementing link access count."""
    link = share_store.create_share_link("test-id", "read", 24)

    share_store.increment_link_access(link.link_token)

    retrieved = share_store.get_share_link(link.link_token)
    assert retrieved.access_count == 1


def test_share_store_add_comment(share_store):
    """Test adding a comment."""
    share_store.add_comment(
        memoria_id="test-id",
        author="user",
        content="Test comment",
    )

    comments = share_store.get_comments("test-id")
    assert len(comments) == 1
    assert comments[0].content == "Test comment"


def test_share_store_get_comments(share_store):
    """Test getting comments for a memoria."""
    share_store.add_comment("test-id", "user1", "Comment 1")
    share_store.add_comment("test-id", "user2", "Comment 2")

    comments = share_store.get_comments("test-id")
    assert len(comments) == 2


def test_share_store_delete_comment(share_store):
    """Test deleting a comment."""
    share_store.add_comment("test-id", "user", "Test comment")
    comments = share_store.get_comments("test-id")

    success = share_store.delete_comment("test-id", comments[0].created_at)
    assert success is True

    comments_after = share_store.get_comments("test-id")
    assert len(comments_after) == 0


def test_share_store_persistence(tmp_cfg):
    """Test sharing data persistence across instances."""
    state_dir = tmp_cfg.state_dir

    # Create first instance and add share
    store1 = ShareStore(state_dir)
    store1.add_share("test-id", "user@example.com", "read")

    # Create second instance and verify persistence
    store2 = ShareStore(state_dir)
    shares = store2.get_shares("test-id")

    assert len(shares) == 1
    assert shares[0].shared_with == "user@example.com"


def test_share_manager_init(share_manager):
    """Test ShareManager initialization."""
    assert share_manager.share_store is not None


def test_share_manager_share_with_user(share_manager):
    """Test sharing with a user."""
    share = share_manager.share_with_user(
        memoria_id="test-id",
        shared_with="user@example.com",
        permission="read",
    )

    assert share.shared_with == "user@example.com"
    assert share.permission == "read"


def test_share_manager_unshare_with_user(share_manager):
    """Test unsharing with a user."""
    share_manager.share_with_user("test-id", "user@example.com", "read")

    success = share_manager.unshare_with_user("test-id", "user@example.com")
    assert success is True


def test_share_manager_create_link(share_manager):
    """Test creating a share link."""
    link = share_manager.create_link(
        memoria_id="test-id",
        permission="read",
        expires_hours=24,
    )

    assert link.startswith("https://memo.app/share/")
    assert "test-id" not in link  # Should use token, not ID


def test_share_manager_validate_link(share_manager):
    """Test validating a share link."""
    link_url = share_manager.create_link("test-id", "read", 24)
    token = link_url.split("/")[-1]

    link = share_manager.validate_link(token)
    assert link is not None
    assert link.memoria_id == "test-id"


def test_share_manager_validate_expired_link(share_manager, tmp_cfg):
    """An expired link fails validation after the store is reloaded from disk.

    Exercises the persisted expiry check (not in-memory object mutation): the
    link is created with a past `expires_at`, persisted, then validated through
    a FRESH ShareManager built from the same state dir.
    """
    link = share_manager.share_store.create_share_link(
        memoria_id="test-id",
        permission="read",
        expires_hours=-1,  # already in the past, persisted to disk
    )

    fresh = ShareManager(ShareStore(tmp_cfg.state_dir))
    assert fresh.validate_link(link.link_token) is None


def test_share_manager_add_comment(share_manager):
    """Test adding a comment."""
    comment = share_manager.add_comment(
        memoria_id="test-id",
        author="user",
        content="Test comment",
    )

    assert comment.content == "Test comment"
    assert comment.author == "user"


def test_share_manager_get_comments(share_manager):
    """Test getting comments."""
    share_manager.add_comment("test-id", "user1", "Comment 1")
    share_manager.add_comment("test-id", "user2", "Comment 2")

    comments = share_manager.get_comments("test-id")
    assert len(comments) == 2


def test_share_manager_delete_comment(share_manager):
    """Test deleting a comment."""
    comment = share_manager.add_comment("test-id", "user", "Test comment")

    success = share_manager.delete_comment("test-id", comment.created_at)
    assert success is True


def test_permission_enum():
    """Test Permission enum values."""
    assert Permission.READ.value == "read"
    assert Permission.COMMENT.value == "comment"
    assert Permission.EDIT.value == "edit"
    assert Permission.ADMIN.value == "admin"


def test_share_dataclass():
    """Test Share dataclass structure."""
    share = Share(
        memoria_id="test-id",
        shared_with="user@example.com",
        permission="read",
        shared_at="2026-01-01T00:00:00Z",
        expires_at=None,
        share_link=None,
    )
    assert share.memoria_id == "test-id"
    assert share.shared_with == "user@example.com"


def test_comment_dataclass():
    """Test Comment dataclass structure."""
    comment = Comment(
        memoria_id="test-id",
        author="user",
        content="Test comment",
        created_at="2026-01-01T00:00:00Z",
        parent_id=None,
    )
    assert comment.memoria_id == "test-id"
    assert comment.content == "Test comment"


def test_share_link_dataclass():
    """Test ShareLink dataclass structure."""
    link = ShareLink(
        memoria_id="test-id",
        link_token="abc123",
        permission="read",
        created_at="2026-01-01T00:00:00Z",
        expires_at="2026-01-02T00:00:00Z",
        password_hash=None,
        access_count=5,
    )
    assert link.memoria_id == "test-id"
    assert link.access_count == 5
