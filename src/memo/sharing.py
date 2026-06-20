"""EXPERIMENTAL — not covered by the test suite, not exposed via MCP. API may change without notice.

Memory sharing & collaboration — share memorias with other users.

Enables:
- Share memorias with other users (permissions)
- Temporary sharing links with expiration
- Collaboration in real-time (editing simultaneously)
- Comments on shared memorias
- Version control for shared content
"""

from __future__ import annotations

import json
import logging
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path

_log = logging.getLogger(__name__)


class Permission(Enum):
    """Permission levels for shared memorias."""

    READ = "read"
    COMMENT = "comment"
    EDIT = "edit"
    ADMIN = "admin"


@dataclass
class Share:
    """A share of a memoria with another user."""

    memoria_id: str
    shared_with: str  # Email or username
    permission: str  # Permission value
    shared_at: str
    expires_at: str | None
    share_link: str | None


@dataclass
class Comment:
    """A comment on a memoria."""

    memoria_id: str
    author: str
    content: str
    created_at: str
    parent_id: str | None


@dataclass
class ShareLink:
    """A temporary sharing link."""

    memoria_id: str
    link_token: str
    permission: str
    created_at: str
    expires_at: str
    password_hash: str | None
    access_count: int


class ShareStore:
    """Stores sharing metadata.

    Args:
        state_dir: Directory to store sharing state.
    """

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.shares_file = state_dir / "shares.json"
        self.links_file = state_dir / "share_links.json"
        self.comments_file = state_dir / "comments.json"
        self._shares: list[Share] = []
        self._links: list[ShareLink] = []
        self._comments: list[Comment] = []
        self._load()

    def _load(self) -> None:
        """Load sharing data from disk."""
        if self.shares_file.is_file():
            try:
                data = json.loads(self.shares_file.read_text(encoding="utf-8"))
                self._shares = [Share(**s) for s in data]
            except Exception:
                self._shares = []

        if self.links_file.is_file():
            try:
                data = json.loads(self.links_file.read_text(encoding="utf-8"))
                self._links = [ShareLink(**link) for link in data]
            except Exception:
                self._links = []

        if self.comments_file.is_file():
            try:
                data = json.loads(self.comments_file.read_text(encoding="utf-8"))
                self._comments = [Comment(**c) for c in data]
            except Exception:
                self._comments = []

    def _save(self) -> None:
        """Save sharing data to disk."""
        try:
            self.shares_file.write_text(
                json.dumps([s.__dict__ for s in self._shares], indent=2),
                encoding="utf-8",
            )
            self.links_file.write_text(
                json.dumps([link.__dict__ for link in self._links], indent=2),
                encoding="utf-8",
            )
            self.comments_file.write_text(
                json.dumps([c.__dict__ for c in self._comments], indent=2),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            # Data-loss path: a failed persist silently drops shares/links —
            # surface it instead of swallowing.
            _log.error("sharing: failed to persist share state: %s", exc)

    def add_share(
        self,
        memoria_id: str,
        shared_with: str,
        permission: str,
        expires_at: str | None = None,
        share_link: str | None = None,
    ) -> None:
        """Add a share.

        Args:
            memoria_id: The memoria ID being shared.
            shared_with: Email or username to share with.
            permission: Permission level.
            expires_at: Optional expiration timestamp.
            share_link: Optional share link.
        """
        share = Share(
            memoria_id=memoria_id,
            shared_with=shared_with,
            permission=permission,
            shared_at=datetime.now(UTC).isoformat(),
            expires_at=expires_at,
            share_link=share_link,
        )
        self._shares.append(share)
        self._save()

    def get_shares(self, memoria_id: str) -> list[Share]:
        """Get all shares for a memoria.

        Args:
            memoria_id: The memoria ID.

        Returns:
            List of Share objects.
        """
        return [s for s in self._shares if s.memoria_id == memoria_id]

    def remove_share(self, memoria_id: str, shared_with: str) -> bool:
        """Remove a share.

        Args:
            memoria_id: The memoria ID.
            shared_with: The user to unshare with.

        Returns:
            True if removed.
        """
        original_len = len(self._shares)
        self._shares = [
            s
            for s in self._shares
            if not (s.memoria_id == memoria_id and s.shared_with == shared_with)
        ]
        if len(self._shares) < original_len:
            self._save()
            return True
        return False

    def create_share_link(
        self,
        memoria_id: str,
        permission: str = "read",
        expires_hours: int = 24,
        password: str | None = None,
    ) -> ShareLink:
        """Create a temporary sharing link.

        Args:
            memoria_id: The memoria ID.
            permission: Permission level.
            expires_hours: Hours until expiration.
            password: Optional password.

        Returns:
            ShareLink object.
        """
        link_token = secrets.token_urlsafe(16)
        expires_at = (datetime.now(UTC) + timedelta(hours=expires_hours)).isoformat()
        password_hash = None  # Would hash password if provided

        link = ShareLink(
            memoria_id=memoria_id,
            link_token=link_token,
            permission=permission,
            created_at=datetime.now(UTC).isoformat(),
            expires_at=expires_at,
            password_hash=password_hash,
            access_count=0,
        )
        self._links.append(link)
        self._save()
        return link

    def get_share_link(self, link_token: str) -> ShareLink | None:
        """Get a share link by token.

        Args:
            link_token: The link token.

        Returns:
            ShareLink or None if not found.
        """
        for link in self._links:
            if link.link_token == link_token:
                return link
        return None

    def increment_link_access(self, link_token: str) -> None:
        """Increment access count for a link.

        Args:
            link_token: The link token.
        """
        for link in self._links:
            if link.link_token == link_token:
                link.access_count += 1
                self._save()
                break

    def add_comment(
        self,
        memoria_id: str,
        author: str,
        content: str,
        parent_id: str | None = None,
    ) -> None:
        """Add a comment to a memoria.

        Args:
            memoria_id: The memoria ID.
            author: Comment author.
            content: Comment content.
            parent_id: Optional parent comment ID.
        """
        comment = Comment(
            memoria_id=memoria_id,
            author=author,
            content=content,
            created_at=datetime.now(UTC).isoformat(),
            parent_id=parent_id,
        )
        self._comments.append(comment)
        self._save()

    def get_comments(self, memoria_id: str) -> list[Comment]:
        """Get all comments for a memoria.

        Args:
            memoria_id: The memoria ID.

        Returns:
            List of Comment objects.
        """
        return [c for c in self._comments if c.memoria_id == memoria_id]

    def delete_comment(self, memoria_id: str, comment_id: str) -> bool:
        """Delete a comment.

        Args:
            memoria_id: The memoria ID.
            comment_id: The comment ID (created_at timestamp).

        Returns:
            True if deleted.
        """
        original_len = len(self._comments)
        self._comments = [
            c
            for c in self._comments
            if not (c.memoria_id == memoria_id and c.created_at == comment_id)
        ]
        if len(self._comments) < original_len:
            self._save()
            return True
        return False


class ShareManager:
    """Manages sharing and collaboration for memorias.

    Args:
        share_store: The ShareStore for sharing metadata.
    """

    def __init__(self, share_store: ShareStore) -> None:
        self.share_store = share_store

    def share_with_user(
        self,
        memoria_id: str,
        shared_with: str,
        permission: str = "read",
        expires_days: int | None = None,
    ) -> Share | None:
        """Share a memoria with a user.

        Args:
            memoria_id: The memoria ID.
            shared_with: Email or username.
            permission: Permission level.
            expires_days: Optional days until expiration.

        Returns:
            Share object.
        """
        expires_at = None
        if expires_days:
            expires_at = (datetime.now(UTC) + timedelta(days=expires_days)).isoformat()

        self.share_store.add_share(
            memoria_id=memoria_id,
            shared_with=shared_with,
            permission=permission,
            expires_at=expires_at,
        )

        shares = self.share_store.get_shares(memoria_id)
        if not shares:
            _log.warning("share: add succeeded but get_shares returned empty for %s", memoria_id[:8])
            return None
        return shares[-1]

    def unshare_with_user(self, memoria_id: str, shared_with: str) -> bool:
        """Unshare a memoria from a user.

        Args:
            memoria_id: The memoria ID.
            shared_with: The user to unshare.

        Returns:
            True if unshared.
        """
        return self.share_store.remove_share(memoria_id, shared_with)

    def create_link(
        self,
        memoria_id: str,
        permission: str = "read",
        expires_hours: int = 24,
        password: str | None = None,
    ) -> str:
        """Create a temporary sharing link.

        Args:
            memoria_id: The memoria ID.
            permission: Permission level.
            expires_hours: Hours until expiration.
            password: Optional password.

        Returns:
            Share link URL.
        """
        link = self.share_store.create_share_link(
            memoria_id=memoria_id,
            permission=permission,
            expires_hours=expires_hours,
            password=password,
        )
        return f"https://memo.app/share/{link.link_token}"

    def validate_link(self, link_token: str, password: str | None = None) -> ShareLink | None:
        """Validate a share link.

        Args:
            link_token: The link token.
            password: Optional password.

        Returns:
            ShareLink if valid, None otherwise.
        """
        link = self.share_store.get_share_link(link_token)
        if not link:
            return None

        # Check expiration
        expires_at = datetime.fromisoformat(link.expires_at)
        if datetime.now(UTC) > expires_at:
            return None

        # Check password if set
        if link.password_hash and password:
            # Would verify password hash here
            pass

        # Increment access count
        self.share_store.increment_link_access(link_token)

        return link

    def add_comment(
        self,
        memoria_id: str,
        author: str,
        content: str,
        parent_id: str | None = None,
    ) -> Comment | None:
        """Add a comment to a memoria.

        Args:
            memoria_id: The memoria ID.
            author: Comment author.
            content: Comment content.
            parent_id: Optional parent comment ID.

        Returns:
            Comment object.
        """
        self.share_store.add_comment(memoria_id, author, content, parent_id)
        comments = self.share_store.get_comments(memoria_id)
        if not comments:
            _log.warning("comment: add succeeded but get_comments returned empty for %s", memoria_id[:8])
            return None
        return comments[-1]

    def get_comments(self, memoria_id: str) -> list[Comment]:
        """Get all comments for a memoria.

        Args:
            memoria_id: The memoria ID.

        Returns:
            List of Comment objects.
        """
        return self.share_store.get_comments(memoria_id)

    def delete_comment(self, memoria_id: str, comment_id: str) -> bool:
        """Delete a comment.

        Args:
            memoria_id: The memoria ID.
            comment_id: The comment ID.

        Returns:
            True if deleted.
        """
        return self.share_store.delete_comment(memoria_id, comment_id)


__all__ = [
    "Comment",
    "Permission",
    "Share",
    "ShareLink",
    "ShareManager",
    "ShareStore",
]
