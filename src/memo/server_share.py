"""MCP tools — sharing domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from memo.memory import Memory


def register(server: FastMCP, memory: Memory) -> None:
    @server.tool()
    def memo_share_with_user(
        memoria_id: str,
        shared_with: str,
        permission: str = "read",
        expires_days: int | None = None,
    ) -> dict[str, Any]:
        """Share a memoria with a user.

        Shares a memoria with a specific user (by email or username)
        with the specified permission level. Optionally expires after
        a number of days.

        Args:
            memoria_id: The memoria ID to share.
            shared_with: Email or username to share with.
            permission: Permission level (read, comment, edit, admin).
            expires_days: Optional days until expiration.
        """
        share = memory.sharing.share_with_user(
            memoria_id=memoria_id,
            shared_with=shared_with,
            permission=permission,
            expires_days=expires_days,
        )
        return share.__dict__

    @server.tool()
    def memo_share_unshare(
        memoria_id: str,
        shared_with: str,
    ) -> dict[str, Any]:
        """Unshare a memoria from a user.

        Removes a share for a specific user.

        Args:
            memoria_id: The memoria ID.
            shared_with: The user to unshare.
        """
        success = memory.sharing.unshare_with_user(memoria_id, shared_with)
        return {"success": success, "memoria_id": memoria_id, "shared_with": shared_with}

    @server.tool()
    def memo_share_create_link(
        memoria_id: str,
        permission: str = "read",
        expires_hours: int = 24,
        password: str | None = None,
    ) -> dict[str, str]:
        """Create a temporary sharing link.

        Creates a temporary share link with optional password protection
        and expiration time.

        Args:
            memoria_id: The memoria ID.
            permission: Permission level.
            expires_hours: Hours until expiration.
            password: Optional password.
        """
        link = memory.sharing.create_link(
            memoria_id=memoria_id,
            permission=permission,
            expires_hours=expires_hours,
            password=password,
        )
        return {"link": link, "memoria_id": memoria_id}

    @server.tool()
    def memo_share_list(
        memoria_id: str,
    ) -> list[dict[str, Any]]:
        """List all shares for a memoria.

        Returns all shares for the specified memoria with their
        permission levels and expiration dates.

        Args:
            memoria_id: The memoria ID.
        """
        shares = memory.sharing.share_store.get_shares(memoria_id)
        return [s.__dict__ for s in shares]

    @server.tool()
    def memo_share_comment(
        memoria_id: str,
        content: str,
        author: str = "user",
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        """Add a comment to a memoria.

        Adds a comment to a memoria, optionally as a reply to another comment.

        Args:
            memoria_id: The memoria ID.
            content: Comment content.
            author: Comment author.
            parent_id: Optional parent comment ID.
        """
        comment = memory.sharing.add_comment(
            memoria_id=memoria_id,
            author=author,
            content=content,
            parent_id=parent_id,
        )
        return comment.__dict__

    @server.tool()
    def memo_share_comments(
        memoria_id: str,
    ) -> list[dict[str, Any]]:
        """List all comments for a memoria.

        Returns all comments for the specified memoria, including
        reply threads.

        Args:
            memoria_id: The memoria ID.
        """
        comments = memory.sharing.get_comments(memoria_id)
        return [c.__dict__ for c in comments]
