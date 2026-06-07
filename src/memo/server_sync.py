"""MCP tools — sync domain (split from server.py).

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
    def memory_sync_diff(
        remote: str | None = None,
    ) -> dict[str, Any]:
        """Compute diff between local and remote vaults.

        Computes the difference between local and remote vaults,
        identifying new, modified, deleted, and conflicted memorias.

        Args:
            remote: Path to remote vault (optional).
        """
        from pathlib import Path

        remote_path = Path(remote) if remote else None

        sync_mgr = memory.sync.__class__(memory, remote_path=remote_path)
        diff = sync_mgr.compute_diff()
        return diff.__dict__

    @server.tool()
    def memory_sync_push(
        remote: str | None = None,
    ) -> dict[str, Any]:
        """Push local changes to remote vault.

        Pushes modified and deleted memorias from local to remote vault.

        Args:
            remote: Path to remote vault (optional).
        """
        from pathlib import Path

        remote_path = Path(remote) if remote else None

        sync_mgr = memory.sync.__class__(memory, remote_path=remote_path)
        diff = sync_mgr.sync(direction="push")
        return diff.__dict__

    @server.tool()
    def memory_sync_pull(
        remote: str | None = None,
    ) -> dict[str, Any]:
        """Pull remote changes to local vault.

        Pulls new and modified memorias from remote to local vault.

        Args:
            remote: Path to remote vault (optional).
        """
        from pathlib import Path

        remote_path = Path(remote) if remote else None

        sync_mgr = memory.sync.__class__(memory, remote_path=remote_path)
        diff = sync_mgr.sync(direction="pull")
        return diff.__dict__

    @server.tool()
    def memory_sync_both(
        remote: str | None = None,
    ) -> dict[str, Any]:
        """Sync both directions (bidirectional).

        Performs bidirectional sync, pulling new changes from remote
        and pushing local changes to remote.

        Args:
            remote: Path to remote vault (optional).
        """
        from pathlib import Path

        remote_path = Path(remote) if remote else None

        sync_mgr = memory.sync.__class__(memory, remote_path=remote_path)
        diff = sync_mgr.sync(direction="both")
        return diff.__dict__

    # -- cache-tier tools ----------------------------------------------------------
    # Opt-in (MEMO_CACHE_MODE != off): memo as a bounded cache fronting an
    # authoritative backing store. These are store-management verbs (not the
    # "brain-like" cognition verbs blocked by test_architecture_boundaries),
    # so they belong on memo's MCP surface.
