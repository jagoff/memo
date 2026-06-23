"""MCP tools — sync domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from memo.memory import Memory


def _resolve_remote_history_db(remote: str | None) -> Path | None:
    """Map a ``--remote`` arg to the remote machine's ``history.db``.

    Accepts either a direct path to a ``.db`` file or a memo state dir that
    contains ``history.db``. The replay sync model reads the remote audit log,
    not a remote vault snapshot.
    """
    if not remote:
        return None
    p = Path(remote)
    return p if p.suffix == ".db" else p / "history.db"


def register(server: FastMCP, memory: Memory) -> None:
    @server.tool()
    def memo_sync_diff(
        remote: str | None = None,
    ) -> dict[str, Any]:
        """Not supported in the replay sync model.

        Sync replays the remote audit log into the local store; there is no
        precomputed file diff. Use ``memo_sync_pull`` to apply remote events.

        Args:
            remote: Path to remote memo state dir (unused).
        """
        return {
            "error": "replay sync model has no precomputed diff; use memo_sync_pull",
        }

    @server.tool()
    def memo_sync_push(
        remote: str | None = None,
    ) -> dict[str, Any]:
        """Not supported in the replay sync model.

        Sync is pull-only: each machine replays the other's audit log locally.
        To propagate local changes, the remote machine pulls from this one.

        Args:
            remote: Path to remote memo state dir (unused).
        """
        return {
            "error": "replay sync model is pull-only; the remote machine pulls instead",
        }

    @server.tool()
    def memo_sync_pull(
        remote: str | None = None,
    ) -> dict[str, Any]:
        """Pull remote changes by replaying the remote audit log.

        Applies events missing from this machine that exist in the remote
        ``history.db``. Returns counts of applied / conflicting / errored events.

        Args:
            remote: Path to remote memo state dir (or its ``history.db``).
        """
        remote_db = _resolve_remote_history_db(remote)
        if remote_db is None:
            return {"error": "remote is required (path to remote memo state dir)"}
        diff = memory.sync.sync_from_remote(remote_db)
        return diff.__dict__

    @server.tool()
    def memo_sync_both(
        remote: str | None = None,
    ) -> dict[str, Any]:
        """Sync from a remote machine (replay model alias for pull).

        In the replay model "both directions" is achieved by each machine
        pulling the other's audit log; from this side that is a pull.

        Args:
            remote: Path to remote memo state dir (or its ``history.db``).
        """
        remote_db = _resolve_remote_history_db(remote)
        if remote_db is None:
            return {"error": "remote is required (path to remote memo state dir)"}
        diff = memory.sync.sync_from_remote(remote_db)
        return diff.__dict__
