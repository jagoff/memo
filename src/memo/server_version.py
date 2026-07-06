"""MCP tools — versioning domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from fastmcp import FastMCP

from memo.memory import Memory
from memo.server_annotations import DESTRUCTIVE, READ_ONLY, annotated_tool


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_version_history(
        memory_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Show version history for a memory.

        Returns a list of all versions of the specified memory,
        most recent first. Each version includes title, type, tags,
        body snapshot, and change reason.

        Args:
            memory_id: The memory ID to get history for.
            limit: Maximum versions to return.
        """
        versions = memory.versioning.get_version_history(memory_id, limit=limit)
        return [dataclasses.asdict(v) for v in versions]

    @annotated_tool(server, **READ_ONLY)
    def memo_version_diff(
        memory_id: str,
        version_a: int | None = None,
        version_b: int | None = None,
    ) -> dict[str, Any] | None:
        """Show diff between two versions of a memory.

        Returns a unified diff between two versions. If version_a or version_b
        is None, uses the latest and latest-1 versions respectively.

        Args:
            memory_id: The memory ID to diff.
            version_a: First version ID (or None for latest).
            version_b: Second version ID (or None for latest-1).
        """
        diff = memory.versioning.diff_versions(memory_id, version_a, version_b)
        return dataclasses.asdict(diff) if diff else None

    @annotated_tool(server, **DESTRUCTIVE)
    def memo_version_rollback(
        memory_id: str,
        version_id: int,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Rollback a memory to a previous version.

        Restores the memory to the specified version by updating its
        content, title, type, and tags to match the version snapshot.

        Args:
            memory_id: The memory ID to rollback.
            version_id: The version ID to rollback to.
            reason: Optional reason for the rollback.
        """
        success = memory.versioning.rollback_to_version(memory_id, version_id, reason)
        return {"success": success, "memory_id": memory_id, "version_id": version_id}
