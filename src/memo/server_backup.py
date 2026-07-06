"""MCP tools — backup domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from fastmcp import FastMCP

from memo.memory import Memory
from memo.server_annotations import DESTRUCTIVE, READ_ONLY, WRITE_IDEMPOTENT, annotated_tool


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_backup_create(
        compress: bool = True,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Create a backup of the entire vault.

        Creates a compressed tar.gz archive of all memory files and
        databases. Returns metadata including checksum and size.

        Args:
            compress: Whether to compress the backup.
            name: Optional backup name (defaults to timestamp).
        """
        metadata = memory.backup.create_backup(compress=compress, name=name)
        return dataclasses.asdict(metadata)

    @annotated_tool(server, **READ_ONLY)
    def memo_backup_list() -> list[dict[str, Any]]:
        """List all available backups.

        Returns a list of all backup archives with their metadata
        including timestamp and size.
        """
        backups = memory.backup.list_backups()
        return [dataclasses.asdict(b) for b in backups]

    @annotated_tool(server, **DESTRUCTIVE)
    def memo_backup_restore(
        backup_name: str,
        restore_memories: bool = True,
        restore_dbs: bool = True,
    ) -> dict[str, Any]:
        """Restore from a backup.

        Restores memory files and/or databases from a backup archive.

        Args:
            backup_name: Name of the backup to restore.
            restore_memories: Whether to restore memory files.
            restore_dbs: Whether to restore databases.
        """
        success = memory.backup.restore_backup(
            backup_name,
            restore_memories=restore_memories,
            restore_dbs=restore_dbs,
        )
        return {"success": success, "backup_name": backup_name}
