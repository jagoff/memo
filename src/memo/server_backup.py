"""MCP tools — backup domain (split from server.py).

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
    def memo_backup_create(
        compress: bool = True,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Create a backup of the entire vault.

        Creates a compressed tar.gz archive of all memoria files and
        databases. Returns metadata including checksum and size.

        Args:
            compress: Whether to compress the backup.
            name: Optional backup name (defaults to timestamp).
        """
        metadata = memory.backup.create_backup(compress=compress, name=name)
        return metadata.__dict__

    @server.tool()
    def memo_backup_list() -> list[dict[str, Any]]:
        """List all available backups.

        Returns a list of all backup archives with their metadata
        including timestamp and size.
        """
        backups = memory.backup.list_backups()
        return [b.__dict__ for b in backups]

    @server.tool()
    def memo_backup_restore(
        backup_name: str,
        restore_memorias: bool = True,
        restore_dbs: bool = True,
    ) -> dict[str, Any]:
        """Restore from a backup.

        Restores memoria files and/or databases from a backup archive.

        Args:
            backup_name: Name of the backup to restore.
            restore_memorias: Whether to restore memoria files.
            restore_dbs: Whether to restore databases.
        """
        success = memory.backup.restore_backup(
            backup_name,
            restore_memorias=restore_memorias,
            restore_dbs=restore_dbs,
        )
        return {"success": success, "backup_name": backup_name}
