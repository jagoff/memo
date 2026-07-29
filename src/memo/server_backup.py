"""MCP tools — backup domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from fastmcp import Context, FastMCP

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
    async def memo_backup_restore(
        backup_name: str,
        restore_memories: bool = True,
        restore_dbs: bool = True,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Restore from a backup.

        Restores memory files and/or databases from a backup archive,
        overwriting the current store. Irreversible: elicitation-capable
        clients are asked to confirm first.

        Args:
            backup_name: Name of the backup to restore.
            restore_memories: Whether to restore memory files.
            restore_dbs: Whether to restore databases.
        """
        from memo.server_elicit import abort_result, confirm_destructive, sanitize_fragment

        safe_name = sanitize_fragment(backup_name)
        try:
            scope = f"the current store ({memory.store.count()} memories)"
        except Exception:
            scope = "the current store"
        gate = await confirm_destructive(
            ctx,
            action="restore",
            detail=(
                f"Restore backup '{safe_name}'? This overwrites {scope}; "
                "the rollback journal is deleted on success."
            ),
        )
        if not gate.proceed:
            return abort_result(
                gate,
                memory,
                tool="memo_backup_restore",
                action="restore",
                target=f"backup '{safe_name}'",
            )
        success = memory.backup.restore_backup(
            backup_name,
            restore_memories=restore_memories,
            restore_dbs=restore_dbs,
        )
        return {"success": success, "backup_name": backup_name}
