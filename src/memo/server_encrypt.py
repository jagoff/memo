"""MCP tools — encryption domain (split from server.py).

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
    def memory_encrypt_unlock(
        password: str,
    ) -> dict[str, Any]:
        """Unlock the vault with password.

        Derives master key from password using PBKDF2 and stores it
        in memory for subsequent encryption/decryption operations.

        Args:
            password: User password for key derivation.
        """
        success = memory.encryption.unlock(password)
        return {"success": success, "status": "unlocked" if success else "failed"}

    @server.tool()
    def memory_encrypt_lock() -> dict[str, str]:
        """Lock the vault (clear master key from memory).

        Clears the master encryption key from memory, preventing
        further encryption/decryption operations until unlock() is called.
        """
        memory.encryption.lock()
        return {"status": "locked"}

    @server.tool()
    def memory_encrypt_status() -> dict[str, Any]:
        """Check if vault is unlocked.

        Returns the current lock status of the vault and whether
        encryption operations can be performed.
        """
        return {
            "is_unlocked": memory.encryption.is_unlocked(),
            "status": "unlocked" if memory.encryption.is_unlocked() else "locked",
        }
