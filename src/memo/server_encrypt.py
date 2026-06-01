"""MCP tools — encryption domain (split from server.py).

Registered by `build_server()` via `register(server, memory)`. Tool names,
signatures, defaults, docstrings and bodies are identical to the originals;
only the enclosing function and indentation changed.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from memo.flags import flag_bool
from memo.memory import Memory

# Returned when the encryption vertical is gated off (default). Mirrors the CLI
# disabled message in cli_encrypt.py.
_DISABLED_MSG = "Encryption disabled (set MEMO_ENCRYPTION_ENABLED=1 to enable)."


def _disabled() -> dict[str, Any]:
    return {"ok": False, "status": "disabled", "error": _DISABLED_MSG}


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
        if not flag_bool("MEMO_ENCRYPTION_ENABLED"):
            return _disabled()
        success = memory.encryption.unlock(password)
        return {"success": success, "status": "unlocked" if success else "failed"}

    @server.tool()
    def memory_encrypt_lock() -> dict[str, Any]:
        """Lock the vault (clear master key from memory).

        Clears the master encryption key from memory, preventing
        further encryption/decryption operations until unlock() is called.
        """
        if not flag_bool("MEMO_ENCRYPTION_ENABLED"):
            return _disabled()
        memory.encryption.lock()
        return {"status": "locked"}

    @server.tool()
    def memory_encrypt_status() -> dict[str, Any]:
        """Check if vault is unlocked.

        Returns the current lock status of the vault and whether
        encryption operations can be performed.
        """
        if not flag_bool("MEMO_ENCRYPTION_ENABLED"):
            return _disabled()
        return {
            "is_unlocked": memory.encryption.is_unlocked(),
            "status": "unlocked" if memory.encryption.is_unlocked() else "locked",
        }
