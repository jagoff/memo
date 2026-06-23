"""`memo encrypt` command group — at-rest encryption lock/unlock.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(encrypt_group)`.
"""

from __future__ import annotations

import click

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config
from memo.flags import flag_bool

# Shown when the encryption vertical is gated off (default). Mirrors the MCP
# disabled payload in server_encrypt.py.
_DISABLED_MSG = "Encryption disabled (set MEMO_ENCRYPTION_ENABLED=1 to enable)."


def _require_enabled() -> None:
    """Refuse with a clear message + non-zero exit unless the flag is on."""
    if not flag_bool("MEMO_ENCRYPTION_ENABLED"):
        console.print(f"[yellow]{_DISABLED_MSG}[/yellow]")
        raise click.exceptions.Exit(1)


# -- encryption commands ----------------------------------------------------------


@click.group(name="encrypt")
def encrypt_group() -> None:
    """Memory encryption — encrypt sensitive memories."""
    pass


@encrypt_group.command(name="unlock")
@click.argument("password")
def encrypt_unlock(password: str) -> None:
    """Unlock the vault with password.

    Derives master key from password and stores in memory.

    Example: memo encrypt unlock mypassword
    """
    _require_enabled()
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    success = mem.encryption.unlock(password)

    if success:
        console.print("[green]Vault unlocked[/green]")
    else:
        console.print("[red]Failed to unlock vault[/red]")


@encrypt_group.command(name="lock")
def encrypt_lock() -> None:
    """Lock the vault (clear master key from memory).

    Example: memo encrypt lock
    """
    _require_enabled()
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    mem.encryption.lock()
    console.print("[green]Vault locked[/green]")


@encrypt_group.command(name="status")
def encrypt_status() -> None:
    """Check if vault is unlocked.

    Example: memo encrypt status
    """
    _require_enabled()
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    if mem.encryption.is_unlocked():
        console.print("[green]Vault is unlocked[/green]")
    else:
        console.print("[yellow]Vault is locked[/yellow]")
