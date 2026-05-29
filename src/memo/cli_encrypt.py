"""`memo encrypt` command group — at-rest encryption lock/unlock.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(encrypt_group)`.
"""

from __future__ import annotations

import click

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config

# -- encryption commands ----------------------------------------------------------


@click.group(name="encrypt")
def encrypt_group() -> None:
    """Memory encryption — encrypt sensitive memorias."""
    pass


@encrypt_group.command(name="unlock")
@click.argument("password")
def encrypt_unlock(password: str) -> None:
    """Unlock the vault with password.

    Derives master key from password and stores in memory.

    Example: memo encrypt unlock mypassword
    """
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
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    mem.encryption.lock()
    console.print("[green]Vault locked[/green]")


@encrypt_group.command(name="status")
def encrypt_status() -> None:
    """Check if vault is unlocked.

    Example: memo encrypt status
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    if mem.encryption.is_unlocked():
        console.print("[green]Vault is unlocked[/green]")
    else:
        console.print("[yellow]Vault is locked[/yellow]")
