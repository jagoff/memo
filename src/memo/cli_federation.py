"""CLI for signed, ACL-aware Memo federation bundles."""

from __future__ import annotations

import json
import stat
from functools import wraps
from pathlib import Path
from typing import Any

import click

from memo.config import Config
from memo.errors import FederationError


def _with_memory(fn: Any) -> Any:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        from memo.memory import Memory

        memory = Memory(Config.from_env())
        try:
            return fn(memory, *args, **kwargs)
        finally:
            memory.close()

    return wrapper


def _json(value: Any) -> None:
    click.echo(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _read_key(path: Path) -> bytes:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise FederationError(f"unsafe federation key file: {path}")
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise FederationError("federation key file must not be group/world accessible")
    key = path.read_bytes().strip()
    if len(key) < 16:
        raise FederationError("federation key must be at least 16 bytes")
    return key


@click.group(name="federation")
def federation_group() -> None:
    """Exchange signed memory subsets with explicit ACLs."""


@federation_group.command(name="preview")
@click.option("--principal", required=True)
@click.option("--owner", "owner_principal", default=None)
@_with_memory
def federation_preview(
    memory: Any,
    principal: str,
    owner_principal: str | None,
) -> None:
    _json(
        memory.federation.preview(
            principal=principal,
            owner_principal=owner_principal,
        )
    )


@federation_group.command(name="export")
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--principal", required=True)
@click.option("--owner", "owner_principal", default=None)
@click.option("--key-file", required=True, type=click.Path(path_type=Path))
@_with_memory
def federation_export(
    memory: Any,
    output_path: Path,
    principal: str,
    owner_principal: str | None,
    key_file: Path,
) -> None:
    _json(
        memory.federation.export_bundle(
            output_path,
            principal=principal,
            owner_principal=owner_principal,
            signing_key=_read_key(key_file),
        )
    )


@federation_group.command(name="verify")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("--principal", default=None)
@click.option("--key-file", required=True, type=click.Path(path_type=Path))
def federation_verify(
    input_path: Path,
    principal: str | None,
    key_file: Path,
) -> None:
    from memo.federation import FederationManager

    bundle = FederationManager.verify_bundle(
        input_path,
        principal=principal,
        signing_key=_read_key(key_file),
    )
    _json(
        {
            "verified": True,
            "schema": bundle["schema"],
            "bundle_id": bundle["bundle_id"],
            "principal": bundle["principal"],
            "source_device": bundle["source_device"],
            "memories": len(bundle["memories"]),
            "operations": len(bundle["operations"]),
            "key_id": bundle["signature"]["key_id"],
        }
    )


@federation_group.command(name="import")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("--principal", required=True)
@click.option("--key-file", required=True, type=click.Path(path_type=Path))
@click.option("--trust-peer", is_flag=True)
@click.option("--dry-run", is_flag=True)
@_with_memory
def federation_import(
    memory: Any,
    input_path: Path,
    principal: str,
    key_file: Path,
    trust_peer: bool,
    dry_run: bool,
) -> None:
    _json(
        memory.federation.import_bundle(
            input_path,
            principal=principal,
            signing_key=_read_key(key_file),
            trust_peer=trust_peer,
            dry_run=dry_run,
        )
    )


__all__ = ["federation_group"]
