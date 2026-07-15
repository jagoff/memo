"""Deterministic MCPB archive construction for release tooling."""

from __future__ import annotations

import os
import stat
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

MCPB_MEMBERS: tuple[str, ...] = (
    "icon.png",
    "manifest.json",
    "server/main.py",
)

MCPB_NODE_MEMBERS: tuple[str, ...] = (
    "icon.png",
    "manifest.json",
    "bootstrap.js",
)

_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_REGULAR_FILE_MODE = 0o100644 << 16


def _validated_source(root: Path, member: str) -> Path:
    """Return a regular in-root source without following symlink components."""
    if root.is_symlink():
        raise ValueError(f"MCPB source root cannot be a symlink: {root}")
    source = root / member
    cursor = root
    for part in Path(member).parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"MCPB source member cannot be a symlink: {source}")
    resolved_root = root.resolve(strict=True)
    resolved_source = source.resolve(strict=True)
    try:
        resolved_source.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"MCPB source member escapes its root: {source}") from exc
    if not resolved_source.is_file():
        raise ValueError(f"MCPB source member must be a regular file: {source}")
    return source


def _read_regular_source(source: Path) -> bytes:
    """Read a validated source through a no-follow descriptor when supported."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"MCPB source member must be a regular file: {source}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _build_zip(source_pairs: Sequence[tuple[str, Path]], destination: Path) -> Path:
    """Write members to destination as a ZIP with stable metadata, atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)

    try:
        with os.fdopen(descriptor, "w+b") as temporary_file:
            descriptor = -1
            with zipfile.ZipFile(temporary_file, "w") as archive:
                for member, source in source_pairs:
                    info = zipfile.ZipInfo(member, date_time=_ZIP_EPOCH)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.create_system = 3
                    info.external_attr = _REGULAR_FILE_MODE
                    archive.writestr(info, _read_regular_source(source))
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary, destination)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)

    return destination


def build_mcpb(repo: Path, output: Path | None = None) -> Path:
    """Build the tracked MCPB from source members with stable ZIP metadata."""
    source_dir = repo / "packaging" / "mcpb"
    destination = output or repo / "packaging" / "memo.mcpb"
    return _build_zip(
        [(member, _validated_source(source_dir, member)) for member in MCPB_MEMBERS],
        destination,
    )


def build_mcpb_node(repo: Path, output: Path | None = None) -> Path:
    """Build the Node MCPB; members missing in mcpb-node/ fall back to mcpb/ (icon)."""
    source_dir = repo / "packaging" / "mcpb-node"
    fallback_dir = repo / "packaging" / "mcpb"
    destination = output or repo / "packaging" / "memo-node.mcpb"
    pairs = []
    for member in MCPB_NODE_MEMBERS:
        source = source_dir / member
        root = source_dir
        if source.is_symlink() or source.exists():
            source = _validated_source(root, member)
        else:
            root = fallback_dir
            source = _validated_source(root, member)
        pairs.append((member, source))
    return _build_zip(pairs, destination)
