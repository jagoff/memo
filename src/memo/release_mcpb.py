"""Deterministic MCPB archive construction for release tooling."""

from __future__ import annotations

import os
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


def _build_zip(source_pairs: Sequence[tuple[str, Path]], destination: Path) -> Path:
    """Write members to destination as a ZIP with stable metadata, atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")

    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            for member, source in source_pairs:
                info = zipfile.ZipInfo(member, date_time=_ZIP_EPOCH)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = _REGULAR_FILE_MODE
                archive.writestr(info, source.read_bytes())
        os.replace(temporary, destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise

    return destination


def build_mcpb(repo: Path, output: Path | None = None) -> Path:
    """Build the tracked MCPB from source members with stable ZIP metadata."""
    source_dir = repo / "packaging" / "mcpb"
    destination = output or repo / "packaging" / "memo.mcpb"
    return _build_zip([(member, source_dir / member) for member in MCPB_MEMBERS], destination)


def build_mcpb_node(repo: Path, output: Path | None = None) -> Path:
    """Build the Node MCPB; members missing in mcpb-node/ fall back to mcpb/ (icon)."""
    source_dir = repo / "packaging" / "mcpb-node"
    fallback_dir = repo / "packaging" / "mcpb"
    destination = output or repo / "packaging" / "memo-node.mcpb"
    pairs = []
    for member in MCPB_NODE_MEMBERS:
        source = source_dir / member
        if not source.exists():
            source = fallback_dir / member
        pairs.append((member, source))
    return _build_zip(pairs, destination)
