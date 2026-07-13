"""Deterministic MCPB archive construction for release tooling."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

MCPB_MEMBERS: tuple[str, ...] = (
    "icon.png",
    "manifest.json",
    "server/main.py",
)

_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_REGULAR_FILE_MODE = 0o100644 << 16


def build_mcpb(repo: Path, output: Path | None = None) -> Path:
    """Build the tracked MCPB from source members with stable ZIP metadata."""
    source_dir = repo / "packaging" / "mcpb"
    destination = output or repo / "packaging" / "memo.mcpb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")

    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            for member in MCPB_MEMBERS:
                info = zipfile.ZipInfo(member, date_time=_ZIP_EPOCH)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = _REGULAR_FILE_MODE
                archive.writestr(info, (source_dir / member).read_bytes())
        os.replace(temporary, destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise

    return destination
