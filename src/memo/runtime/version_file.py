"""Version file read/write for cross-machine auto-update."""
from __future__ import annotations

import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path


VERSION_FILE = "memo-version.json"


def read_version_file(sync_root: Path) -> dict | None:
    """Read memo-version.json from sync repo root.

    Returns None if file doesn't exist.
    """
    version_file = sync_root / VERSION_FILE
    if not version_file.is_file():
        return None
    try:
        return json.loads(version_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_version_file(sync_root: Path, version: str | None = None) -> dict:
    """Write memo-version.json to sync repo root.

    If version is None, reads current version from metadata.
    Returns the written dict.
    """
    if version is None:
        version = importlib.metadata.version("mlx-memo")

    content = {
        "version": version,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    version_file = sync_root / VERSION_FILE
    version_file.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    return content