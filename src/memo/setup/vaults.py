"""Obsidian vault auto-detection.

Obsidian writes its known-vault registry to a per-OS location:
- macOS:   `~/Library/Application Support/obsidian/obsidian.json`
- Linux:   `~/.config/obsidian/obsidian.json`
- Windows: `%APPDATA%/obsidian/obsidian.json`

Schema:

```json
{
  "vaults": {
    "<hex-id>": {"path": "/abs/path/to/vault", "ts": 1776791665907, "open": true},
    ...
  }
}
```

`ts` is epoch milliseconds of last open. We sort by `ts` desc so the
most-recently-used vault appears first in the picker.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _default_obsidian_registry() -> Path:
    """Obsidian's known-vault registry path for the current OS."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "obsidian" / "obsidian.json"
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / "obsidian" / "obsidian.json"
    # Linux / other POSIX
    return Path.home() / ".config" / "obsidian" / "obsidian.json"


OBSIDIAN_REGISTRY = _default_obsidian_registry()


@dataclass(frozen=True)
class VaultInfo:
    """One detected Obsidian vault."""

    name: str  # basename of the vault path
    path: Path  # absolute path on disk
    last_opened_ms: int  # epoch ms; 0 if registry omits `ts`


def detect_obsidian_vaults(
    registry_path: Path | None = None,
) -> list[VaultInfo]:
    """Return Obsidian vaults registered in the per-OS registry.

    Sorted by `last_opened_ms` desc. Skips entries whose path no longer
    exists on disk. Returns `[]` when the registry file is missing
    (Obsidian not installed) or unparseable.
    """
    p = registry_path or OBSIDIAN_REGISTRY
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    vaults_dict = data.get("vaults") or {}
    if not isinstance(vaults_dict, dict):
        return []
    out: list[VaultInfo] = []
    for entry in vaults_dict.values():
        if not isinstance(entry, dict):
            continue
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        path = Path(raw_path).expanduser()
        if not path.is_dir():
            continue
        ts = entry.get("ts")
        last_opened = int(ts) if isinstance(ts, int | float) else 0
        out.append(VaultInfo(name=path.name, path=path, last_opened_ms=last_opened))
    out.sort(key=lambda v: v.last_opened_ms, reverse=True)
    return out
