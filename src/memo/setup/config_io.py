"""TOML config I/O at `~/.config/memo/config.toml`.

File shape:

```toml
[storage]
data_dir = "/Users/you/Documents/memo"
vault_path = "/Users/you/Library/.../Notes"  # optional, only when Obsidian was picked
```

Read uses stdlib `tomllib` (Python ≥3.11; project requires 3.13).
Write uses `tomli_w` — a tiny dep that's the natural pair for `tomllib`.

Override the file location via `MEMO_CONFIG_FILE` for tests / CI.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

import tomli_w


def _resolve_config_path() -> Path:
    """Resolve `MEMO_CONFIG_FILE` env var if set, else default."""
    override = os.environ.get("MEMO_CONFIG_FILE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "memo" / "config.toml"


# NB: we resolve once at import-time. Tests that need a different
# location must set `MEMO_CONFIG_FILE` before importing this module
# (the `tmp_cfg` test fixture does this) OR pass an explicit `path=`
# argument to the load/write helpers.
CONFIG_FILE_PATH: Path = _resolve_config_path()


def load_config_file(path: Path | None = None) -> dict[str, Any] | None:
    """Read TOML config. Returns parsed dict, or None if missing/unreadable.

    Never raises on parse errors — the caller cannot depend on a config
    file existing, so a corrupt file is treated as absent. Log to stderr
    so the user knows.
    """
    p = path or _resolve_config_path()
    if not p.is_file():
        return None
    try:
        return tomllib.loads(p.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        import sys

        print(f"[memo] warning: failed to parse {p}: {exc}", file=sys.stderr)
        return None


def write_config_file(
    *,
    data_dir: Path,
    vault_path: Path | None = None,
    memories_in_vault: bool = False,
    single_db: bool = False,
    path: Path | None = None,
) -> Path:
    """Write `[storage]` section atomically. Creates parent dirs.

    Returns the path written.
    """
    p = path or _resolve_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    storage: dict[str, Any] = {"data_dir": str(data_dir)}
    if vault_path is not None:
        storage["vault_path"] = str(vault_path)
    if memories_in_vault:
        storage["memories_in_vault"] = True
    if single_db:
        storage["single_db"] = True
    payload = tomli_w.dumps({"storage": storage}).encode("utf-8")
    # Write to a sibling temp then rename for atomicity. Avoids a half-
    # written file if the process is killed mid-write.
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(p)
    return p
