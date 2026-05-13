"""First-run setup helpers — Obsidian vault detection, interactive picker, TOML config I/O.

Public API used by `memo.cli` (first-run gate, `memo init`, `memo migrate-vault`):

- `detect_obsidian_vaults()` — read Obsidian's macOS registry.
- `run_picker()` — interactive TUI; returns chosen `(data_dir, vault_path | None)`.
- `load_config_file()` / `write_config_file()` — TOML round-trip at `~/.config/memo/config.toml`.
"""

from memo.setup.config_io import (
    CONFIG_FILE_PATH,
    load_config_file,
    write_config_file,
)
from memo.setup.picker import PickerResult, run_picker
from memo.setup.vaults import VaultInfo, detect_obsidian_vaults

__all__ = [
    "CONFIG_FILE_PATH",
    "PickerResult",
    "VaultInfo",
    "detect_obsidian_vaults",
    "load_config_file",
    "run_picker",
    "write_config_file",
]
