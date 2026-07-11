"""First-run setup helpers — vault detection, picker, and config I/O.

Public API used by `memo.cli` (first-run gate, `memo init`, `memo migrate-vault`):

- `detect_obsidian_vaults()` — read Obsidian's macOS registry.
- `run_picker()` — interactive TUI; returns chosen `(data_dir, vault_path | None)`.
- `load_config_file()` / `write_config_file()` — TOML round-trip at `~/.config/memo/config.toml`.
- `snapshot_config_file()` — reversible config snapshot before migrations.
"""

from memo.config_md import (
    config_dir,
    config_home,
)
from memo.config_md import (
    index_path as markdown_config_index_path,
)
from memo.config_md import (
    write_default_config as write_markdown_config,
)
from memo.setup.config_io import (
    CONFIG_FILE_PATH,
    load_config_file,
    snapshot_config_file,
    write_config_file,
)
from memo.setup.picker import PickerResult, run_picker
from memo.setup.vaults import VaultInfo, detect_obsidian_vaults

__all__ = [
    "CONFIG_FILE_PATH",
    "PickerResult",
    "VaultInfo",
    "config_dir",
    "config_home",
    "detect_obsidian_vaults",
    "load_config_file",
    "markdown_config_index_path",
    "run_picker",
    "snapshot_config_file",
    "write_config_file",
    "write_markdown_config",
]
