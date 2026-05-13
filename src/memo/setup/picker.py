"""Interactive first-run picker — questionary-backed TUI.

Three branches:

1. **Standard macOS path** (default highlight): `~/Documents/memo/`. Visible in
   Finder, iCloud-syncable. Recommended for users who don't use Obsidian.
2. **An Obsidian vault** (one option per detected vault). Memorias land
   under `<vault>/<subdir>/`, subdir prompted with default `99-AI/memory`.
3. **Custom path…**: free-text absolute path.

Returns a `PickerResult` to the caller (CLI). `vault_path` is set only
when the user chose an Obsidian vault — it's used by the cross-vault
`memo ingest` command.

`questionary` is imported lazily so the rest of `memo` stays usable
even if the dep is unavailable (e.g. someone shaved the venv).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from memo.setup.vaults import VaultInfo, detect_obsidian_vaults


@dataclass(frozen=True)
class PickerResult:
    data_dir: Path
    vault_path: Path | None  # set only when an Obsidian vault was chosen


# Default subdir name when the user picks an Obsidian vault. `99-AI/`
# matches the user's existing system convention; nesting `memory/`
# under it keeps memo-managed files separate from other AI artefacts.
DEFAULT_VAULT_SUBDIR = "99-AI/memory"


def run_picker(
    *,
    default_data_dir: Path | None = None,
    default_vault_subdir: str = DEFAULT_VAULT_SUBDIR,
    detected_vaults: list[VaultInfo] | None = None,
) -> PickerResult:
    """Run the interactive picker. Blocks until the user chooses.

    `detected_vaults` is injectable for tests; in production it's
    auto-detected from the Obsidian registry.

    Raises `KeyboardInterrupt` if the user aborts (ESC / Ctrl-C). The
    caller is expected to catch and exit gracefully.
    """
    import questionary

    default_data_dir = (default_data_dir or Path.home() / "Documents" / "memo").expanduser()
    vaults = detected_vaults if detected_vaults is not None else detect_obsidian_vaults()

    std_label = f"Standard macOS path: {default_data_dir}  (recommended)"
    custom_label = "Custom path…"

    choices: list[str] = [std_label]
    vault_label_to_info: dict[str, VaultInfo] = {}
    for v in vaults:
        label = f"Obsidian vault: {v.name}  ({v.path})"
        choices.append(label)
        vault_label_to_info[label] = v
    choices.append(custom_label)

    answer = questionary.select(
        "Where should memo store your memorias?",
        choices=choices,
        default=std_label,
    ).ask()

    if answer is None:
        raise KeyboardInterrupt

    if answer == std_label:
        return PickerResult(data_dir=default_data_dir, vault_path=None)

    if answer == custom_label:
        custom = questionary.path(
            "Enter absolute path:",
            default=str(default_data_dir),
        ).ask()
        if not custom:
            raise KeyboardInterrupt
        return PickerResult(
            data_dir=Path(custom).expanduser().resolve(),
            vault_path=None,
        )

    chosen_vault = vault_label_to_info.get(answer)
    if chosen_vault is None:
        raise RuntimeError(f"unexpected picker answer: {answer!r}")

    subdir = questionary.text(
        f"Subfolder inside '{chosen_vault.name}' for memorias:",
        default=default_vault_subdir,
    ).ask()
    if subdir is None:
        raise KeyboardInterrupt
    subdir = subdir.strip().strip("/")
    if not subdir:
        # Empty subdir = vault root; allow it but warn the user is
        # pasting memorias next to their actual notes. We don't block.
        subdir = ""
    data_dir = chosen_vault.path / subdir if subdir else chosen_vault.path
    return PickerResult(data_dir=data_dir, vault_path=chosen_vault.path)
