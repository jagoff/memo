from __future__ import annotations

import zipfile
from pathlib import Path

from click.testing import CliRunner

from memo.cli_memory import restore


def test_restore_portable_backup_writes_to_memory_dir_when_memories_live_in_vault(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    vault = tmp_path / "vault"
    memory_dir = vault / "Obsidian" / "AI" / "memory"
    data_dir.mkdir()
    state_dir.mkdir()
    memory_dir.mkdir(parents=True)

    backup = tmp_path / "memo-backup.zip"
    with zipfile.ZipFile(backup, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", "{}")
        zf.writestr("memory/project/example.md", "restored body")

    env = {
        "MEMO_DATA_DIR": str(data_dir),
        "MEMO_STATE_DIR": str(state_dir),
        "MEMO_VAULT_PATH": str(vault),
        "MEMO_MEMORIES_IN_VAULT": "1",
        "MEMO_NONINTERACTIVE": "1",
    }

    result = CliRunner().invoke(restore, [str(backup), "--yes"], env=env)

    assert result.exit_code == 0, result.output
    assert (memory_dir / "project" / "example.md").read_text(encoding="utf-8") == "restored body"
    assert not (data_dir / "project" / "example.md").exists()
