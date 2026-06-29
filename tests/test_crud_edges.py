"""Edge-case CRUD behaviours: exit codes + tombstone-aware id resolution."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from memo.cli import cli
from memo.config import Config


# find_by_prefix excludes soft-deleted tombstones
def test_find_by_prefix_excludes_soft_deleted(tmp_cfg: Config) -> None:
    from memo.memory.facade import Memory

    mem = Memory(tmp_cfg)
    try:
        rec = mem.save(content="x", title="A", auto_project=False, defer_embed=True)
        full = rec.id
        mem.delete(full)  # soft-delete (default)
        # prefix must no longer resolve to the tombstone
        assert full not in mem.store.find_by_prefix(full[:8])
    finally:
        mem.close()


# delete of a missing id exits non-zero (parity with get/update)
def test_delete_missing_id_exits_nonzero(tmp_path: Path) -> None:
    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }
    result = CliRunner().invoke(cli, ["delete", "deadbeef", "--yes"], env=env)
    assert result.exit_code == 1, result.output
    assert "not found" in result.output


# save of empty content yields a clean error, not a raw ValueError traceback
def test_save_empty_content_clean_error(tmp_path: Path) -> None:
    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }
    result = CliRunner().invoke(cli, ["save", ""], env=env)
    assert result.exit_code != 0
    assert "non-empty" in result.output
    assert not isinstance(result.exception, ValueError)
