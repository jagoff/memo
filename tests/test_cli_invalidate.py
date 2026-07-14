"""C5: `memo invalidate <pattern> --reason` — reversible bulk weakening."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from memo.cli import cli


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def test_invalidate_preview_mutates_nothing(mock_memory, tmp_path):
    r = mock_memory.save(content="usamos webpack para el bundling", title="Bundler decision")
    with patch("memo.cli_invalidate._get_memory", return_value=mock_memory):
        result = CliRunner().invoke(cli, ["invalidate", "webpack"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "Would weaken 1" in result.output
    rec = mock_memory.get(r.id)
    assert "_invalidated" not in rec.tags
    health = mock_memory.store.get_health_batch([r.id])
    assert (health.get(r.id) or {}).get("confidence", 1.0) == pytest.approx(1.0)


def test_invalidate_requires_reason_with_yes(mock_memory, tmp_path):
    with patch("memo.cli_invalidate._get_memory", return_value=mock_memory):
        result = CliRunner().invoke(cli, ["invalidate", "webpack", "--yes"], env=_env(tmp_path))
    assert result.exit_code != 0
    assert "--reason" in result.output


def test_invalidate_apply_and_undo_roundtrip(mock_memory, tmp_path):
    r1 = mock_memory.save(content="usamos webpack para el bundling", title="Bundler decision")
    r2 = mock_memory.save(content="preferencia: tabs no spaces", title="Style pref")
    env = _env(tmp_path)
    with patch("memo.cli_invalidate._get_memory", return_value=mock_memory):
        result = CliRunner().invoke(
            cli,
            ["invalidate", "webpack", "--reason", "migramos a vite", "--yes"],
            env=env,
        )
        assert result.exit_code == 0, result.output

        rec = mock_memory.get(r1.id)
        assert "_invalidated" in rec.tags
        assert rec.extra["invalidated_reason"] == "migramos a vite"
        conf = mock_memory.store.get_health_batch([r1.id])[r1.id]["confidence"]
        assert conf == pytest.approx(0.7)  # 1.0 - default penalty 0.3
        # non-matching memory untouched
        assert "_invalidated" not in mock_memory.get(r2.id).tags

        # receipt exists
        receipts = list((tmp_path / "state" / "invalidate").glob("*.json"))
        assert len(receipts) == 1

        # undo restores everything
        result = CliRunner().invoke(cli, ["invalidate", "--undo"], env=env)
        assert result.exit_code == 0, result.output
        rec = mock_memory.get(r1.id)
        assert "_invalidated" not in rec.tags
        assert "invalidated_reason" not in rec.extra
        conf = mock_memory.store.get_health_batch([r1.id])[r1.id]["confidence"]
        assert conf == pytest.approx(1.0)


def test_undo_without_receipts_errors(mock_memory, tmp_path):
    with patch("memo.cli_invalidate._get_memory", return_value=mock_memory):
        result = CliRunner().invoke(cli, ["invalidate", "--undo"], env=_env(tmp_path))
    assert result.exit_code != 0
    assert "No invalidate receipts" in result.output
