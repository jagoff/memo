"""`memo dream tune` CLI — status/rollback paths (no MLX load)."""

from __future__ import annotations

import json

from click.testing import CliRunner

from memo.cli import cli


def _env(tmp_path):
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "d"),
        "MEMO_STATE_DIR": str(tmp_path / "s"),
    }


def test_dream_tune_status_empty(tmp_path):
    r = CliRunner().invoke(cli, ["dream", "tune", "--status"], env=_env(tmp_path))
    assert r.exit_code == 0, r.output
    payload = json.loads(r.output)
    assert payload["overlay"] == {}
    assert payload["baseline"] is None


def test_dream_tune_rollback_noop(tmp_path):
    r = CliRunner().invoke(cli, ["dream", "tune", "--rollback"], env=_env(tmp_path))
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["rolled_back"] is None
