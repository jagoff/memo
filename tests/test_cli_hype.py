"""Tests for `memo hype` commands."""

from __future__ import annotations

import json

from click.testing import CliRunner

from memo.cli import cli


def _env(tmp_path):
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
        "MEMO_SKIP_MODEL_VERSION_CHECK": "1",
    }


def test_hype_status_empty(tmp_path):
    """memo hype status on empty state → exit 0, zero stats."""
    (tmp_path / "data").mkdir()
    (tmp_path / "state").mkdir()
    res = CliRunner().invoke(cli, ["hype", "status"], env=_env(tmp_path))
    assert res.exit_code == 0, res.output
    assert "indexed" in res.output.lower() or "memor" in res.output.lower()


def test_hype_status_json_empty(tmp_path):
    """memo hype status --json on empty state → valid JSON with 0s."""
    (tmp_path / "data").mkdir()
    (tmp_path / "state").mkdir()
    res = CliRunner().invoke(cli, ["hype", "status", "--json"], env=_env(tmp_path))
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert "indexed_memories" in data
    assert "questions" in data
    assert "durable_total" in data
    assert "coverage_pct" in data
    assert "backlog" in data
    assert data["indexed_memories"] == 0
    assert data["questions"] == 0
    assert data["durable_total"] == 0
    assert data["coverage_pct"] == 0.0


def test_hype_status_help(tmp_path):
    """memo hype status --help works."""
    res = CliRunner().invoke(cli, ["hype", "status", "--help"])
    assert res.exit_code == 0
    assert "status" in res.output.lower() or "coverage" in res.output.lower()


def test_hype_help(tmp_path):
    """memo hype --help works."""
    res = CliRunner().invoke(cli, ["hype", "--help"])
    assert res.exit_code == 0
    assert "hype" in res.output.lower()
