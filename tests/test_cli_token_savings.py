"""Tests for `memo token-savings` reporting measured per-lever savings."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from memo.cli import cli


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def test_reports_measured_savings_from_baseline(tmp_path):
    eval_dir = tmp_path / "state" / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "token_baseline.json").write_text(
        json.dumps(
            {
                "recall_format_compact": {"saved_frac": 0.31, "quality_delta": 0.0, "passed": True},
                "verbosity_steer_L2": {"saved_frac": -0.12, "quality_delta": 0.0, "passed": False},
            }
        )
    )
    r = CliRunner().invoke(cli, ["token-savings"], env=_env(tmp_path))
    assert r.exit_code == 0, r.output
    assert "recall_format_compact" in r.output
    assert "31" in r.output  # measured 31% saving surfaced
    assert "verbosity_steer_L2" not in r.output  # a non-passing lever is not claimed


def test_reports_unmeasured_when_no_baseline(tmp_path):
    r = CliRunner().invoke(cli, ["token-savings"], env=_env(tmp_path))
    assert r.exit_code == 0, r.output
    assert "memo eval tokens --update-baseline" in r.output
    assert "65%" not in r.output  # the fabricated estimate is gone
