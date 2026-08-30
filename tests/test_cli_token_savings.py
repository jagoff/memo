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
                "recall_format_compact": {
                    "saved_frac": 0.31,
                    "quality_delta": 0.0,
                    "passed": True,
                    "n_samples": 49,
                },
                "verbosity_steer_L2": {
                    "saved_frac": -0.12,
                    "quality_delta": 0.0,
                    "passed": False,
                    "n_samples": 49,
                },
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


# Every fixture above writes a baseline entry with no sample size, because
# `gate_metrics` never recorded one — so the command could not tell a lever
# measured over 49 live prompts from one measured over 3 synthetic cases, and
# printed both as "measured, gate-passed". Live example (2026-08-30):
# `crusher_L1 +44.4% (measured, gate-passed)` rested on the 3 cases in
# `eval/token_corpus.json`, whose own `_doc` predicts the quality guard should
# fail — and `memo eval tokens --gate` did exit 1 on it while the headline
# still read as a supported claim.


def test_a_lever_measured_on_a_toy_sample_is_not_published_as_a_headline(tmp_path):
    eval_dir = tmp_path / "state" / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "token_baseline.json").write_text(
        json.dumps(
            {
                "crusher_L1": {
                    "saved_frac": 0.4443,
                    "quality_delta": 0.0,
                    "passed": True,
                    "n_samples": 3,
                }
            }
        )
    )
    r = CliRunner().invoke(cli, ["token-savings"], env=_env(tmp_path))
    assert r.exit_code == 0, r.output
    assert "+44.4%" not in r.output, "a 3-sample result is printed as a supported claim"
    assert "3 sample" in r.output, "the sample it rests on is not disclosed"


def test_a_published_lever_states_the_sample_it_rests_on(tmp_path):
    eval_dir = tmp_path / "state" / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "token_baseline.json").write_text(
        json.dumps(
            {
                "recall_format_compact": {
                    "saved_frac": 0.31,
                    "quality_delta": 0.0,
                    "passed": True,
                    "n_samples": 49,
                }
            }
        )
    )
    r = CliRunner().invoke(cli, ["token-savings"], env=_env(tmp_path))
    assert r.exit_code == 0, r.output
    assert "+31.0%" in r.output
    assert "49 sample" in r.output


def test_a_baseline_predating_sample_recording_is_not_published(tmp_path):
    """No `n_samples` means the number's basis is unknown, not large."""
    eval_dir = tmp_path / "state" / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "token_baseline.json").write_text(
        json.dumps({"crusher_L1": {"saved_frac": 0.4443, "quality_delta": 0.0, "passed": True}})
    )
    r = CliRunner().invoke(cli, ["token-savings"], env=_env(tmp_path))
    assert r.exit_code == 0, r.output
    assert "+44.4%" not in r.output
    assert "memo eval tokens --update-baseline" in r.output
