"""Tests for `memo tokens` reporting the real proxy holdout measurement
instead of the retired hardcoded-constant estimate (spec finding 4: two
contradicting numbers on one screen — a measured cost beside a fabricated
"tokens saved" claim)."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from memo.cli_tokens import tokens_cmd
from memo.proxy import meter


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def _seed(state_dir: Path, records: list[meter.Record]) -> None:
    for r in records:
        meter.append(state_dir, r)


def test_no_data_says_so_instead_of_printing_a_zero(tmp_path):
    result = CliRunner().invoke(tokens_cmd, [], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "no measured data" in result.output.lower()


def test_the_estimated_panel_is_gone(tmp_path):
    result = CliRunner().invoke(tokens_cmd, [], env=_env(tmp_path))
    assert "estimated" not in result.output.lower()


def test_the_roi_constants_are_no_longer_registered():
    from memo.flags import REGISTRY

    assert "MEMO_ROI_TOKENS_PER_GROUNDED" not in REGISTRY
    assert "MEMO_ROI_TOKENS_PER_CONSULT" not in REGISTRY
    assert "MEMO_ROI_TOKENS_PER_REASK" not in REGISTRY


def test_measured_panel_reports_treated_vs_holdout_when_data_exists(tmp_path):
    state_dir = tmp_path / "state"
    for i in range(10):
        _seed(
            state_dir,
            [
                meter.Record(
                    request_key=f"t{i}",
                    holdout=False,
                    transforms=["toolschemas"],
                    est_saved_tokens=50,
                    input_tokens=500,
                    output_tokens=10,
                    cache_creation_tokens=0,
                    cache_read_tokens=0,
                    retrieved=0,
                )
            ],
        )
    for i in range(10):
        _seed(
            state_dir,
            [
                meter.Record(
                    request_key=f"h{i}",
                    holdout=True,
                    transforms=[],
                    est_saved_tokens=0,
                    input_tokens=1000,
                    output_tokens=10,
                    cache_creation_tokens=0,
                    cache_read_tokens=0,
                    retrieved=0,
                )
            ],
        )
    result = CliRunner().invoke(tokens_cmd, [], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "no measured data" not in result.output.lower()
    assert "50" in result.output  # 50% measured_saving_frac
    assert "10" in result.output  # n_treated / n_holdout


def test_json_reports_no_data_as_none_not_zero(tmp_path):
    result = CliRunner().invoke(tokens_cmd, ["--json"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["proxy"]["measured_saving_frac"] is None
    assert data["proxy"]["n_treated"] == 0


def test_by_transform_breaks_down_savings_and_flags_over_cutting(tmp_path):
    state_dir = tmp_path / "state"
    # toolschemas: applied to 10 requests, never retrieved back (healthy).
    for i in range(10):
        _seed(
            state_dir,
            [
                meter.Record(
                    request_key=f"a{i}",
                    holdout=False,
                    transforms=["toolschemas"],
                    est_saved_tokens=100,
                    input_tokens=500,
                    output_tokens=10,
                    cache_creation_tokens=0,
                    cache_read_tokens=0,
                    retrieved=0,
                )
            ],
        )
    # jsoncrush: applied to 10 requests, retrieved back on 6 of them — well
    # above the 0.05 default alarm threshold (over-cutting: a recovered
    # original costs its tokens twice).
    for i in range(10):
        _seed(
            state_dir,
            [
                meter.Record(
                    request_key=f"b{i}",
                    holdout=False,
                    transforms=["jsoncrush"],
                    est_saved_tokens=20,
                    input_tokens=500,
                    output_tokens=10,
                    cache_creation_tokens=0,
                    cache_read_tokens=0,
                    retrieved=1 if i < 6 else 0,
                )
            ],
        )
    result = CliRunner().invoke(tokens_cmd, ["--by-transform"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "toolschemas" in result.output
    assert "jsoncrush" in result.output
    assert "over-cutting" in result.output.lower()
    # The healthy transform's row must not itself be flagged.
    lines = result.output.lower().splitlines()
    toolschemas_lines = [line for line in lines if "toolschemas" in line]
    assert toolschemas_lines and all("over-cutting" not in line for line in toolschemas_lines)
