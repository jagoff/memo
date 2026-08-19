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


def _record(key: str, *, holdout: bool, input_tokens: int, **kw) -> meter.Record:
    return meter.Record(
        request_key=key,
        holdout=holdout,
        transforms=kw.pop("transforms", []),
        est_saved_tokens=kw.pop("est_saved_tokens", 0),
        input_tokens=input_tokens,
        output_tokens=10,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        retrieved=0,
        **kw,
    )


def test_no_data_says_so_instead_of_printing_a_zero(tmp_path):
    result = CliRunner().invoke(tokens_cmd, [], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "no measured data" in result.output.lower()


def test_the_estimated_panel_is_gone(tmp_path):
    """Round-2 hardening (item 8): the empty-state early-return never
    constructs a panel at all, so this assertion used to pass for free —
    retitling the panel to "memo MUTANT tokens saved (estimated)" left the
    file green. Seed real treated/holdout data first so a panel is actually
    rendered before checking "estimated" is absent from it."""
    state_dir = tmp_path / "state"
    for i in range(30):
        _seed(state_dir, [_record(f"t{i}", holdout=False, input_tokens=500)])
    for i in range(30):
        _seed(state_dir, [_record(f"h{i}", holdout=True, input_tokens=1000)])
    result = CliRunner().invoke(tokens_cmd, ["--by-transform"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "no measured data" not in result.output.lower()
    assert "estimated" not in result.output.lower()


def test_the_roi_constants_are_no_longer_registered():
    from memo.flags import REGISTRY

    assert "MEMO_ROI_TOKENS_PER_GROUNDED" not in REGISTRY
    assert "MEMO_ROI_TOKENS_PER_CONSULT" not in REGISTRY
    assert "MEMO_ROI_TOKENS_PER_REASK" not in REGISTRY


def test_measured_panel_reports_treated_vs_holdout_when_data_exists(tmp_path):
    """Round-2 hardening (item 7): the original assertions ("50" in output,
    "10" in output) survive a sign-inversion mutation (`verb = "cost" if
    frac > 0 else "saved"`) since `abs(frac)` prints the magnitude sign-free —
    only `verb`/`colour` carry the sign, and nothing asserted them. Assert
    the actual claim."""
    state_dir = tmp_path / "state"
    for i in range(10):
        _seed(state_dir, [_record(f"t{i}", holdout=False, input_tokens=500)])
    for i in range(10):
        _seed(state_dir, [_record(f"h{i}", holdout=True, input_tokens=1000)])
    result = CliRunner().invoke(tokens_cmd, [], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "no measured data" not in result.output.lower()
    assert "50.0% saved" in result.output
    assert "treated 500 vs holdout 1000" in result.output
    assert "n=10 treated / 10 holdout" in result.output
    assert "cost" not in result.output.lower()


def test_equal_arms_reads_as_saved_not_cost(tmp_path):
    """Sibling to the above (item 7): a real 0% delta is a neutral result,
    not a loss — today's `verb = "saved" if frac > 0 else "cost"` renders an
    exact-zero delta as "0.0% cost", which overstates a null result as a
    regression. Zero should read as "saved" (nothing lost, nothing gained)."""
    state_dir = tmp_path / "state"
    for i in range(30):
        _seed(state_dir, [_record(f"t{i}", holdout=False, input_tokens=500)])
    for i in range(30):
        _seed(state_dir, [_record(f"h{i}", holdout=True, input_tokens=500)])
    result = CliRunner().invoke(tokens_cmd, [], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "cost" not in result.output.lower()
    assert "0.0% saved" in result.output


def test_json_reports_no_data_as_none_not_zero(tmp_path):
    result = CliRunner().invoke(tokens_cmd, ["--json"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["proxy"]["measured_saving_frac"] is None
    assert data["proxy"]["n_treated"] == 0


def test_proxy_panel_flags_a_thin_sample(tmp_path):
    """Round-2 (item 6): the sibling transcript panel applies a
    _MIN_COHORT_TURNS=30 floor and marks a thin cohort as provisional; the
    proxy panel applied none, so a n=1-vs-n=1 fluke could print e.g. "33.3%
    saved" in bold green with no qualifier at all."""
    state_dir = tmp_path / "state"
    _seed(state_dir, [_record("t0", holdout=False, input_tokens=500)])
    _seed(state_dir, [_record("h0", holdout=True, input_tokens=1000)])
    result = CliRunner().invoke(tokens_cmd, [], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "provisional" in result.output.lower()


def test_proxy_panel_no_thin_marker_at_a_healthy_sample(tmp_path):
    state_dir = tmp_path / "state"
    for i in range(30):
        _seed(state_dir, [_record(f"t{i}", holdout=False, input_tokens=500)])
    for i in range(30):
        _seed(state_dir, [_record(f"h{i}", holdout=True, input_tokens=1000)])
    result = CliRunner().invoke(tokens_cmd, [], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "provisional" not in result.output.lower()


def test_partial_data_says_not_enough_instead_of_a_fabricated_percentage(tmp_path):
    """Round-2 hardening (item 9): treated-only data (no holdout rows yet —
    the normal early state under a small MEMO_PROXY_HOLDOUT_FRAC) hits the
    "no measured data" branch's sibling — `measured_saving_frac` stays None
    since there's nothing to compare against. Replacing that whole branch's
    body with a literal "0.0% saved" left `pytest -k "proxy or token"` green
    because nothing asserted this specific path."""
    state_dir = tmp_path / "state"
    for i in range(20):
        _seed(state_dir, [_record(f"t{i}", holdout=False, input_tokens=500)])
    result = CliRunner().invoke(tokens_cmd, ["--by-transform"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "not enough data" in result.output.lower()
    assert "0.0%" not in result.output


def test_by_transform_shows_an_honest_per_transform_share(tmp_path):
    """Round-2 (item 5a): meter.py used to credit the row's whole scalar
    est_saved_tokens to every transform listed in `transforms`, so N
    transforms that merely ran (whether or not they saved anything) always
    reported a flat 1/N share. `--by-transform` must show the real split."""
    state_dir = tmp_path / "state"
    for i in range(10):
        _seed(
            state_dir,
            [
                _record(
                    f"a{i}",
                    holdout=False,
                    input_tokens=500,
                    transforms=["toolschemas", "jsoncrush"],
                    est_saved_tokens=100,
                    saved_by={"jsoncrush": 100},
                )
            ],
        )
    result = CliRunner().invoke(tokens_cmd, ["--by-transform"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "toolschemas" in result.output
    assert "jsoncrush" in result.output
    lines = result.output.splitlines()
    toolschemas_line = next(line for line in lines if "toolschemas" in line)
    jsoncrush_line = next(line for line in lines if "jsoncrush" in line)
    assert "100%" in jsoncrush_line
    assert "0%" in toolschemas_line
    assert "100%" not in toolschemas_line


def test_by_transform_has_no_dead_retrieval_rate_column(tmp_path):
    """Round-2 (item 5b): `retrieved` has no writer anywhere in production —
    the rate was a constant 0.0 and the over-cutting alarm could never fire.
    A column that can never show anything but "healthy" is the same banned
    fabricated shape this task exists to remove; dropped, not faked."""
    state_dir = tmp_path / "state"
    for i in range(10):
        _seed(
            state_dir,
            [
                _record(
                    f"a{i}",
                    holdout=False,
                    input_tokens=500,
                    transforms=["toolschemas"],
                    est_saved_tokens=100,
                    saved_by={"toolschemas": 100},
                )
            ],
        )
    result = CliRunner().invoke(tokens_cmd, ["--by-transform"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "retrieval rate" not in result.output.lower()
    assert "over-cutting" not in result.output.lower()
