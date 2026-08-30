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
    # Arms are drawn per session, and the panel gates on the distinct-session
    # count, so a row with no session identity is a row the reporter cannot
    # count toward a healthy sample. Spread rows deterministically across four
    # sessions per arm by default; a test that cares about session shape (see
    # test_proxy_panel_withholds_a_ratio_drawn_from_one_holdout_session) passes
    # `session_key` explicitly.
    arm = "h" if holdout else "t"
    default_session = f"{arm}-sess-{sum(map(ord, key)) % 4}"
    return meter.Record(
        request_key=key,
        holdout=holdout,
        session_key=kw.pop("session_key", default_session),
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
    for i in range(30):
        _seed(state_dir, [_record(f"t{i}", holdout=False, input_tokens=500)])
    for i in range(30):
        _seed(state_dir, [_record(f"h{i}", holdout=True, input_tokens=1000)])
    result = CliRunner().invoke(tokens_cmd, [], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "no measured data" not in result.output.lower()
    assert "50.0% saved" in result.output
    assert "treated 500 vs holdout 1000" in result.output
    assert "n=30 treated / 30 holdout" in result.output
    # The panel label is "on prompt cost" unconditionally now (defect 1's
    # relabel), so a bare "cost" substring check would trip on the noun in
    # every panel, saved or not -- assert the VERB specifically stayed
    # "saved", not "cost".
    assert "% cost" not in result.output.lower()


def test_passthrough_rows_are_reported_but_excluded_from_the_treated_count(tmp_path):
    """Defect 3 fix, surfaced: a row recorded while the proxy was disabled
    (`rewritten=False`) must not inflate n=treated, but it also must not
    just silently vanish — the panel says how many were excluded and why."""
    state_dir = tmp_path / "state"
    for i in range(30):
        _seed(state_dir, [_record(f"t{i}", holdout=False, input_tokens=500)])
    for i in range(30):
        _seed(state_dir, [_record(f"h{i}", holdout=True, input_tokens=1000)])
    _seed(
        state_dir,
        [_record("p0", holdout=False, input_tokens=999, rewritten=False)],
    )
    result = CliRunner().invoke(tokens_cmd, [], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "n=30 treated / 30 holdout" in result.output
    assert "1 passthrough request" in result.output


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
    # See the sibling test above for why this checks the VERB, not the bare
    # "cost" substring (the panel label is "on prompt cost" unconditionally).
    assert "% cost" not in result.output.lower()
    assert "0.0% saved" in result.output


def test_json_reports_no_data_as_none_not_zero(tmp_path):
    result = CliRunner().invoke(tokens_cmd, ["--json"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["proxy"]["measured_saving_frac"] is None
    assert data["proxy"]["n_treated"] == 0


def test_proxy_panel_withholds_a_thin_sample(tmp_path):
    """Round-2 (item 6) originally: the sibling transcript panel applies a
    _MIN_COHORT_TURNS=30 floor and marks a thin cohort as provisional; the
    proxy panel applied none, so a n=1-vs-n=1 fluke could print e.g. "33.3%
    saved" in bold green with no qualifier at all. That round added the word
    "provisional" and kept the number.

    Round-3 (2026-08-28): annotating was not enough — live traffic rendered
    "386295.8% cost" WITH the provisional marker attached, and a reader takes
    the bold four-order-of-magnitude number, not the grey caveat. Below the
    floor the panel now withholds the ratio entirely."""
    state_dir = tmp_path / "state"
    _seed(state_dir, [_record("t0", holdout=False, input_tokens=500)])
    _seed(state_dir, [_record("h0", holdout=True, input_tokens=1000)])
    result = CliRunner().invoke(tokens_cmd, [], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "not enough data" in result.output.lower()
    assert "% saved" not in result.output
    assert "% cost" not in result.output


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


def test_a_thin_holdout_suppresses_the_percentage_instead_of_annotating_it(tmp_path):
    """Live defect (2026-08-28): `memo tokens` printed "386295.8% cost" —
    treated 19320 vs holdout 5 tok-equiv/request off n=4984 treated / 2
    holdout requests in ONE holdout session.

    Holdout is assigned per session at MEMO_PROXY_HOLDOUT_FRAC=0.05, so ~20
    sessions must elapse before one lands in the control arm, and the one
    that did was a 2-request session that never carried a real prompt. The
    `thin` flag already caught this and only appended a "provisional" word
    to a bold headline that was wrong by four orders of magnitude. A sample
    too thin to compare is not a measurement with a caveat; it is not a
    measurement. Suppress the number, keep the counts.
    """
    state_dir = tmp_path / "state"
    for i in range(40):
        _seed(state_dir, [_record(f"t{i}", holdout=False, input_tokens=19320, session_key="s-t")])
    for i in range(2):
        _seed(state_dir, [_record(f"h{i}", holdout=True, input_tokens=5, session_key="s-h")])
    result = CliRunner().invoke(tokens_cmd, [], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    out = result.output.lower()
    assert "not enough data" in out
    # The bogus headline and its verb must both be gone, not merely qualified.
    assert "%" not in result.output.split("memo · proxy")[-1].split("╰")[0]
    # The counts stay visible so the user can see why it is withheld.
    assert "40" in result.output and "2" in result.output


def test_suppressed_headline_still_reports_the_session_counts(tmp_path):
    """The effective sample unit is the session, not the request (holdout is
    assigned per session), so the withheld-measurement line must show it —
    otherwise "40 treated / 2 holdout requests" reads as a near-miss on the
    30-request floor when the real shortfall is 1 control session."""
    state_dir = tmp_path / "state"
    for i in range(40):
        _seed(state_dir, [_record(f"t{i}", holdout=False, input_tokens=19320, session_key="s-t")])
    for i in range(2):
        _seed(state_dir, [_record(f"h{i}", holdout=True, input_tokens=5, session_key="s-h")])
    result = CliRunner().invoke(tokens_cmd, [], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "session" in result.output.lower()


def test_by_transform_does_not_claim_to_decompose_the_billed_saving(tmp_path):
    """The per-transform split is raw removed text, not the cache-weighted cost.

    `measured_saving_frac` is a billed-cost ratio built with the 1.25x / 2.0x /
    0.1x cache weights; `saved_by` is an unweighted chars/4 diff that cannot
    carry a weight and cannot go negative. On the live ledger the two differ by
    9.11x, because cached prefix content is credited at full value every turn
    while the counterfactual bills it at 0.1x. The panel must not present one
    as a breakdown of the other.
    """
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
    assert "share of savings" not in result.output, "implies the billed saving"
    assert "text removed" in result.output, "must name the unit it actually reports"


def test_by_transform_help_names_the_unit_it_reports(tmp_path):
    """`--by-transform` help must not advertise the measured proxy saving."""
    result = CliRunner().invoke(tokens_cmd, ["--help"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "measured proxy saving" not in result.output


def test_proxy_panel_withholds_a_ratio_drawn_from_one_holdout_session(tmp_path):
    """The request-count floor cannot see the failure it was added for.

    Arms are assigned per SESSION, so every request inside one holdout session
    is the same draw repeated. A single large session clears `n_h >= 30`
    trivially — 400 requests here — and the panel would then print a bold
    percentage comparing one cluster against many. That is the exact shape of
    the "386295.8% cost" render the floor exists to prevent, at a scale the
    floor cannot catch.
    """
    state_dir = tmp_path / "state"
    for i in range(400):
        _seed(
            state_dir,
            [
                _record(
                    f"t{i}",
                    holdout=False,
                    input_tokens=500,
                    session_key=f"treated-{i % 8}",
                )
            ],
        )
    for i in range(400):
        _seed(
            state_dir,
            [
                _record(
                    f"h{i}",
                    holdout=True,
                    input_tokens=1000,
                    session_key="the-one-holdout-session",
                )
            ],
        )
    result = CliRunner().invoke(tokens_cmd, [], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "not enough data" in result.output.lower()
    assert "% saved" not in result.output
    assert "% cost" not in result.output
