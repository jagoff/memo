import dataclasses

import pytest

from memo.graduation.registry import NumericCandidate, numeric_candidates


def test_numeric_candidates_are_well_formed():
    cands = numeric_candidates()
    assert cands, "expected at least one numeric candidate"
    for c in cands:
        assert c.flag.startswith("MEMO_")
        assert c.field, "must name the RankKnobs field it pins"
        assert c.k >= 1
        assert c.epsilon >= 0.0
        # OFF baseline must be the flag's CURRENT default, never a blind 0.
        assert isinstance(c.off_value, float)
        assert isinstance(c.on_value, float)


def test_off_value_matches_live_flag_default():
    # The whole point of the extension: OFF is the real default, not "0".
    by_flag = {c.flag: c for c in numeric_candidates()}
    assert by_flag["MEMO_RECALL_MMR_LAMBDA"].off_value == 0.0
    assert by_flag["MEMO_RECALL_GLOBAL_BOOST"].off_value == 0.10
    assert by_flag["MEMO_RECALL_SYNTHESIS_BOOST"].off_value == 0.0


def test_numeric_candidate_is_frozen():
    c = numeric_candidates()[0]
    assert isinstance(c, NumericCandidate)
    assert dataclasses.is_dataclass(c)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.flag = "x"  # type: ignore[misc]


def test_project_boost_is_report_only():
    # project boost isn't offline-measurable without project-tagged labels.
    by_flag = {c.flag: c for c in numeric_candidates()}
    assert by_flag["MEMO_RECALL_PROJECT_BOOST"].auto_flip is False
