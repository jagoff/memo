"""Test that the default candidate set spans boolean, numeric, and report-only candidates."""

from memo.graduation.registry import (
    Candidate,
    NumericCandidate,
    all_candidates,
    report_only_candidates,
)


def test_all_candidates_spans_bool_numeric_and_report_only() -> None:
    """Verify that all_candidates aggregates all three sources."""
    cands = all_candidates()
    flags = {c.flag for c in cands}

    # Phase-0 boolean seed still present:
    assert "MEMO_GRAPH_SIGNAL_ENABLED" in flags

    # Phase-1 numeric knobs present:
    assert "MEMO_RECALL_MMR_LAMBDA" in flags
    assert "MEMO_RECALL_SYNTHESIS_BOOST" in flags

    # report-only present:
    assert "MEMO_RECALL_RERANK_INPUT_K" in flags
    assert "MEMO_DREAM_GRADUATION_ENABLED" in flags
    assert "MEMO_RELATION_CANDIDATES_ENABLED" not in flags
    assert "MEMO_RELATION_ANNOTATIONS_ENABLED" not in flags

    # Both types are present:
    assert any(isinstance(c, NumericCandidate) for c in cands)
    assert any(isinstance(c, Candidate) and not isinstance(c, NumericCandidate) for c in cands)


def test_report_only_candidates_never_auto_flip() -> None:
    """Verify that report-only candidates have auto_flip=False."""
    for c in report_only_candidates():
        assert c.auto_flip is False
