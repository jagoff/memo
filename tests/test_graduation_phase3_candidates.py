from memo.graduation.registry import all_candidates, report_only_candidates


def test_interject_and_ask_are_report_only_candidates():
    ro = report_only_candidates()
    by_flag = {c.flag: c for c in ro}
    assert "MEMO_INTERJECT_ENABLED" in by_flag
    assert "MEMO_ASK_GAPS_ENABLED" in by_flag
    assert by_flag["MEMO_INTERJECT_ENABLED"].auto_flip is False
    assert by_flag["MEMO_ASK_GAPS_ENABLED"].auto_flip is False


def test_phase3_candidates_appear_in_all_candidates():
    flags = {c.flag for c in all_candidates()}
    assert {"MEMO_INTERJECT_ENABLED", "MEMO_ASK_GAPS_ENABLED"} <= flags


def test_existing_report_only_candidates_still_present():
    # back-compat: Phase-0/1 report-only entries survive the extension
    flags = {c.flag for c in report_only_candidates()}
    assert "MEMO_RECALL_RERANK_INPUT_K" in flags
    assert "MEMO_DREAM_GRADUATION_ENABLED" in flags
