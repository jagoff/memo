from memo.dashboard_logs import append_grounding_log, read_grounding_log


def test_grounding_row_stamped_base_without_overlay(tmp_path):
    append_grounding_log(
        tmp_path, session_id="s1", turn=1, recall_id="abc1234567", used_score=0.9, method="lexical"
    )
    rows = read_grounding_log(tmp_path)
    assert len(rows) == 1
    assert rows[0]["params_version"] == "base"


def test_grounding_row_stamped_with_active_overlay(tmp_path):
    from memo.tuned_overlay import params_version, write_overlay

    write_overlay(tmp_path, {"MEMO_RECALL_MIN_SIM": 0.7}, {"set_by": "test"})
    append_grounding_log(
        tmp_path, session_id="s1", turn=2, recall_id="deadbeef99", used_score=0.9, method="embed"
    )
    rows = read_grounding_log(tmp_path)
    assert len(rows) == 1
    expected = params_version(tmp_path)
    assert expected != "base"
    assert rows[0]["params_version"] == expected
