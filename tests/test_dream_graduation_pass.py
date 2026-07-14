import memo.cli_dream as cli_dream


def test_count_graduated_counts_graduated_candidates():
    receipt = {
        "graduation": {
            "candidates": [
                {"flag": "A", "status": "graduated"},
                {"flag": "B", "status": "accumulating"},
                {"flag": "C", "status": "graduated"},
            ]
        }
    }
    assert cli_dream._count_graduated(receipt) == 2


def test_count_graduated_handles_missing_graduation_key():
    assert cli_dream._count_graduated({}) == 0


def test_count_graduated_handles_non_dict_graduation():
    assert cli_dream._count_graduated({"graduation": "oops"}) == 0


def test_count_graduated_handles_non_list_candidates():
    assert cli_dream._count_graduated({"graduation": {"candidates": "oops"}}) == 0


def test_count_graduated_handles_non_dict_candidate_entries():
    assert cli_dream._count_graduated({"graduation": {"candidates": ["oops", 1, None]}}) == 0


def test_dream_runs_graduation_when_enabled(monkeypatch, tmp_cfg):
    seen = {}

    def fake_controller(cfg, mem, *, dry_run=False):
        seen["ran"] = True
        return {"candidates": [{"flag": "MEMO_GRAPH_SIGNAL_ENABLED", "status": "accumulating"}]}

    monkeypatch.setattr("memo.graduation.controller.run_graduation_controller", fake_controller)
    monkeypatch.setenv("MEMO_GRADUATION_CONTROLLER_ENABLED", "1")

    receipt = cli_dream._run_graduation_pass(
        tmp_cfg, object(), dry_run=False, receipt={"errors": []}
    )
    assert seen.get("ran") is True
    assert receipt["graduation"]["candidates"][0]["flag"] == "MEMO_GRAPH_SIGNAL_ENABLED"


def test_dream_graduation_pass_swallows_errors(monkeypatch, tmp_cfg):
    def boom(cfg, mem, *, dry_run=False):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("memo.graduation.controller.run_graduation_controller", boom)
    monkeypatch.setenv("MEMO_GRADUATION_CONTROLLER_ENABLED", "1")
    receipt = cli_dream._run_graduation_pass(
        tmp_cfg, object(), dry_run=False, receipt={"errors": []}
    )
    assert any("graduation" in e for e in receipt["errors"])
    assert "graduation" not in receipt  # no partial receipt on failure
