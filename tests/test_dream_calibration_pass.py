import memo.cli_dream as cli_dream


def test_dream_builds_calibration_when_gate_enabled(monkeypatch, tmp_cfg):
    seen = {}

    def fake_build(state_dir, mem, **kw):
        seen["ran"] = True
        return {"bins": {}, "map": {"high": "high"}}

    monkeypatch.setattr("memo.confidence_calibration.build_calibration", fake_build)
    monkeypatch.setenv("MEMO_RECALL_CONFIDENCE_GATE", "1")
    receipt = cli_dream._run_calibration_pass(
        tmp_cfg, object(), dry_run=False, receipt={"errors": []}
    )
    assert seen.get("ran") is True
    assert receipt["calibration"]["map"]["high"] == "high"


def test_dream_calibration_noop_when_gate_off(monkeypatch, tmp_cfg):
    monkeypatch.delenv("MEMO_RECALL_CONFIDENCE_GATE", raising=False)
    receipt = cli_dream._run_calibration_pass(
        tmp_cfg, object(), dry_run=False, receipt={"errors": []}
    )
    assert "calibration" not in receipt


def test_dream_calibration_swallows_errors(monkeypatch, tmp_cfg):
    def boom(state_dir, mem, **kw):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("memo.confidence_calibration.build_calibration", boom)
    monkeypatch.setenv("MEMO_RECALL_CONFIDENCE_GATE", "1")
    receipt = cli_dream._run_calibration_pass(
        tmp_cfg, object(), dry_run=False, receipt={"errors": []}
    )
    assert any("calibration" in e for e in receipt["errors"])
    assert "calibration" not in receipt
