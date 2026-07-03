import json
import re

from click.testing import CliRunner

from memo import dream_tune_online
from memo.cli_dream import dream_cmd


def test_status_renders_proof_loop_ledger(tmp_path, monkeypatch):
    state = tmp_path / "state"
    monkeypatch.setenv("MEMO_STATE_DIR", str(state))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    (state / "dream").mkdir(parents=True)
    (state / "dream" / "last.json").write_text(
        json.dumps({"ts": 1751000000, "tuner": {"status": "applied"}})
    )
    dream_tune_online.append_ledger(
        state,
        {
            "verdict": "confirmed",
            "floor_before": 0.5,
            "floor_after": 0.6,
            "online_before": 0.5,
            "online_after": 0.55,
            "realized_delta": 0.05,
            "n_after": 40,
        },
    )

    res = CliRunner().invoke(dream_cmd, ["status"])
    assert res.exit_code == 0, res.output
    assert "proof loop" in res.output
    assert "confirmed" in res.output


def test_status_renders_capture_weights_fragment(tmp_path, monkeypatch):
    state = tmp_path / "state"
    monkeypatch.setenv("MEMO_STATE_DIR", str(state))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    (state / "dream").mkdir(parents=True)
    (state / "dream" / "last.json").write_text(
        json.dumps({"ts": 1751000000, "capture_weights": {"types": 2, "top": "decision:1.4"}})
    )

    res = CliRunner().invoke(dream_cmd, ["status"])
    assert res.exit_code == 0, res.output
    plain = re.sub(r"\x1b\[[0-9;]*m", "", res.output)  # rich number-highlighting
    assert "capture weights: 2 type(s)" in plain
    assert "top decision:1.4" in plain
