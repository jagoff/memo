import json

from click.testing import CliRunner

from memo import dream_tune_online
from memo.cli_dream import dream_cmd


def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")


def test_timeline_empty(tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    res = CliRunner().invoke(dream_cmd, ["timeline"])
    assert res.exit_code == 0, res.output
    assert "no proof-loop history yet" in res.output


def test_timeline_renders_entries_and_summary(tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    state = tmp_path / "state"
    dream_tune_online.append_ledger(state, {
        "resolved_ts": "2026-07-01T03:00:00+00:00", "verdict": "confirmed",
        "floor_before": 0.5, "floor_after": 0.6, "online_before": 0.5,
        "online_after": 0.55, "realized_delta": 0.05, "n_after": 40})
    dream_tune_online.append_ledger(state, {
        "resolved_ts": "2026-07-02T03:00:00+00:00", "verdict": "reverted",
        "floor_before": 0.6, "floor_after": 0.65, "online_before": 0.55,
        "online_after": 0.40, "realized_delta": -0.15, "n_after": 50})
    res = CliRunner().invoke(dream_cmd, ["timeline"])
    assert res.exit_code == 0, res.output
    assert "proof-loop timeline" in res.output
    assert "kept" in res.output and "reverted" in res.output
    assert "kept ·" in res.output and "reverted ·" in res.output


def test_timeline_json(tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    state = tmp_path / "state"
    dream_tune_online.append_ledger(state, {"verdict": "confirmed", "realized_delta": 0.02})
    res = CliRunner().invoke(dream_cmd, ["timeline", "--json"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)[0]["verdict"] == "confirmed"


def test_timeline_shows_knob_generically(tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    state = tmp_path / "state"
    dream_tune_online.append_ledger(state, {
        "resolved_ts": "2026-07-03T03:00:00+00:00", "verdict": "confirmed",
        "knob": "MEMO_RECALL_GRAPH_PROXIMITY_WEIGHT",
        "floor_before": 0.0, "floor_after": 0.2, "online_before": 0.5,
        "online_after": 0.55, "realized_delta": 0.05, "n_after": 40})
    res = CliRunner().invoke(dream_cmd, ["timeline"])
    assert res.exit_code == 0, res.output
    assert "graph_proximity_weight" in res.output
    assert "min_sim" not in res.output


def test_timeline_minsim_entry_still_labeled_min_sim(tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path)
    state = tmp_path / "state"
    dream_tune_online.append_ledger(state, {
        "verdict": "confirmed", "floor_before": 0.5, "floor_after": 0.6,
        "realized_delta": 0.03, "n_after": 30})  # no knob → defaults to min_sim
    res = CliRunner().invoke(dream_cmd, ["timeline"])
    assert res.exit_code == 0, res.output
    assert "min_sim" in res.output
