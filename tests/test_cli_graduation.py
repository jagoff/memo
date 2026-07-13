from click.testing import CliRunner

from memo.cli import cli


def _env(tmp_path):
    return {"MEMO_NONINTERACTIVE": "1",
            "MEMO_DATA_DIR": str(tmp_path / "data"),
            "MEMO_STATE_DIR": str(tmp_path / "state")}


def test_graduation_status_lists_seed_candidate(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "state").mkdir()
    res = CliRunner().invoke(cli, ["graduation", "status"], env=_env(tmp_path))
    assert res.exit_code == 0
    assert "MEMO_GRAPH_SIGNAL_ENABLED" in res.output


def test_graduation_revert_reports_when_nothing_to_revert(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "state").mkdir()
    res = CliRunner().invoke(
        cli, ["graduation", "revert", "MEMO_GRAPH_SIGNAL_ENABLED"], env=_env(tmp_path))
    assert res.exit_code == 0
    assert "MEMO_GRAPH_SIGNAL_ENABLED" in res.output


def test_graduation_explain_renders_receipt(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "state").mkdir()

    def fake_ctrl(cfg, mem, *, dry_run=False):
        return {"candidates": [{"flag": "MEMO_GRAPH_SIGNAL_ENABLED", "status": "accumulating",
                                "delta_prec": 0.01, "streak": 1, "k": 5}]}

    monkeypatch.setattr("memo.graduation.controller.run_graduation_controller", fake_ctrl)
    monkeypatch.setattr("memo.memory.Memory", lambda cfg: object())
    res = CliRunner().invoke(cli, ["graduation", "explain"], env=_env(tmp_path))
    assert res.exit_code == 0, res.output
    assert "MEMO_GRAPH_SIGNAL_ENABLED" in res.output


def test_graduation_status_lists_numeric_candidate(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "state").mkdir()
    res = CliRunner().invoke(cli, ["graduation", "status"], env=_env(tmp_path))
    assert res.exit_code == 0
    assert "MEMO_RECALL_MMR_LAMBDA" in res.output  # numeric candidate now listed


def test_graduation_explain_shows_best_value_for_numeric_candidate(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "state").mkdir()

    def fake_ctrl(cfg, mem, *, dry_run=False):
        return {"candidates": [{"flag": "MEMO_RECALL_MMR_LAMBDA", "status": "accumulating",
                                "delta_prec": 0.02, "streak": 2, "k": 5, "best_value": 0.3}]}

    monkeypatch.setattr("memo.graduation.controller.run_graduation_controller", fake_ctrl)
    monkeypatch.setattr("memo.memory.Memory", lambda cfg: object())
    res = CliRunner().invoke(cli, ["graduation", "explain"], env=_env(tmp_path))
    assert res.exit_code == 0, res.output
    assert "MEMO_RECALL_MMR_LAMBDA" in res.output
    assert "0.3" in res.output
