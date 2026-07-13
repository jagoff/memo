from click.testing import CliRunner

from memo.cli import cli


def _env(tmp_path):
    return {"MEMO_NONINTERACTIVE": "1",
            "MEMO_DATA_DIR": str(tmp_path / "data"),
            "MEMO_STATE_DIR": str(tmp_path / "state")}


def test_confidence_status_reports_empty(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "state").mkdir()
    res = CliRunner().invoke(cli, ["confidence", "status"], env=_env(tmp_path))
    assert res.exit_code == 0
    assert "calibration" in res.output.lower()


def test_confidence_status_shows_saved_map(tmp_path):
    (tmp_path / "data").mkdir()
    sd = tmp_path / "state"
    sd.mkdir()
    from memo.confidence_calibration import save_calibration
    save_calibration(sd, {"bins": {"high": {"predicted": 0.9, "observed": 0.4, "n": 12}},
                          "map": {"high": "med"}})
    res = CliRunner().invoke(cli, ["confidence", "status"], env=_env(tmp_path))
    assert res.exit_code == 0
    assert "high" in res.output and "med" in res.output
