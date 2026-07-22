from click.testing import CliRunner

from memo.cli import cli


def _env(tmp_path):
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


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
        cli, ["graduation", "revert", "MEMO_GRAPH_SIGNAL_ENABLED"], env=_env(tmp_path)
    )
    assert res.exit_code == 0
    assert "MEMO_GRAPH_SIGNAL_ENABLED" in res.output


def test_graduation_status_lists_numeric_candidate(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "state").mkdir()
    res = CliRunner().invoke(cli, ["graduation", "status"], env=_env(tmp_path))
    assert res.exit_code == 0
    assert "MEMO_RECALL_MMR_LAMBDA" in res.output  # numeric candidate now listed
