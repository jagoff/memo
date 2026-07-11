from click.testing import CliRunner


def test_doctor_off_hint_points_at_sync_setup(tmp_path):
    from memo.cli import cli

    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "memorias"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }
    (tmp_path / "memorias").mkdir()
    r = CliRunner().invoke(cli, ["doctor"], env=env)
    # data_dir is not a git clone here → the OFF branch fires
    assert "memo sync setup" in r.output
