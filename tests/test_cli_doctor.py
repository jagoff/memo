from unittest.mock import MagicMock, patch

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


def test_gc_report_closes_memory_after_scan():
    from memo.cli_doctor import _gc_report

    cfg = MagicMock()
    memory = MagicMock()
    expected = {"orphan_store": [], "orphan_disk": [], "stale_synthesis": []}
    memory.gc.return_value = expected

    with patch("memo.memory.Memory", return_value=memory):
        report = _gc_report(cfg, fix=False)

    assert report == expected
    memory.gc.assert_called_once_with(fix=False)
    memory.close.assert_called_once_with()
