from click.testing import CliRunner

from memo.cli import cli


def test_digest_off_by_default(tmp_path):
    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "d"),
        "MEMO_STATE_DIR": str(tmp_path / "s"),
    }
    r = CliRunner().invoke(cli, ["digest"], env=env)
    assert r.exit_code == 0
    assert "disabled" in r.output.lower() or "MEMO_PROACTIVE_ENABLED" in r.output


def test_digest_empty_when_enabled_no_candidates(tmp_path):
    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "d"),
        "MEMO_STATE_DIR": str(tmp_path / "s"),
        "MEMO_PROACTIVE_ENABLED": "1",
    }
    r = CliRunner().invoke(cli, ["digest"], env=env)
    assert r.exit_code == 0
    assert "nothing to surface" in r.output
