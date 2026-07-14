import json

from click.testing import CliRunner

from memo.cli import cli


def _env(tmp_path):
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def test_dream_status_shows_distilled_fragment(tmp_path):
    (tmp_path / "data").mkdir()
    sd = tmp_path / "state"
    (sd / "dream").mkdir(parents=True)
    (sd / "dream" / "last.json").write_text(
        json.dumps(
            {
                "ts": 1.0,
                "distilled": {"status": "done", "distilled": [{"status": "saved", "title": "P"}]},
            }
        ),
        encoding="utf-8",
    )
    res = CliRunner().invoke(cli, ["dream", "status"], env=_env(tmp_path))
    assert res.exit_code == 0, res.output
    assert "distill" in res.output.lower()


def test_dream_status_no_crash_when_distilled_missing(tmp_path):
    """Verify dream status is backward-compatible when distilled key is absent."""
    (tmp_path / "data").mkdir()
    sd = tmp_path / "state"
    (sd / "dream").mkdir(parents=True)
    (sd / "dream" / "last.json").write_text(
        json.dumps(
            {
                "ts": 1.0,
                "superseded": [],
                "merged": [],
            }
        ),
        encoding="utf-8",
    )
    res = CliRunner().invoke(cli, ["dream", "status"], env=_env(tmp_path))
    assert res.exit_code == 0, res.output
    # Should not crash; distill line simply absent
