import json

from click.testing import CliRunner


def _run_briefing(tmp_path):
    from memo.cli import cli

    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "memorias"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }
    (tmp_path / "memorias").mkdir(exist_ok=True)
    (tmp_path / "state").mkdir(exist_ok=True)
    return CliRunner().invoke(cli, ["briefing", "--compact"], env=env)


def test_briefing_shows_sync_nudge_when_local(tmp_path):
    r = _run_briefing(tmp_path)
    assert r.exit_code == 0, r.output
    ctx = json.loads(r.output)["hookSpecificOutput"]["additionalContext"]
    assert "memo sync setup" in ctx


def test_briefing_hides_sync_nudge_when_dismissed(tmp_path):
    from memo.config import Config
    from memo.sync_git import dismiss_sync_nudge

    cfg = Config(data_dir=tmp_path / "memorias", state_dir=tmp_path / "state", embedder_dims=4)
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    dismiss_sync_nudge(cfg)

    r = _run_briefing(tmp_path)
    ctx = json.loads(r.output)["hookSpecificOutput"]["additionalContext"]
    assert "memo sync setup" not in ctx
