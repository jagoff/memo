import json

from click.testing import CliRunner

from memo.cli import cli


def _env(tmp_path):
    return {"MEMO_NONINTERACTIVE": "1",
            "MEMO_DATA_DIR": str(tmp_path / "data"),
            "MEMO_STATE_DIR": str(tmp_path / "state")}


def test_dream_run_calls_distill_when_enabled(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "state").mkdir()
    seen = {}

    def fake_run(cfg, mem, **kw):
        seen["ran"] = True
        seen["kw"] = kw
        return {"status": "done", "distilled": [{"status": "saved"}]}

    monkeypatch.setattr("memo.dream_distill.run_distill", fake_run)
    env = _env(tmp_path)
    env["MEMO_DREAM_DISTILL_ENABLED"] = "1"
    env["MEMO_DREAM_DISTILL_THRESHOLD"] = "0.9"
    env["MEMO_DREAM_DISTILL_MIN_CONFIDENCE"] = "0.7"
    # --dry-run keeps other passes cheap; distill is stubbed so no MLX.
    res = CliRunner().invoke(cli, ["dream", "run", "--dry-run"], env=env)
    assert res.exit_code == 0, res.output
    assert seen.get("ran") is True
    assert seen["kw"]["threshold"] == 0.9
    assert seen["kw"]["min_confidence"] == 0.7


def test_dream_run_skips_distill_when_disabled(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "state").mkdir()
    seen = {}

    def fake_run(cfg, mem, **kw):
        seen["ran"] = True
        return {"status": "done", "distilled": []}

    monkeypatch.setattr("memo.dream_distill.run_distill", fake_run)
    env = _env(tmp_path)
    # MEMO_DREAM_DISTILL_ENABLED left unset — default OFF.
    res = CliRunner().invoke(cli, ["dream", "run", "--dry-run"], env=env)
    assert res.exit_code == 0, res.output
    assert seen.get("ran") is None


def test_dream_run_distill_error_is_swallowed(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "state").mkdir()

    def fake_run(cfg, mem, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr("memo.dream_distill.run_distill", fake_run)
    env = _env(tmp_path)
    env["MEMO_DREAM_DISTILL_ENABLED"] = "1"
    res = CliRunner().invoke(
        cli, ["dream", "run", "--dry-run", "--json", "--skip-orientation"], env=env
    )
    assert res.exit_code == 0, res.output
    receipt = json.loads(res.output[res.output.index("{") :])
    assert "distilled" not in receipt
    assert any("distill:" in e for e in receipt["errors"])


def test_dream_distill_subcommand_exists(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "state").mkdir()
    seen = {}

    def fake_run(cfg, mem, **kw):
        seen["kw"] = kw
        return {"status": "done", "distilled": [{"status": "saved", "title": "P"}]}

    monkeypatch.setattr("memo.dream_distill.run_distill", fake_run)
    env = _env(tmp_path)
    env["MEMO_DREAM_DISTILL_ENABLED"] = "1"
    env["MEMO_DREAM_DISTILL_THRESHOLD"] = "0.85"
    env["MEMO_DREAM_DISTILL_MIN_CONFIDENCE"] = "0.6"
    res = CliRunner().invoke(cli, ["dream", "distill", "--dry-run", "--json"], env=env)
    assert res.exit_code == 0, res.output
    assert "done" in res.output
    assert seen["kw"]["threshold"] == 0.85
    assert seen["kw"]["min_confidence"] == 0.6
