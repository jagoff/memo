import json
from pathlib import Path

from click.testing import CliRunner

from memo import eval_tokens
from memo.cli import cli


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def test_tokens_update_baseline_then_gate_pass(tmp_path, monkeypatch):
    canned = [eval_tokens.LeverRow("recall_format_compact", "recall_output", 100, 80, 1.0, 1.0)]
    monkeypatch.setattr(eval_tokens, "run_all", lambda **kw: canned)
    # Skip the live-index/labels wiring the stub doesn't need.
    monkeypatch.setattr("memo.cli_eval._get_memory", lambda cfg: object())

    runner = CliRunner()
    r1 = runner.invoke(cli, ["eval", "tokens", "--update-baseline"], env=_env(tmp_path))
    assert r1.exit_code == 0, r1.output
    baseline = tmp_path / "state" / "eval" / "token_baseline.json"
    assert baseline.exists()
    saved = json.loads(baseline.read_text())
    assert saved["recall_format_compact"]["passed"] is True

    r2 = runner.invoke(cli, ["eval", "tokens", "--gate"], env=_env(tmp_path))
    assert r2.exit_code == 0, r2.output


def test_tokens_gate_fails_on_regression(tmp_path, monkeypatch):
    monkeypatch.setattr("memo.cli_eval._get_memory", lambda cfg: object())
    good = [eval_tokens.LeverRow("recall_format_compact", "recall_output", 100, 80, 1.0, 1.0)]
    monkeypatch.setattr(eval_tokens, "run_all", lambda **kw: good)
    runner = CliRunner()
    runner.invoke(cli, ["eval", "tokens", "--update-baseline"], env=_env(tmp_path))

    bad = [eval_tokens.LeverRow("recall_format_compact", "recall_output", 100, 99, 1.0, 1.0)]
    monkeypatch.setattr(eval_tokens, "run_all", lambda **kw: bad)
    r = runner.invoke(cli, ["eval", "tokens", "--gate"], env=_env(tmp_path))
    assert r.exit_code == 1
    assert "FAIL" in r.output


def test_tokens_gate_without_baseline_errors(tmp_path, monkeypatch):
    monkeypatch.setattr("memo.cli_eval._get_memory", lambda cfg: object())
    monkeypatch.setattr(eval_tokens, "run_all", lambda **kw: [])
    r = CliRunner().invoke(cli, ["eval", "tokens", "--gate"], env=_env(tmp_path))
    assert r.exit_code != 0
    assert "baseline" in r.output.lower()
