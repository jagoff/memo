import json

from memo.guard import log_guard_fire


def test_log_guard_fire_appends_jsonl(tmp_path):
    log_guard_fire(tmp_path, prompt="switch to X instead", ids=["a1b2c3d4"])
    log_guard_fire(tmp_path, prompt="revert that", ids=["e5f6g7h8"])
    lines = (tmp_path / "guard.log").read_text().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["ids"] == ["a1b2c3d4"]
    assert "prompt" in rec and "ts" not in rec  # no wall-clock in record (deterministic)


def test_log_guard_fire_never_raises_on_bad_dir(tmp_path):
    bad = tmp_path / "nope" / "deeper"
    # parent missing is fine — function mkdirs; unwritable would still not raise
    log_guard_fire(bad, prompt="p", ids=["x"])
    assert (bad / "guard.log").exists()
