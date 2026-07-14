"""memo dream run wiring for the profile pass (flags + receipt + errors)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from click.testing import CliRunner

from memo.cli_dream import dream_cmd
from memo.contradict import ScanResult

_SKIPS = [
    "--skip-orientation",
    "--skip-signal-gather",
    "--skip-entities",
    "--skip-decay",
    "--skip-prune-floor",
    "--skip-evict",
    "--skip-compress",
    "--skip-prewarm",
    "--skip-presynthesis",
]


def _mock_mem():
    mem = MagicMock()
    mem.lifecycle.enforce_forget_ttl.return_value = []
    mem.contradict_scanner.scan_corpus.return_value = ScanResult(
        scanned_memories=0,
        pairs_examined=0,
        pairs_inserted=0,
        pairs_refreshed=0,
        pairs_skipped_resolved=0,
        contradictions_found=0,
        evolutions_found=0,
    )
    mem.contradict_store.list_open.return_value = []
    mem.consolidator.consolidate_all.return_value = {"results": []}
    mem.temporal.detect_stale_memories.return_value = []
    mem.synthesize_cross_cluster.return_value = []
    return mem


def _run(tmp_cfg, monkeypatch, *, env_extra=None):
    monkeypatch.setattr("memo.cli_dream._get_memory", lambda _cfg: _mock_mem())
    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
        "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
        **(env_extra or {}),
    }
    result = CliRunner().invoke(dream_cmd, ["run", "--dry-run", "--json", *_SKIPS], env=env)
    assert result.exit_code == 0, result.output
    out = result.output
    return json.loads(out[out.index("{") :])


def test_profile_pass_runs_when_flag_on(tmp_cfg, monkeypatch):
    calls: list[dict] = []

    def _fake(cfg, mem, **kwargs):
        calls.append(kwargs)
        return {
            "status": "done",
            "written": [{"scope": "global", "status": "would_write"}],
            "standing_rules": 0,
        }

    monkeypatch.setattr("memo.dream_profile.run_profile_pass", _fake)
    receipt = _run(tmp_cfg, monkeypatch, env_extra={"MEMO_DREAM_PROFILE_ENABLED": "1"})
    assert receipt["profile"]["status"] == "done"
    assert calls and calls[0]["dry_run"] is True
    assert calls[0]["char_budget"] == 4000  # flag default flows through
    assert not receipt["errors"]


def test_profile_pass_off_by_default(tmp_cfg, monkeypatch):
    monkeypatch.setattr(
        "memo.dream_profile.run_profile_pass",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    receipt = _run(tmp_cfg, monkeypatch)
    assert "profile" not in receipt


def test_profile_pass_error_lands_in_receipt_errors(tmp_cfg, monkeypatch):
    monkeypatch.setattr(
        "memo.dream_profile.run_profile_pass",
        lambda *a, **k: {
            "status": "error",
            "written": [],
            "standing_rules": 0,
            "error": "RuntimeError: boom",
        },
    )
    receipt = _run(tmp_cfg, monkeypatch, env_extra={"MEMO_DREAM_PROFILE_ENABLED": "1"})
    assert any(e.startswith("profile:") and "boom" in e for e in receipt["errors"])


def test_profile_flags_registered_with_defaults(monkeypatch):
    from memo.flags import flag_bool, flag_float, flag_int

    monkeypatch.delenv("MEMO_DREAM_PROFILE_ENABLED", raising=False)
    assert flag_bool("MEMO_DREAM_PROFILE_ENABLED") is False  # default-off
    assert flag_int("MEMO_DREAM_PROFILE_CHAR_BUDGET") == 4000
    assert flag_int("MEMO_DREAM_PROFILE_MAX_PROJECTS") == 5
    assert flag_int("MEMO_DREAM_PROFILE_DIRECTIVE_K") == 3
    assert flag_float("MEMO_DREAM_PROFILE_DIRECTIVE_MIN_USED") == 0.5
