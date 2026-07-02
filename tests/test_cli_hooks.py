"""Tests for `memo install-recall-hook` — memo-owned, self-healing recall wiring.

The systemic guarantee under test: memo wires its ``UserPromptSubmit`` recall
hook directly into ``~/.claude/settings.json`` (not only via the fragile plugin
``hooks.json``), *coexisting* with foreign UserPromptSubmit hooks, using the
absolute path to the ``memo`` binary. This survives a de-registered/clobbered
plugin (see the f5232b2 regression) and a minimal GUI PATH. Re-asserted on every
memo-mcp start via ``selfheal_recall_hook`` (``MEMO_HOOK_SELFHEAL``).
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from memo.cli import cli
from memo.cli_hooks import recall_hook_wired, selfheal_recall_hook, wire_recall_hook


def _env(tmp_path: Path) -> dict:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
    }


def _settings(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "claude" / "settings.json").read_text())


def _ups(tmp_path: Path) -> list:
    return _settings(tmp_path)["hooks"]["UserPromptSubmit"]


def _all_commands(ups: list) -> list[str]:
    return [h["command"] for g in ups for h in g.get("hooks", [])]


# ── install: fresh machine (no prior hooks) ───────────────────────────────────


def test_install_fresh_wires_recall_hook(tmp_path):
    result = CliRunner().invoke(cli, ["install-recall-hook"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    cmds = _all_commands(_ups(tmp_path))
    assert len(cmds) == 1
    assert cmds[0].endswith(" recall-hook")
    assert "MEMO_NONINTERACTIVE=1" in cmds[0]


def test_wired_command_uses_absolute_memo_path(tmp_path):
    # The whole point of the fix: a GUI launch with a minimal PATH must still
    # find memo. The wired command must reference an absolute path, not bare `memo`.
    wire_recall_hook(tmp_path / "claude", memo_bin="/opt/rt/bin/memo")
    cmd = _all_commands(_ups(tmp_path))[0]
    assert "/opt/rt/bin/memo recall-hook" in cmd


def test_wired_command_does_not_pin_min_sim(tmp_path):
    # MEMO_RECALL_MIN_SIM must NOT be pinned in the hook env — pinning it as an
    # env var would override the nightly recall tuner's overlay (env > overlay).
    wire_recall_hook(tmp_path / "claude", memo_bin="/opt/rt/bin/memo")
    cmd = _all_commands(_ups(tmp_path))[0]
    assert "MEMO_RECALL_MIN_SIM" not in cmd


# ── coexist: foreign UserPromptSubmit hooks are preserved ─────────────────────


def test_wire_preserves_foreign_ups_hooks(tmp_path):
    claude = tmp_path / "claude"
    claude.mkdir(parents=True)
    foreign = {
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": [{"type": "command", "command": "/x/caveman-tracker.js"}]}
            ]
        }
    }
    (claude / "settings.json").write_text(json.dumps(foreign))
    wire_recall_hook(claude, memo_bin="/opt/rt/bin/memo")
    cmds = _all_commands(_ups(tmp_path))
    assert "/x/caveman-tracker.js" in cmds
    assert any(c.endswith(" recall-hook") for c in cmds)


def test_wire_preserves_other_settings_keys(tmp_path):
    claude = tmp_path / "claude"
    claude.mkdir(parents=True)
    (claude / "settings.json").write_text(json.dumps({"model": "opus", "env": {"A": "1"}}))
    wire_recall_hook(claude, memo_bin="/opt/rt/bin/memo")
    s = _settings(tmp_path)
    assert s["model"] == "opus"
    assert s["env"] == {"A": "1"}


# ── idempotency: repeated wiring never duplicates the memo hook ────────────────


def test_wire_is_idempotent(tmp_path):
    claude = tmp_path / "claude"
    wire_recall_hook(claude, memo_bin="/opt/rt/bin/memo")
    result = wire_recall_hook(claude, memo_bin="/opt/rt/bin/memo")
    assert result["action"] == "already"
    cmds = _all_commands(_ups(tmp_path))
    assert sum(1 for c in cmds if c.endswith(" recall-hook")) == 1


def test_wire_updates_stale_memo_hook_in_place(tmp_path):
    claude = tmp_path / "claude"
    wire_recall_hook(claude, memo_bin="/old/bin/memo")
    result = wire_recall_hook(claude, memo_bin="/new/bin/memo")
    assert result["action"] == "updated"
    cmds = _all_commands(_ups(tmp_path))
    recall = [c for c in cmds if c.endswith(" recall-hook")]
    assert len(recall) == 1
    assert "/new/bin/memo" in recall[0]
    assert "/old/bin/memo" not in recall[0]


# ── self-heal gating ──────────────────────────────────────────────────────────


def test_selfheal_noop_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("MEMO_HOOK_SELFHEAL", "0")
    selfheal_recall_hook()
    assert not (tmp_path / "claude" / "settings.json").exists()


def test_selfheal_wires_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("MEMO_HOOK_SELFHEAL", "1")
    selfheal_recall_hook()
    cmds = _all_commands(_ups(tmp_path))
    assert any(c.endswith(" recall-hook") for c in cmds)


def test_selfheal_never_raises_on_bad_settings(tmp_path, monkeypatch):
    claude = tmp_path / "claude"
    claude.mkdir(parents=True)
    (claude / "settings.json").write_text("{not valid json")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude))
    monkeypatch.setenv("MEMO_HOOK_SELFHEAL", "1")
    # Must not raise — best-effort self-heal must never break memo-mcp start.
    selfheal_recall_hook()


# ── doctor detection: recall_hook_wired ───────────────────────────────────────


def test_recall_hook_wired_false_when_missing(tmp_path):
    assert recall_hook_wired(tmp_path / "claude") is False


def test_recall_hook_wired_true_after_wiring(tmp_path):
    claude = tmp_path / "claude"
    wire_recall_hook(claude, memo_bin="/opt/rt/bin/memo")
    assert recall_hook_wired(claude) is True


def test_recall_hook_wired_false_with_only_foreign_hooks(tmp_path):
    claude = tmp_path / "claude"
    claude.mkdir(parents=True)
    foreign = {
        "hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command", "command": "/x/y.js"}]}]}
    }
    (claude / "settings.json").write_text(json.dumps(foreign))
    assert recall_hook_wired(claude) is False
