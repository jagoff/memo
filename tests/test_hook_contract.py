"""Contract tests for memo's hook wiring.

Two delivery surfaces after the f5232b2 regression fix:
- The plugin ``hooks/hooks.json`` (Claude Code nested format) delivers the
  ambient SessionStart/Stop/UserPromptSubmit suite (briefing, prewarm, daemon,
  capture, sync) — but NOT recall.
- The recall hook is memo-owned and wired into ``settings.json`` by
  ``memo.cli_hooks`` (self-healed on memo-mcp start), so it survives a
  de-registered/clobbered plugin.
"""

from __future__ import annotations

import json
from pathlib import Path


def _plugin_hooks() -> dict:
    path = Path(__file__).resolve().parents[1] / "hooks" / "hooks.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _all_commands(payload: dict) -> list[str]:
    """Flatten every command in the Claude Code nested hooks format."""
    commands: list[str] = []
    hooks = payload.get("hooks", {})
    assert isinstance(hooks, dict), "plugin hooks.json must use the CC nested format"
    for groups in hooks.values():
        for group in groups:
            for hook in group.get("hooks", []):
                cmd = hook.get("command")
                if isinstance(cmd, str):
                    commands.append(cmd)
    return commands


def test_hook_commands_stay_noninteractive() -> None:
    """Every plugin hook must set MEMO_NONINTERACTIVE=1 so the first-run picker
    can never fire inside a non-interactive hook."""
    commands = _all_commands(_plugin_hooks())
    assert commands, "plugin hooks.json wires no commands"
    missing = [c for c in commands if "MEMO_NONINTERACTIVE=1" not in c]
    assert not missing, f"hook commands missing MEMO_NONINTERACTIVE=1: {missing}"


def test_recall_hook_not_in_plugin_to_avoid_double_fire() -> None:
    """Recall is owned by settings.json (cli_hooks self-heal). It must NOT also
    be wired in the plugin, or every prompt would recall twice."""
    commands = _all_commands(_plugin_hooks())
    assert not [c for c in commands if "recall-hook" in c]


def test_idle_maintenance_hooks_have_enough_timeout() -> None:
    """Idle-maintenance mines the session chunk; it needs a generous timeout."""
    hooks = _plugin_hooks()["hooks"]
    idle = [
        hook
        for groups in hooks.values()
        for group in groups
        for hook in group.get("hooks", [])
        if "memo session idle-maintenance" in str(hook.get("command") or "")
    ]
    assert idle, "idle-maintenance hook missing from plugin hooks.json"
    for hook in idle:
        assert int(hook.get("timeout", 0)) >= 20


def test_recall_hook_has_small_context_defaults() -> None:
    """The memo-owned recall command (settings.json) must keep the small-context
    budget defaults, and must NOT pin MEMO_RECALL_MIN_SIM (that stays governed by
    the nightly recall tuner's overlay — env would override it)."""
    from memo.cli_hooks import _memo_command

    cmd = _memo_command(memo_bin="memo")
    assert "MEMO_RECALL_TOP_K=1" in cmd
    assert "MEMO_RECALL_TOKEN_BUDGET=160" in cmd
    assert "MEMO_RECALL_MIN_SIM" not in cmd


def _load_plugin_hooks() -> dict:
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "hooks" / "hooks.json"
    return json.loads(path.read_text(encoding="utf-8"))["hooks"]


def test_precompact_force_flushes_capture() -> None:
    pre = _load_plugin_hooks().get("PreCompact")
    assert pre, "PreCompact hook missing from plugin hooks.json"
    cmds = [h["command"] for g in pre for h in g["hooks"]]
    assert any("capture-tick --force" in c for c in cmds)


def test_sessionstart_briefing_rebriefs_after_compact() -> None:
    groups = _load_plugin_hooks()["SessionStart"]
    briefing_matchers = [
        g.get("matcher", "")
        for g in groups
        if any("briefing" in h["command"] for h in g["hooks"])
    ]
    assert briefing_matchers and all("compact" in m for m in briefing_matchers)
