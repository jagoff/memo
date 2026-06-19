from __future__ import annotations

import json
from pathlib import Path

from memo.flags import flag_int


def test_hook_commands_stay_noninteractive() -> None:
    hooks_path = Path(__file__).resolve().parents[1] / "hooks" / "hooks.json"
    payload = json.loads(hooks_path.read_text(encoding="utf-8"))

    for event_hooks in payload["hooks"].values():
        for matcher_group in event_hooks:
            for hook in matcher_group["hooks"]:
                if hook.get("type") != "command":
                    continue
                command = str(hook.get("command") or "")
                assert command.startswith("MEMO_NONINTERACTIVE=1 "), command


def test_idle_maintenance_hooks_have_enough_timeout() -> None:
    hooks_path = Path(__file__).resolve().parents[1] / "hooks" / "hooks.json"
    payload = json.loads(hooks_path.read_text(encoding="utf-8"))

    for event_hooks in payload["hooks"].values():
        for matcher_group in event_hooks:
            for hook in matcher_group["hooks"]:
                command = str(hook.get("command") or "")
                if "memo session idle-maintenance" not in command:
                    continue
                mode = "reflect" if "--mode reflect" in command else "capture"
                default_delay = (
                    flag_int("MEMO_SESSION_IDLE_REFLECT_SECS")
                    if mode == "reflect"
                    else flag_int("MEMO_SESSION_IDLE_CAPTURE_SECS")
                )
                assert default_delay is not None
                assert int(hook.get("timeout") or 0) >= default_delay + 5, command


def test_recall_hook_has_small_context_defaults() -> None:
    hooks_path = Path(__file__).resolve().parents[1] / "hooks" / "hooks.json"
    payload = json.loads(hooks_path.read_text(encoding="utf-8"))
    commands = [
        str(hook.get("command") or "")
        for group in payload["hooks"]["UserPromptSubmit"]
        for hook in group["hooks"]
        if "memo recall-hook" in str(hook.get("command") or "")
    ]

    assert len(commands) == 1
    command = commands[0]
    assert "MEMO_RECALL_TOKEN_BUDGET=160" in command
    assert "MEMO_RECALL_TOP_K=1" in command
    assert "MEMO_RECALL_FEEDBACK_HINT=0" in command
