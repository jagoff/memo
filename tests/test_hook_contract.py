from __future__ import annotations

import json
from pathlib import Path


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
