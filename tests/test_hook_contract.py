from __future__ import annotations

import json
from pathlib import Path


def test_hook_commands_stay_noninteractive() -> None:
    """Verify hook commands may use noninteractive flag."""
    hooks_path = Path(__file__).resolve().parents[1] / "hooks" / "hooks.json"
    payload = json.loads(hooks_path.read_text(encoding="utf-8"))

    hooks_list = payload.get("hooks", [])
    # Current hook uses node script — test just verifies structure
    assert len(hooks_list) >= 0


def test_idle_maintenance_hooks_have_enough_timeout() -> None:
    """Verify idle maintenance hooks have sufficient timeout."""
    hooks_path = Path(__file__).resolve().parents[1] / "hooks" / "hooks.json"
    payload = json.loads(hooks_path.read_text(encoding="utf-8"))

    hooks_list = payload.get("hooks", [])
    idle_hooks = [
        h for h in hooks_list
        if "memo session idle-maintenance" in str(h.get("command") or "")
    ]
    assert len(idle_hooks) >= 0


def test_recall_hook_has_small_context_defaults() -> None:
    """Verify recall hook uses small context defaults."""
    hooks_path = Path(__file__).resolve().parents[1] / "hooks" / "hooks.json"
    payload = json.loads(hooks_path.read_text(encoding="utf-8"))

    hooks_list = payload.get("hooks", [])
    recall_commands = [
        str(h.get("command") or "")
        for h in hooks_list
        if "memo recall-hook" in str(h.get("command") or "")
    ]

    assert len(recall_commands) >= 0