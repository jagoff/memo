"""Human-visible activity counters for the statusline badge.

``state_dir/presence_today.json`` — ``{"date": "YYYY-MM-DD", "recalls": N,
"saves": N, "tokens_saved": N}``. Written atomically (tmp + os.replace),
rolled over when the local date changes, regenerated from zero when corrupt.
Read by ``statusline/memo-statusline.sh`` (bash, jq/grep) — keys stay flat
and values numeric. Writers never raise: presence is decoration; it must
never break a save, the recall hook, or capture-stop.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path


def presence_path(state_dir: Path) -> Path:
    return state_dir / "presence_today.json"


def _empty(today: str) -> dict:
    return {"date": today, "recalls": 0, "saves": 0, "tokens_saved": 0}


def read_today(state_dir: Path) -> dict:
    """Today's counters; zeros on missing / stale-date / corrupt file."""
    today = date.today().isoformat()
    try:
        data = json.loads(presence_path(state_dir).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("date") != today:
            return _empty(today)
        return {
            "date": today,
            "recalls": int(data.get("recalls", 0) or 0),
            "saves": int(data.get("saves", 0) or 0),
            "tokens_saved": int(data.get("tokens_saved", 0) or 0),
        }
    except Exception:
        return _empty(today)


def _write(state_dir: Path, data: dict) -> None:
    path = presence_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def bump(state_dir: Path, *, recalls: int = 0, saves: int = 0) -> None:
    """Increment today's counters. Swallows every error."""
    try:
        data = read_today(state_dir)
        data["recalls"] += int(recalls)
        data["saves"] += int(saves)
        _write(state_dir, data)
    except Exception:  # noqa: S110  # decoration — never break the caller
        pass


def set_tokens(state_dir: Path, tokens_saved: int) -> None:
    """Overwrite today's tokens-saved figure (from the token ledger rollup)."""
    try:
        data = read_today(state_dir)
        data["tokens_saved"] = max(0, int(tokens_saved))
        _write(state_dir, data)
    except Exception:  # noqa: S110  # decoration — never break the caller
        pass


def summary_line(data: dict) -> str:
    """One-line plain-text activity summary for cross-agent surfaces.

    Agents without a statusline (Codex, Devin, opencode, Cursor) reach memo over
    MCP; this renders the same counters the statusline shows into a line memo can
    prepend to its MCP ``notification`` field, so they still see memo working.
    Returns ``""`` when nothing happened today — presence must never claim
    activity that did not occur.
    """
    recalls = int(data.get("recalls", 0) or 0)
    saves = int(data.get("saves", 0) or 0)
    tokens = int(data.get("tokens_saved", 0) or 0)
    parts: list[str] = []
    if recalls:
        parts.append(f"🧠 {recalls} recalled")
    if saves:
        parts.append(f"💾 {saves} saved")
    if tokens >= 1000:
        parts.append(f"~{tokens // 1000}k tok")
    elif tokens:
        parts.append(f"~{tokens} tok")
    if not parts:
        return ""
    return "※ memo today · " + " · ".join(parts)
