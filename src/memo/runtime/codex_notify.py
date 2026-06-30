from __future__ import annotations

import base64
import importlib.metadata
import os
from pathlib import Path

from memo.flags import flag_str


def agent_tty_path() -> Path | None:
    raw = flag_str("MEMO_AGENT_TTY").strip()
    if raw:
        return Path(raw)

    data_home = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local/share"))
    try:
        raw = (data_home / "memo" / "agent_tty").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return Path(raw) if raw else None


def emit_codex_notify(title: str, body: str = "") -> bool:
    """Emit a Supacode/Codex top-line notification via the active agent TTY."""
    tty = agent_tty_path()
    if tty is None:
        return False

    encoded_title = base64.b64encode(title.encode("utf-8")).decode("ascii")
    encoded_body = base64.b64encode(body.encode("utf-8")).decode("ascii")
    payload = f"\033]3008;start=codex;kind=notify;title={encoded_title};body={encoded_body}\033\\"
    try:
        with open(tty, "ab", buffering=0) as fh:
            fh.write(payload.encode("ascii"))
        return True
    except OSError:
        return False


def memo_version_badge() -> str:
    try:
        version = importlib.metadata.version("mlx-memo")
    except Exception:
        version = "?"
    return f"[Memo {version}]"


def emit_memo_badge() -> bool:
    return emit_codex_notify(memo_version_badge())
