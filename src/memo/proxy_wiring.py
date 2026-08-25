"""Point Claude Code at the local proxy — and, just as importantly, un-point it.

`ANTHROPIC_BASE_URL` is a HARD dependency, not a hint. With it set to a loopback
port where nothing listens, Claude Code fails exactly like a dead network: from
the CLI's side there is no proxy between here and there, just connection
refused (the same failure `cli_doctor._check_proxy` exists to name). Wiring it
is therefore only ever done *after* the proxy has answered, and `unwire` exists
so a broken install is one command away from undone rather than a mystery
outage.

Two rules are load-bearing, and both are about not touching what isn't ours:

- A non-loopback base URL is somebody's deliberate routing decision — a
  corporate LLM gateway, a staging endpoint. `wire` refuses to overwrite it and
  `unwire` refuses to delete it. memo only ever owns a 127.0.0.1 URL.
- Every other key in settings.json belongs to the user. The file is read,
  minimally mutated, and written through a tmp + os.replace, the same idiom
  `cli_hooks` uses on this file.

Nothing here raises for a caller: a settings.json that is missing, unreadable
or malformed reports "did not change anything" rather than taking down the
install that called it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_ENV_KEY = "ANTHROPIC_BASE_URL"
_LOOPBACK = ("127.0.0.1", "localhost", "::1")


def settings_path(claude_dir: Path) -> Path:
    return Path(claude_dir) / "settings.json"


def _load(claude_dir: Path) -> dict[str, Any] | None:
    """Parsed settings, or None when there is nothing usable to edit."""
    path = settings_path(claude_dir)
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _save(claude_dir: Path, data: dict[str, Any]) -> bool:
    path = settings_path(claude_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        return False
    return True


def _is_ours(url: str) -> bool:
    """True only for a loopback URL — the only kind memo is allowed to own."""
    try:
        return (urlparse(url.strip()).hostname or "").lower() in _LOOPBACK
    except ValueError:
        return False


def wired_port(claude_dir: Path) -> int | None:
    """The loopback port settings.json currently points at, if any."""
    data = _load(claude_dir)
    if data is None:
        return None
    url = str((data.get("env") or {}).get(_ENV_KEY) or "")
    if not url or not _is_ours(url):
        return None
    try:
        return urlparse(url.strip()).port
    except ValueError:
        return None


def wire(claude_dir: Path, port: int) -> bool:
    """Point Claude Code at 127.0.0.1:port. True when the file changed.

    Returns False -- without touching anything -- when settings.json is
    unreadable, when it already points at this exact port, or when it points
    somewhere that is not loopback.
    """
    data = _load(claude_dir)
    if data is None:
        return False
    env = data.get("env")
    env = env if isinstance(env, dict) else {}
    existing = str(env.get(_ENV_KEY) or "").strip()
    if existing and not _is_ours(existing):
        return False
    url = f"http://127.0.0.1:{port}"
    if existing.rstrip("/") == url:
        return False
    env[_ENV_KEY] = url
    data["env"] = env
    return _save(claude_dir, data)


def unwire(claude_dir: Path) -> bool:
    """Remove a memo-owned base URL. True when the file changed."""
    data = _load(claude_dir)
    if data is None:
        return False
    env = data.get("env")
    if not isinstance(env, dict):
        return False
    existing = str(env.get(_ENV_KEY) or "").strip()
    if not existing or not _is_ours(existing):
        return False
    del env[_ENV_KEY]
    data["env"] = env
    return _save(claude_dir, data)
