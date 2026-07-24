"""`memo startup-banner` — prints [Memo ver] status at agent launch.

Called by agent shims (installed via `memo install-shims`). Must stay
fast and MLX-free: only git + importlib.metadata, no embedding.

For opencode the printed banner is wiped by its TUI, so we also stamp the
live memo version into opencode's `username` — the only config-controlled
persistent text slot in its TUI (it has no native statusline/tagline for
custom text; plugin status-bar widgets are an open feature request). That
makes `[Memo <ver>]` show next to every user message.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import click


@click.command(name="startup-banner")
@click.option("--agent", default="", help="Agent name shown in header (e.g. codex).")
def startup_banner_cmd(agent: str) -> None:
    """Print memo status banner to stderr. Called by agent shims on startup."""
    import importlib.metadata

    try:
        version = importlib.metadata.version("mlx-memo")
    except Exception:
        version = "?"

    sync_str = _fast_sync_state()
    update_tag = _pending_update_tag()
    update_str = f" | ⬆ {update_tag} available — run: {_update_command()}" if update_tag else ""
    memo_line = f"[Memo {version}] | sync {sync_str}{update_str}"
    label = f"─── memo / {agent} " if agent else "─── memo "
    width = max(len(memo_line) + 2, len(label) + 4, 44)
    pad = max(0, width - len(label))

    sys.stderr.write(f"{label}{'─' * pad}\n")
    sys.stderr.write(f"{memo_line}\n")
    sys.stderr.write(f"{'─' * width}\n")

    # opencode's TUI wipes the banner above; stamp the version into its
    # `username` so [Memo <ver>] persists next to user messages.
    if agent == "opencode":
        _refresh_opencode_username(version)


def _opencode_config_path() -> Path:
    import os

    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "opencode" / "opencode.json"


# Matches a trailing " · [Memo x.y.z]" badge so it can be re-stamped idempotently.
_OPENCODE_BADGE_RE = re.compile(r"\s*·?\s*\[Memo [^\]]*\]\s*$")


def _refresh_opencode_username(version: str) -> str | None:
    """Stamp the live memo version into opencode's `username`.

    opencode exposes no statusline/tagline config for arbitrary text, so
    `username` (shown beside each user message) is the only persistent slot.
    Writes the pure-JSON `opencode.json` (comment-safe; it merges with the
    user's `opencode.jsonc`). Idempotent — only writes when the value changes.
    Returns the new value, or None if unchanged / opencode absent / on error.
    """
    import json

    path = _opencode_config_path()
    if not path.parent.is_dir():
        return None  # opencode not installed on this machine

    data: dict = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, ValueError):
            data = {}

    existing = data.get("username")
    base = _OPENCODE_BADGE_RE.sub("", existing).strip() if isinstance(existing, str) else ""
    if not base:
        import getpass

        try:
            base = getpass.getuser()
        except Exception:
            base = "user"

    desired = f"{base} · [Memo {version}]"
    if existing == desired:
        return None

    data["username"] = desired
    try:
        import os

        tmp = path.with_suffix(".json.memo-tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        return None
    return desired


def _pending_update_tag() -> str | None:
    """Read the pending update notification file. Fast, no network."""
    try:
        from memo.config import Config
        from memo.runtime.autoupdate import pending_update_tag

        return pending_update_tag(Config.from_env())
    except Exception:
        return None


def _update_command() -> str:
    """The update command for the current install channel.

    Homebrew is user-managed, so we offer `brew upgrade mlx-memo`; every other
    channel (pipx / uv tool / PyPI) updates through `memo update`.
    """
    try:
        from memo.runtime.detect import is_homebrew_install

        homebrew = is_homebrew_install()
    except Exception:
        homebrew = False
    return "brew upgrade mlx-memo" if homebrew else "memo update"


def _fast_sync_state() -> str:
    """Git-only sync check — no network, no MLX. Returns human-readable label."""
    try:
        from memo.config import Config
        from memo.sync_git import sync_status

        cfg = Config.from_env()
        st = sync_status(cfg)
        if not st.get("is_git_clone"):
            return "local"
        ahead = int(st.get("ahead") or 0)
        return "al día" if ahead == 0 else f"ahead {ahead}"
    except Exception:
        return "-"


@click.command(name="codex-badge")
@click.option("--agent", default="", help="Agent name; accepted for shim symmetry.")
def codex_badge_cmd(agent: str) -> None:
    """Show memo's version badge in Codex/Supacode's top-line notification area."""
    del agent
    from memo.runtime.codex_notify import emit_memo_badge

    emit_memo_badge()
