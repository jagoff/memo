"""`memo install-statusline` — install the memo statusline badge for Claude Code.

Copies the bundled ``memo-statusline.sh`` to ``~/.claude/memo-statusline.sh`` and
wires it into ``~/.claude/settings.json`` as the ``statusLine`` command. The
statusline prints a compact ``<dir> · <branch> · <model> · [MEMO <version>]``
line, with the version derived from the installed dist-info dir (no python
launch — fast enough to run every prompt).

Chain-aware (the systemic guarantee): if a *foreign* ``statusLine`` already
exists (caveman, memflow, a hand-rolled one), memo does **not** skip — it
**wraps** that command so the inner statusline still renders and ``[MEMO
<version>]`` is prepended. This makes the badge appear on ANY machine,
coexisting with whatever was there, instead of silently never wiring. ``--force``
collapses back to a memo-only standalone line.

The same idempotent core (`wire_statusline`) is re-asserted on memo-mcp start
(``MEMO_STATUSLINE_SELFHEAL``) so drift — e.g. another tool clobbering the key
later — self-repairs on the next session.
"""

from __future__ import annotations

import json
import os
import shlex
import stat
from importlib.resources import files as package_files
from pathlib import Path

import click

from memo.cli_common import console

_STATUSLINE_ASSET = ("agent_assets", "statusline", "memo-statusline.sh")
_DEST_NAME = "memo-statusline.sh"


def _claude_dir() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or (Path.home() / ".claude"))


def _bundled_statusline() -> Path:
    # Installed wheel: force-included at memo/agent_assets/statusline/. Dev tree:
    # the tracked source lives at the repo-root statusline/ dir (agent_assets is a
    # build artifact and may be absent in a fresh checkout). Try both, like
    # runtime/mcp.py resolves its assets.
    try:
        asset = package_files("memo")
        for part in _STATUSLINE_ASSET:
            asset = asset / part
        installed = Path(str(asset))
        if installed.is_file():
            return installed
    except Exception:
        installed = None  # type: ignore[assignment]
    repo_root_src = Path(__file__).resolve().parents[2] / "statusline" / _DEST_NAME
    if repo_root_src.is_file():
        return repo_root_src
    # Fall back to the (possibly missing) installed path so the caller reports it.
    return installed if installed is not None else repo_root_src


def _ensure_script(claude_dir: Path) -> Path:
    """Copy the bundled statusline script to ``claude_dir`` (only when changed)."""
    src = _bundled_statusline()
    if not src.is_file():
        raise FileNotFoundError(f"bundled statusline asset not found at {src}")
    claude_dir.mkdir(parents=True, exist_ok=True)
    dest = claude_dir / _DEST_NAME
    new = src.read_text(encoding="utf-8")
    if (not dest.is_file()) or dest.read_text(encoding="utf-8") != new:
        dest.write_text(new, encoding="utf-8")
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return dest


def _compute_statusline(
    existing: object, dest: Path, *, force: bool
) -> tuple[dict[str, str] | None, str]:
    """Decide the ``statusLine`` value. Returns ``(new_value_or_None, action)``.

    ``None`` means "leave the existing value unchanged" (already correct).
    ``action`` ∈ {standalone, replaced, wrapped, already}.
    """
    memo_cmd = f'bash "{dest}"'
    standalone = {"type": "command", "command": memo_cmd}

    # Force, or no/invalid prior config → memo-only standalone line.
    if force or not isinstance(existing, dict):
        if existing == standalone:
            return None, "already"
        return standalone, ("replaced" if isinstance(existing, dict) else "standalone")

    existing_cmd = existing.get("command")
    if not isinstance(existing_cmd, str) or not existing_cmd.strip():
        # Malformed prior statusLine (no usable inner command) → standalone.
        return standalone, "standalone"

    if str(dest) in existing_cmd:
        # Already memo's own command (standalone or an existing wrap) → no-op.
        # Prevents memo-around-memo recursion on re-runs / self-heal.
        return None, "already"

    # Foreign statusline → wrap it once so both badges coexist.
    wrapped_cmd = f"{memo_cmd} --wrap {shlex.quote(existing_cmd)}"
    return {"type": "command", "command": wrapped_cmd}, "wrapped"


def wire_statusline(claude_dir: Path | None = None, *, force: bool = False) -> dict[str, str]:
    """Idempotently install the script + wire the statusLine. Reusable core.

    Returns ``{"action": ..., "dest": ...}``. Writes ``settings.json`` only when
    the value actually changes, so a correct install is a true no-op (safe to
    call on every memo-mcp start). Raises on unreadable/unwritable config.
    """
    claude_dir = claude_dir or _claude_dir()
    dest = _ensure_script(claude_dir)

    settings_path = claude_dir / "settings.json"
    settings: dict[str, object] = {}
    if settings_path.is_file():
        loaded = json.loads(settings_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            settings = loaded

    new_value, action = _compute_statusline(settings.get("statusLine"), dest, force=force)
    if new_value is not None:
        settings["statusLine"] = new_value
        tmp = settings_path.with_suffix(settings_path.suffix + ".tmp")
        tmp.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, settings_path)
    return {"action": action, "dest": str(dest)}


def selfheal_statusline() -> None:
    """Best-effort statusLine re-assert for memo-mcp start. Never raises."""
    try:
        from memo.flags import flag_bool

        if not flag_bool("MEMO_STATUSLINE_SELFHEAL"):
            return
        wire_statusline(force=False)
    except Exception:  # noqa: S110 — best-effort self-heal must never break mcp start
        pass


@click.command(name="install-statusline")
@click.option(
    "--force",
    is_flag=True,
    help="Collapse to a memo-only standalone statusLine (drop any wrapped statusline).",
)
def install_statusline(force: bool) -> None:
    """Install the memo version-badge statusline into Claude Code."""
    src = _bundled_statusline()
    if not src.is_file():
        raise click.ClickException(f"bundled statusline asset not found at {src}")

    claude_dir = _claude_dir()
    try:
        result = wire_statusline(claude_dir, force=force)
    except json.JSONDecodeError as exc:
        raise click.ClickException(
            f"could not parse {claude_dir / 'settings.json'}: {exc}. "
            "Fix it or add statusLine manually."
        ) from exc

    dest = result["dest"]
    console.print(f"[green]✓[/green] statusline script → {dest}")
    action = result["action"]
    if action == "wrapped":
        console.print(
            "[green]✓[/green] wrapped the existing statusLine — "
            "[MEMO <version>] is prepended to it."
        )
    elif action == "already":
        console.print("[dim]statusLine already points at the memo badge — nothing to do.[/dim]")
    elif action == "replaced":
        console.print("[green]✓[/green] replaced the existing statusLine with the memo badge.")
    else:  # standalone
        console.print("[green]✓[/green] set statusLine to the memo badge.")
    console.print("[dim]Open a new Claude Code session to see the [MEMO <version>] badge.[/dim]")
