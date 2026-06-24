"""`memo install-statusline` — install the memo statusline badge for Claude Code.

Copies the bundled ``memo-statusline.sh`` to ``~/.claude/memo-statusline.sh`` and
wires it into ``~/.claude/settings.json`` as the ``statusLine`` command. The
statusline prints a compact ``<dir> · <branch> · <model> · [MEMO <version>]``
line, with the version derived from the installed dist-info dir (no python
launch — fast enough to run every prompt).

No-clobber by default: if a ``statusLine`` already exists, it is left untouched
and the user is told how to add the badge manually. ``--force`` overwrites.
"""

from __future__ import annotations

import json
import os
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


@click.command(name="install-statusline")
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite an existing statusLine in settings.json with the memo badge.",
)
def install_statusline(force: bool) -> None:
    """Install the memo version-badge statusline into Claude Code."""
    src = _bundled_statusline()
    if not src.is_file():
        raise click.ClickException(f"bundled statusline asset not found at {src}")

    claude_dir = _claude_dir()
    claude_dir.mkdir(parents=True, exist_ok=True)
    dest = claude_dir / _DEST_NAME
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    console.print(f"[green]✓[/green] copied statusline script → {dest}")

    settings_path = claude_dir / "settings.json"
    settings: dict[str, object] = {}
    if settings_path.is_file():
        try:
            loaded = json.loads(settings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise click.ClickException(
                f"could not parse {settings_path}: {exc}. Fix it or add statusLine manually."
            ) from exc
        if isinstance(loaded, dict):
            settings = loaded

    statusline_cmd = {
        "type": "command",
        "command": f'bash "{dest}"',
    }

    if "statusLine" in settings and not force:
        console.print(
            "[yellow]![/yellow] a statusLine is already configured in "
            f"{settings_path}; leaving it untouched.\n"
            "[dim]To add the memo badge manually, set statusLine.command to:[/dim]\n"
            f'  bash "{dest}"\n'
            "[dim]or re-run with --force to overwrite the existing statusLine.[/dim]"
        )
        return

    settings["statusLine"] = statusline_cmd
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    verb = "replaced" if force else "set"
    console.print(f"[green]✓[/green] {verb} statusLine in {settings_path}")
    console.print("[dim]Open a new Claude Code session to see the [MEMO <version>] badge.[/dim]")
