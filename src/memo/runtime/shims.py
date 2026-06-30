"""Memo agent shims — wrap agent binaries to show [Memo] banner at startup.

Each shim is a small bash script placed in ~/.memo/bin/ (before other
agent locations in PATH). When an agent starts:

  memo shim → detects next binary in PATH
    show memo banner, then exec the next binary

Chain: ~/.memo/bin/codex → any downstream wrapper or real codex

Safety: shims always exec the real binary; idempotent; no-clobber for
        non-memo files (marker line `# memo-shim` used to detect ours).
"""

from __future__ import annotations

import stat
from pathlib import Path

import click

_AGENTS = ("codex", "devin", "opencode", "gemini", "blackbox")
_DEFAULT_BIN_DIR = Path.home() / ".memo" / "bin"
_SHIM_MARKER = "# memo-shim"

_SHIM_TEMPLATE = """\
#!/usr/bin/env bash
# memo-shim v2 — written by `memo install-shims`. Do not edit manually.
set -euo pipefail
_MEMO_BIN_DIR="$(cd "$(dirname "$0")" && pwd -P)"
_AGENT="$(basename "$0")"
# Capture the TTY before the agent changes it, so async hooks can write
# idle-capture notifications directly to this terminal.
if [ -z "${MEMO_AGENT_TTY:-}" ] && [ -t 2 ]; then
    MEMO_AGENT_TTY="$(tty 2>/dev/null || true)"
    export MEMO_AGENT_TTY
fi
_NEXT=""
IFS=':' read -ra _DIRS <<< "$PATH"
for _D in "${_DIRS[@]:-}"; do
    [ "$_D" = "$_MEMO_BIN_DIR" ] && continue
    [ -x "$_D/$_AGENT" ] && { _NEXT="$_D/$_AGENT"; break; }
done
if [ -z "$_NEXT" ]; then
    printf 'memo: shim: %s: not found in PATH\\n' "$_AGENT" >&2
    exit 127
fi
_MEMO="$(command -v memo 2>/dev/null || true)"
if [ -n "$_MEMO" ] && [ "${MEMO_STARTUP_BANNER:-1}" != "0" ] && [ -t 2 ]; then
    "$_MEMO" startup-banner --agent "$_AGENT" 2>/dev/null || true
fi
exec "$_NEXT" "$@"
"""


_PATH_MARKER = "# memo-shims PATH"
_TTY_MARKER = "# memo-agent-tty"
# Writes TTY to env AND to state file so Claude Code hooks (which don't inherit
# shell env) can still find the active terminal via _read_agent_tty_file().
_TTY_SNIPPET = (
    "\n{m}\n"
    '[ -t 1 ] && export MEMO_AGENT_TTY="$(tty 2>/dev/null || true)"'
    " && printf '%s' \"$MEMO_AGENT_TTY\""
    ' > "${{XDG_DATA_HOME:-$HOME/.local/share}}/memo/agent_tty" 2>/dev/null || true'
    "  {m}\n"
)
_TTY_SNIPPET_V1 = "[ -t 1 ] && export MEMO_AGENT_TTY="  # old snippet without file write


def install_path_snippet(
    bin_dir: Path = _DEFAULT_BIN_DIR,
    *,
    dry_run: bool = False,
) -> str:
    """Prepend bin_dir to PATH and export MEMO_AGENT_TTY in ~/.zshrc / ~/.bashrc. Idempotent.

    Returns a short status string: "written", "already", or "skipped:<reason>".
    """
    import os

    shell = Path(os.environ.get("SHELL", "")).name
    rc_name = ".zshrc" if shell == "zsh" else ".bashrc"
    rc_path = Path.home() / rc_name
    path_snippet = f'\n{_PATH_MARKER}\nexport PATH="{bin_dir}:$PATH"  {_PATH_MARKER}\n'
    tty_snippet = _TTY_SNIPPET.format(m=_TTY_MARKER)

    existing = rc_path.read_text(encoding="utf-8") if rc_path.is_file() else ""
    path_present = _PATH_MARKER in existing
    tty_present = _TTY_MARKER in existing
    # Detect v1 snippet (no file write) and upgrade it in-place.
    tty_needs_upgrade = tty_present and _TTY_SNIPPET_V1 in existing and "agent_tty" not in existing

    if path_present and tty_present and not tty_needs_upgrade:
        return "already"
    if dry_run:
        return f"would-write:{rc_path}"
    try:
        if tty_needs_upgrade:
            import re

            new_content = re.sub(
                rf"{re.escape(_TTY_MARKER)}.*?{re.escape(_TTY_MARKER)}",
                tty_snippet.strip("\n"),
                existing,
                flags=re.DOTALL,
            )
            tmp = rc_path.with_suffix(rc_path.suffix + ".tmp")
            tmp.write_text(new_content, encoding="utf-8")
            os.replace(tmp, rc_path)
            return f"upgraded:{rc_path}"
        with rc_path.open("a", encoding="utf-8") as fh:
            if existing and not existing.endswith("\n"):
                fh.write("\n")
            if not path_present:
                fh.write(path_snippet)
            if not tty_present:
                fh.write(tty_snippet)
        return f"written:{rc_path}"
    except OSError as exc:
        return f"skipped:{exc}"


def install_shims(
    agents: tuple[str, ...] = _AGENTS,
    bin_dir: Path = _DEFAULT_BIN_DIR,
    *,
    dry_run: bool = False,
) -> list[str]:
    """Write agent shims to bin_dir. Returns list of result strings.

    Idempotent: overwrites only files that already contain our marker.
    Files without the marker are skipped (no-clobber).
    """
    bin_dir = Path(bin_dir)
    written: list[str] = []
    if not dry_run:
        bin_dir.mkdir(parents=True, exist_ok=True)

    for agent in agents:
        path = bin_dir / agent
        if path.exists():
            try:
                existing = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                existing = ""
            if _SHIM_MARKER not in existing:
                written.append(f"skip:{path}")
                continue

        if not dry_run:
            path.write_text(_SHIM_TEMPLATE, encoding="utf-8")
            path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        written.append(f"wrote:{path}")

    return written


@click.command(name="install-shims")
@click.option(
    "--agents",
    default=",".join(_AGENTS),
    show_default=True,
    help="Comma-separated agent names to shim.",
)
@click.option(
    "--bin-dir",
    default=str(_DEFAULT_BIN_DIR),
    show_default=True,
    help="Shim directory. Must appear in PATH before other agent locations.",
)
@click.option("--dry-run", is_flag=True, help="Show what would be written, write nothing.")
def install_shims_cmd(agents: str, bin_dir: str, dry_run: bool) -> None:
    """Install PATH shims that show [Memo ver] banner when agents start.

    \b
    Each shim wraps the next agent binary in PATH:
      - Show memo's own banner
      - Exec the next downstream wrapper or real binary

    After running, add the shim directory to PATH (before other agent dirs):

      export PATH="$HOME/.memo/bin:$PATH"

    Add that line to ~/.zshrc or ~/.bashrc.
    """
    from memo.cli_common import console

    agent_list = tuple(a.strip() for a in agents.split(",") if a.strip())
    bin_path = Path(bin_dir).expanduser()

    if dry_run:
        console.print(f"[dim]dry-run — would write to {bin_path}[/dim]")

    results = install_shims(agent_list, bin_path, dry_run=dry_run)

    for r in results:
        kind, path = r.split(":", 1)
        if kind == "skip":
            console.print(f"[yellow]skip[/yellow]  {path}  [dim](not a memo-shim)[/dim]")
        else:
            console.print(f"[green]wrote[/green] {path}")

    path_status = install_path_snippet(bin_path, dry_run=dry_run)
    if path_status.startswith("written"):
        console.print(f"[green]wrote[/green] PATH snippet → {path_status.split(':', 1)[1]}")
        console.print(f"[dim]Reload shell or run: source {path_status.split(':', 1)[1]}[/dim]")
    elif path_status == "already":
        console.print("[dim]✓ PATH snippet already in shell rc[/dim]")
    else:
        console.print(
            f"\n[bold]Add to your shell rc (before other agent dirs):[/bold]\n"
            f'  [cyan]export PATH="{bin_path}:$PATH"[/cyan]'
        )
