"""Memo agent shims — wrap agent binaries to show [MEMO] banner at startup.

Each shim is a small bash script placed in ~/.memo/bin/ (before other
agent locations in PATH). When an agent starts:

  memo shim → detects next binary in PATH
    if next is a memflow shim  → exec it (memflow shows combined banner)
    if next is the real binary → show memo banner, then exec it

Chain: ~/.memo/bin/codex → ~/.memflow/bin/codex → real codex
       ~/.memo/bin/opencode → real opencode  (no memflow shim for opencode)

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
# memo-shim v1 — written by `memo install-shims`. Do not edit manually.
set -euo pipefail
_MEMO_BIN_DIR="$(cd "$(dirname "$0")" && pwd -P)"
_AGENT="$(basename "$0")"
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
# If next is a memflow shim, it already shows memo via its banner — skip ours.
if ! grep -qF 'MEMFLOW_STARTUP_BANNER' "$_NEXT" 2>/dev/null; then
    _MEMO="$(command -v memo 2>/dev/null || true)"
    if [ -n "$_MEMO" ] && [ "${MEMO_STARTUP_BANNER:-1}" != "0" ] && [ -t 2 ]; then
        "$_MEMO" startup-banner --agent "$_AGENT" 2>/dev/null || true
    fi
fi
exec "$_NEXT" "$@"
"""


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
    """Install PATH shims that show [MEMO ver] banner when agents start.

    \b
    Each shim wraps the next agent binary in PATH:
      - If memflow is next → exec it (memflow shows a combined memo+memflow box)
      - If the real binary is next → show memo's own banner, then exec it

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

    if not dry_run and results:
        console.print(
            f"\n[bold]Add to your shell rc (before other agent dirs):[/bold]\n"
            f"  [cyan]export PATH=\"{bin_path}:$PATH\"[/cyan]"
        )
