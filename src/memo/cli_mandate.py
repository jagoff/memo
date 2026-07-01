"""`memo mandate` — the cross-client inevitability lever (P4, lever a).

Clients without an ambient recall hook (Codex, Devin, Devin Desktop, OpenCode,
Cursor) won't read memo unless instructed to. This command emits the canonical
"consult memo first" mandate and, with --write, drops it into the project-local
instruction file each client actually reads — so memo becomes a mandated first
step, and the silent-gap flag in `memo usefulness` / `memo roi`
(EXPECTED_CONSUMERS now covers all four) makes non-compliance visible.

No external/global config is touched: --write only creates project-local files
in the current repo (AGENTS.md, .cursor/rules/memo.md). Safe, idempotent
(skips if the marker is already present).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import click

_MARKER = "<!-- memo-mandate -->"

MANDATE_TEXT = f"""{_MARKER}
## Memory-first (memo)

Before deciding or answering anything that prior work might already cover,
consult memo FIRST:
- Start with `memo_unified_briefing` (or `memo_search` / `memo_ask`) to
  pull durable facts, decisions, and preferences.
- Pass `source="<this-client>"` on the read tools so usage is attributed
  (e.g. `source="codex"`). A client that never appears in memo's consult log is
  flagged as a silent gap by `memo usefulness`.
- Write durable outcomes back with `memo_save` so the next session inherits
  them. memo is the source of truth for what is *known*; build on it, and
  contradict a surfaced memory only explicitly.
"""

# Project-local instruction file each client reads (relative to cwd).
_CLIENT_FILES: dict[str, str] = {
    "codex": "AGENTS.md",
    "devin": "AGENTS.md",
    "devin-desktop": "AGENTS.md",
    "opencode": "AGENTS.md",
    "cursor": ".cursor/rules/memo.md",
    "blackbox": "AGENTS.md",
}


def write_mandates_for_clients(
    clients: Iterable[str], *, cwd: Path | None = None, dry_run: bool = False
) -> list[tuple[str, str]]:
    """Write the mandate into the project-local files for the given clients.

    Returns a list of ``(relative_path, status)`` pairs in write order. Duplicate
    targets are collapsed so clients that share the same instruction file (codex /
    devin / opencode) only touch the file once.
    """
    root = cwd or Path.cwd()
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for client in clients:
        rel = _CLIENT_FILES.get(client)
        if rel is None or rel in seen:
            continue
        seen.add(rel)
        out.append((rel, _write_mandate(root / rel, dry_run=dry_run)))
    return out


def _write_mandate(target: Path, *, dry_run: bool) -> str:
    """Append the mandate to a project-local file (idempotent). Returns a status
    string. Creates parent dirs for nested paths (.cursor/rules)."""
    if target.is_file():
        existing = target.read_text(encoding="utf-8")
        if _MARKER in existing:
            return "already present (skip)"
        new = existing.rstrip() + "\n\n" + MANDATE_TEXT
    else:
        new = MANDATE_TEXT
    if dry_run:
        return "would write"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(new, encoding="utf-8")
    return "written"


@click.command(name="mandate")
@click.option(
    "--client",
    "client",
    type=click.Choice(
        ["all", "codex", "devin", "devin-desktop", "opencode", "cursor", "blackbox"]
    ),
    default=None,
    help="Write the mandate into this client's project-local instruction file.",
)
@click.option("--write", "do_write", is_flag=True, help="Write the file(s) (default: print only).")
@click.option(
    "--dry-run", is_flag=True, help="With --write, show what would change without writing."
)
def mandate(*, client: str | None = None, do_write: bool = False, dry_run: bool = False) -> None:
    """Print (or install) the 'consult memo first' mandate for non-hook clients."""
    if not client and not do_write:
        click.echo(MANDATE_TEXT)
        click.echo(
            "\n# Paste into the client's instruction file, or run with "
            "--client <codex|devin|devin-desktop|opencode|cursor|all> --write "
            "(project-local)."
        )
        return

    if not do_write:
        click.echo(MANDATE_TEXT)
        return

    if client in (None, "all"):
        targets = list(_CLIENT_FILES)
    elif client:
        targets = [client]
    cwd = Path.cwd()
    seen: set[str] = set()
    for c in targets:
        rel = _CLIENT_FILES[c]
        if rel in seen:  # codex+devin share AGENTS.md
            continue
        seen.add(rel)
        status = _write_mandate(cwd / rel, dry_run=dry_run)
        click.echo(f"  {rel:<22} {status}")
