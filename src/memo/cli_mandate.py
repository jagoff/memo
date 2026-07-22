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
- Keep it honest: when a surfaced memory is stale or contradicted, correct it
  instead of silently working around it — `memo_feedback_flag(kind="outdated")`
  to retire it, or `kind="wrong"` (with `superseded_by` when a replacement
  exists). Both archive reversibly, never hard-delete.
"""

# Project-local instruction file each client reads (relative to cwd).
_CLIENT_FILES: dict[str, str] = {
    "codex": "AGENTS.md",
    "devin": "AGENTS.md",
    "devin-desktop": "AGENTS.md",
    "opencode": "AGENTS.md",
    "cursor": ".cursor/rules/memo.md",
    "blackbox": "AGENTS.md",
    # Multi-agent expansion: AGENTS.md where the agent honors it, else its own file.
    "zed": "AGENTS.md",
    "antigravity": "AGENTS.md",
    "continue": "AGENTS.md",
    "vscode": ".github/copilot-instructions.md",
    "windsurf": ".windsurf/rules/memo.md",
    "cline": ".clinerules/memo.md",
    "roo": ".roo/rules/memo.md",
    "kiro": ".kiro/steering/memo.md",
    "goose": ".goosehints",
    "jetbrains": ".junie/guidelines.md",
    "warp": "WARP.md",
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
        [
            "all",
            "codex",
            "devin",
            "devin-desktop",
            "opencode",
            "cursor",
            "blackbox",
            "zed",
            "antigravity",
            "continue",
            "vscode",
            "windsurf",
            "cline",
            "roo",
            "kiro",
            "goose",
            "jetbrains",
            "warp",
        ]
    ),
    default=None,
    help="Write the mandate into this client's project-local instruction file.",
)
@click.option("--write", "do_write", is_flag=True, help="Write the file(s) (default: print only).")
@click.option(
    "--dry-run", is_flag=True, help="With --write, show what would change without writing."
)
@click.option(
    "--dynamic",
    is_flag=True,
    help="Also project your durable decisions/preferences as concrete standing "
    "rules (from memo). Implies --write.",
)
@click.option(
    "--sync",
    "do_sync",
    is_flag=True,
    help="Regenerate the auto-rules block in files that already have one "
    "(retires superseded rules, adds new ones). Creates no new files.",
)
def mandate(
    *,
    client: str | None = None,
    do_write: bool = False,
    dry_run: bool = False,
    dynamic: bool = False,
    do_sync: bool = False,
) -> None:
    """Print (or install) the 'consult memo first' mandate for non-hook clients."""
    if do_sync:
        _run_rules_sync(dry_run=dry_run)
        return

    if not client and not do_write and not dynamic:
        click.echo(MANDATE_TEXT)
        click.echo(
            "\n# Paste into the client's instruction file, or run with "
            "--client <codex|devin|opencode|cursor|vscode|kiro|goose|...|all> --write "
            "(project-local). Add --dynamic to also install your durable rules."
        )
        return

    do_write = do_write or dynamic  # --dynamic is inherently a write

    if not do_write:
        click.echo(MANDATE_TEXT)
        return

    targets = list(_CLIENT_FILES) if client in (None, "all") else [client]
    cwd = Path.cwd()
    seen: set[str] = set()
    for c in targets:
        rel = _CLIENT_FILES[c]
        if rel in seen:  # codex+devin share AGENTS.md
            continue
        seen.add(rel)
        status = _write_mandate(cwd / rel, dry_run=dry_run)
        click.echo(f"  {rel:<22} {status}")

    if dynamic:
        _run_rules_write(targets, dry_run=dry_run)


def _gather_dynamic_rules() -> list[tuple[str, str]]:
    """Load the standing rules from the live store (heavy imports deferred)."""
    from memo.cli_common import get_memory
    from memo.config import Config
    from memo.constitution import gather_rules

    cfg = Config.from_env()
    return gather_rules(get_memory(cfg), cfg)


def _run_rules_write(targets: list[str], *, dry_run: bool) -> None:
    from pathlib import Path

    from memo.constitution import write_rules_for_clients

    rules = _gather_dynamic_rules()
    click.echo("dynamic rules (durable decisions/preferences):")
    results = write_rules_for_clients(targets, rules, dry_run=dry_run)
    for rel, status in results:
        click.echo(f"  {rel:<22} {status}")
    if not dry_run and any(status == "written" for _rel, status in results):
        from memo.config import Config
        from memo.constitution import register_repo

        register_repo(Config.from_env().state_dir, Path.cwd())
        click.echo(
            "  (repo registered for nightly auto-sync — gated by MEMO_DYNAMIC_MANDATE_SYNC_ENABLED)"
        )
    click.echo(f"  ({len(rules)} standing rule(s))")


def _run_rules_sync(*, dry_run: bool) -> None:
    from memo.constitution import resync_rules_in_repo

    rules = _gather_dynamic_rules()
    results = resync_rules_in_repo(rules, dry_run=dry_run)
    if not results:
        click.echo("no memo-rules block found here (run `memo mandate --dynamic --write` first)")
    for rel, status in results:
        click.echo(f"  {rel:<22} {status}")
    click.echo(f"  ({len(rules)} standing rule(s) from durable decisions/preferences)")
