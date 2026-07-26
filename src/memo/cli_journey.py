"""``memo journey-check`` — run the user-journey verification harness.

Thin CLI wrapper over :mod:`memo.journey_check`. Renders a pass/fail/warn/skip
line per check (or ``--json``), and exits nonzero on any failure so it can gate.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict

import click

from memo.cli_common import console
from memo.journey_check import CHECK_NAMES, FAIL, PASS, SKIP, WARN, run_all

_SYMBOL = {
    PASS: "[green]✓[/green]",
    FAIL: "[red]✗[/red]",
    WARN: "[yellow]⚠[/yellow]",
    SKIP: "[dim]∅[/dim]",
}


@click.command(name="journey-check")
@click.option("--json", "as_json", is_flag=True, help="Emit a structured JSON list of results.")
@click.option(
    "--only",
    "only",
    multiple=True,
    type=click.Choice(CHECK_NAMES),
    help="Run only the named check(s). Repeatable.",
)
def journey_check(as_json: bool, only: tuple[str, ...]) -> None:
    """Verify the practical user journey end-to-end against a seeded isolated store.

    Exercises each user-facing function through its real code path (auto-save,
    auto-recall, uses-memory, token-savings, ux-messages, Day-0 install, live
    wiring), reports per-check status, and exits nonzero on any failure. The live
    corpus is never touched.
    """
    results, exit_code = run_all(only=list(only) or None)

    if as_json:
        click.echo(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))
        sys.exit(exit_code)

    console.print("[bold]memo journey-check[/bold]")
    for r in results:
        symbol = _SYMBOL.get(r.status, "?")
        console.print(f"  {symbol} [bold]{r.name:<14}[/bold] {r.detail}")

    failed = sum(1 for r in results if r.status == FAIL)
    warned = sum(1 for r in results if r.status == WARN)
    skipped = sum(1 for r in results if r.status == SKIP)
    parts = [f"{failed} failed", f"{warned} warning{'s' if warned != 1 else ''}"]
    if skipped:
        parts.append(f"{skipped} skipped")
    console.print(f"→ {' · '.join(parts)} · exit {exit_code}")
    sys.exit(exit_code)
