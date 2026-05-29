"""`memo diff` / `memo historia` — corpus diff + per-memoria history views.

Extracted from cli.py (2b god-module decomposition). Registered via
`cli.add_command(...)`.
"""

from __future__ import annotations

import json
from datetime import UTC

import click
from rich.panel import Panel

from memo.cli_common import _parse_as_of_date, console
from memo.config import Config


@click.command(name="diff")
@click.option("--from", "from_date", required=True,
              help="Start date — YYYY-MM-DD or full ISO 8601.")
@click.option("--to", "to_date", required=False, default=None,
              help="End date (default: now).")
@click.option("--json", "as_json", is_flag=True)
def diff_cmd(from_date: str, to_date: str | None, as_json: bool) -> None:
    """Diff the corpus between two snapshots.

    Shows added / removed / updated memorias plus a summary line. Useful
    for "what changed since last Monday" or "what evolved between two
    releases".
    """
    from datetime import datetime as _dt

    from memo.memory import Memory
    from memo.time_machine import diff as _diff

    to_iso = _dt.now(UTC).isoformat() if to_date is None else _parse_as_of_date(to_date)
    from_iso = _parse_as_of_date(from_date)

    mem = Memory(Config.from_env())
    d = _diff(mem, from_ts=from_iso, to_ts=to_iso)

    if as_json:
        click.echo(json.dumps({
            "from_ts": d.from_ts.isoformat(),
            "to_ts": d.to_ts.isoformat(),
            "added": [{"id": r.id, "title": r.title, "type": r.type} for r in d.added],
            "removed": [{"id": r.id, "title": r.title, "type": r.type} for r in d.removed],
            "updated": d.updated,
        }, ensure_ascii=False, indent=2))
        return

    console.print(Panel.fit(
        f"{d.from_ts.date().isoformat()}  →  {d.to_ts.date().isoformat()}\n"
        f"[bold]{d.summary()}[/bold]",
        title="corpus diff",
        border_style="cyan",
    ))
    if d.added:
        console.print(f"\n[green]+ added ({len(d.added)})[/green]")
        for r in d.added[:20]:
            console.print(f"  [green]+[/green] [{r.id[:8]}] {r.title}  [dim]({r.type})[/dim]")
    if d.removed:
        console.print(f"\n[red]- removed ({len(d.removed)})[/red]")
        for r in d.removed[:20]:
            console.print(f"  [red]-[/red] [{r.id[:8]}] {r.title}  [dim]({r.type})[/dim]")
    if d.updated:
        console.print(f"\n[yellow]~ updated ({len(d.updated)})[/yellow]")
        for u in d.updated[:20]:
            console.print(
                f"  [yellow]~[/yellow] [{u['id'][:8]}] {u['title']}  "
                f"[dim](fields: {', '.join(u['changed_fields'])})[/dim]",
            )


@click.command(name="historia")
@click.argument("id_or_prefix")
@click.option("--limit", default=50, type=int, show_default=True,
              help="Max events to show.")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def historia_cmd(id_or_prefix: str, limit: int, as_json: bool) -> None:
    """Show the full edit history for one memoria.

    Displays every save / update / delete event from the audit log,
    with field-level diffs on each update (title, type, tags, body_hash).
    Useful for answering "when did I change this?" or reviewing how a
    decision evolved over time.

    Examples:

      memo historia abc12345
      memo historia abc12345 --json
    """
    from memo.memory import AmbiguousIdError, Memory

    mem = Memory(Config.from_env())
    try:
        resolved = mem.resolve_id(id_or_prefix)
    except AmbiguousIdError as exc:
        console.print(f"[red]Ambiguous prefix:[/red] {exc}")
        raise SystemExit(1) from exc
    if resolved is None:
        console.print(f"[red]No record found for:[/red] {id_or_prefix!r}")
        raise SystemExit(1)

    events = mem.history.list_recent(limit=limit, record_id=resolved)
    events = list(reversed(events))  # chronological order

    if as_json:
        click.echo(json.dumps(events, ensure_ascii=False, indent=2, default=str))
        return

    r = mem.get(resolved)
    title_str = f"{r.title}" if r else resolved[:8]
    console.print(Panel.fit(
        f"[bold]{title_str}[/bold]  [dim]{resolved[:8]}[/dim]",
        title="historia",
        border_style="cyan",
    ))

    if not events:
        console.print("  [dim](no events in audit log)[/dim]")
        return

    _OP_STYLE = {"save": "green", "update": "yellow", "delete": "red"}

    for ev in events:
        op = ev.get("op", "?")
        ts = ev.get("ts", "")
        style = _OP_STYLE.get(op, "white")
        ts_short = ts[:16].replace("T", " ") if ts else "?"
        console.print(f"\n  [{style}]{op.upper():6s}[/{style}]  [dim]{ts_short}[/dim]")

        delta = ev.get("delta")
        if not delta:
            continue
        for field, pair in delta.items():
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            old_v, new_v = pair
            if field == "tags":
                old_s = ", ".join(old_v) if isinstance(old_v, list) else str(old_v)
                new_s = ", ".join(new_v) if isinstance(new_v, list) else str(new_v)
            elif field == "body_hash":
                old_s, new_s = str(old_v)[:12], str(new_v)[:12]
            else:
                old_s, new_s = str(old_v), str(new_v)
            console.print(
                f"           [dim]{field}:[/dim]  "
                f"[red]{old_s}[/red]  →  [green]{new_s}[/green]"
            )

    last_ts = events[-1].get("ts", "")
    console.print(f"\n  [dim]{len(events)} event(s) · last: {last_ts[:16].replace('T', ' ')}[/dim]")

