"""`memo capture-stop` / `resume` — Stop-hook capture + session resume.

Extracted from cli.py (god-module decomposition); ingest/mine-history/reflect
moved out to cli_ingest.py / cli_transcripts.py. Registered onto the root group
in cli.py via `cli.add_command(...)`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.panel import Panel
from rich.table import Table

from memo.cli_common import console
from memo.config import Config


@click.command(name="capture-stop")
def capture_stop() -> None:
    """Stop hook — passive auto-extract of insights from the last turn.

    Reads the Stop hook payload from stdin (Claude Code format), pulls
    the last (user, assistant) exchange from the transcript, asks the
    helper LLM (Qwen2.5-3B) to extract any actionable insights, dedups
    against the existing corpus, and saves survivors via Memory.save().

    Hook input (stdin, JSON):
      {"transcript_path": "/path/to/...jsonl", ...}

    Hook output (stdout):
      `{}`  — always. Capture is silent; the user discovers new
      memorias via `memo list` or the next ambient recall.

    Env vars:
      MEMO_CAPTURE_DISABLE  — set to "1" to make this a no-op.
      MEMO_CAPTURE_DEBUG    — set to "1" to print extraction progress
                              to stderr (helpful while tuning the
                              extraction prompt or trigger keywords).

    Failure modes are absorbed. The hook never blocks the user — at
    worst you get no auto-save for that turn.
    """
    import json as _json
    import sys as _sys

    from memo.flags import flag_bool

    if flag_bool("MEMO_CAPTURE_DISABLE"):
        print("{}")
        _sys.exit(0)

    debug = flag_bool("MEMO_CAPTURE_DEBUG")

    try:
        raw = _sys.stdin.read()
        payload = _json.loads(raw) if raw.strip() else {}
    except _json.JSONDecodeError:
        print("{}")
        _sys.exit(0)

    transcript_path = payload.get("transcript_path")
    if not transcript_path:
        print("{}")
        _sys.exit(0)

    try:
        from memo.capture import run_capture
        run_capture(Path(transcript_path), debug=debug)
    except Exception as exc:
        if debug:
            print(f"# memo capture-stop failed: {exc}", file=_sys.stderr)

    # Grounding (P0): score how much the answer used this turn's recalled
    # memorias → grounding.log (the outcome-based utility signal). Best-effort,
    # budget-guarded inside score_turn, never fails the turn.
    try:
        from memo import grounding
        from memo.config import Config
        summary = grounding.score_turn(Config.from_env().state_dir, payload)
        if debug and summary:
            print(f"# memo grounding: {summary}", file=_sys.stderr)
    except Exception as exc:
        if debug:
            print(f"# memo grounding failed: {exc}", file=_sys.stderr)

    print("{}")
    _sys.exit(0)


# ── Session reflection (v0.5.0) ─────────────────────────────────────────────
#
# `memo reflect` — read a full session transcript, extract durable insights
# (decisions, facts, bugs, follow-ups), and save them as memorias + a session
# arc nota. Auto-idempotent via `reflected_at` stamp in the session snapshot.

@click.command(name="resume")
@click.argument("session_id", required=False)
@click.option("--limit", default=10, type=int, show_default=True,
              help="Max sessions to show (only used when SESSION_ID is omitted).")
@click.option("--project", default=None, help="Filter to one project basename.")
@click.option("--cwd", "cwd_filter", default=None,
              help="Filter to sessions for this exact cwd (resolved). "
                   "Used by the shell wrapper to ask 'what was open here?' "
                   "without manual path comparison.")
@click.option("--json", "as_json", is_flag=True)
def resume(
    session_id: str | None, limit: int,
    project: str | None, cwd_filter: str | None, as_json: bool,
) -> None:
    """Recent sessions to retomar — picker for the SessionStart flow.

    With no argument, prints a table of the most recent sessions
    (cwd / branch / summary / id). Pass SESSION_ID (full or unique
    prefix ≥4 chars) to inspect one session in detail.

    Storage is sidecar JSON under `~/.local/share/memo/sessions/`,
    auto-written by the Stop hook (`memo session checkpoint`) and
    LRU-capped at 50.
    """
    from memo.session import format_relative, get_session, list_sessions

    cfg = Config.from_env()

    # Detail view — one session.
    if session_id:
        snap = get_session(cfg.state_dir, session_id)
        if snap is None:
            console.print(f"[red]not found:[/red] {session_id}")
            sys.exit(1)
        if as_json:
            click.echo(json.dumps(snap, ensure_ascii=False, indent=2))
            return
        mods = snap.get("modified_files") or []
        mods_line = ", ".join(mods[:5])
        if len(mods) > 5:
            mods_line += f", …(+{len(mods) - 5})"
        sid = snap.get("session_id") or ""
        console.print(Panel.fit(
            f"[bold]{snap.get('summary') or snap.get('last_user_msg') or 'session'}[/bold]\n"
            f"[dim]session_id:[/dim] {sid}\n"
            f"[dim]project:[/dim]    {snap.get('project') or '—'}\n"
            f"[dim]cwd:[/dim]        {snap.get('cwd') or '—'}\n"
            f"[dim]branch:[/dim]     {snap.get('branch') or '—'}\n"
            f"[dim]head:[/dim]       {snap.get('head_commit') or '—'}\n"
            f"[dim]modified:[/dim]   {mods_line or '—'}\n"
            f"[dim]transcript:[/dim] {snap.get('transcript_path') or '—'}\n"
            f"[dim]created:[/dim]    {snap.get('created')}  ({format_relative(snap.get('created'))})\n"
            f"[dim]updated:[/dim]    {snap.get('updated')}  ({format_relative(snap.get('updated'))})\n"
            f"[dim]turns:[/dim]      {snap.get('turn_count')}\n\n"
            f"{snap.get('last_user_msg') or ''}",
            title="session", border_style="cyan",
        ))
        if sid:
            console.print(
                f"\n[bold green]Para retomar:[/bold green]  "
                f"[cyan]claude --resume {sid}[/cyan]\n"
                f"[dim](copy-paste; corré el comando desde "
                f"`{snap.get('cwd') or '?'}`)[/dim]",
            )
        return

    # List view — picker.
    rows = list_sessions(
        cfg.state_dir, limit=limit, project=project, cwd=cwd_filter,
    )
    if as_json:
        click.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    if not rows:
        console.print("[dim]no sessions yet — run a checkpoint first[/dim]")
        return

    # When the caller passed an explicit --cwd, the list is already
    # filtered to that cwd — printing a "Última en este proyecto"
    # banner on top of a homogeneous list would be redundant.
    if cwd_filter:
        same_cwd = []
    else:
        # Bias: if there's a session for the current cwd, surface it on top
        # with the exact resume command. The whole point of the picker is
        # crash recovery — if you crashed and reopened terminal in the same
        # project, the very first thing you want to see is "click here to
        # resume", not a generic chronological list.
        import os as _os
        from pathlib import Path as _Path
        cur_cwd = str(_Path(_os.getcwd()).resolve())
        same_cwd = [r for r in rows if (r.get("cwd") or "") == cur_cwd]
    if same_cwd:
        top = same_cwd[0]
        sid = top.get("session_id") or ""
        console.print(
            f"[bold green]Última en este proyecto[/bold green]  "
            f"[dim]({format_relative(top.get('updated'))})[/dim]: "
            f"{(top.get('summary') or top.get('last_user_msg') or '—')[:80]}",
        )
        console.print(
            f"[bold green]Para retomar:[/bold green]  "
            f"[cyan]claude --resume {sid}[/cyan]\n",
        )

    tbl = Table(show_lines=False, expand=True)
    tbl.add_column("when", width=10)
    tbl.add_column("project", width=14, overflow="fold")
    tbl.add_column("branch", width=14, overflow="fold")
    tbl.add_column("turns", justify="right", width=5)
    tbl.add_column("summary", overflow="fold")
    tbl.add_column("session_id", overflow="fold")
    for r in rows:
        tbl.add_row(
            format_relative(r.get("updated")),
            r.get("project") or "—",
            r.get("branch") or "—",
            str(r.get("turn_count") or 0),
            (r.get("summary") or r.get("last_user_msg") or "—")[:80],
            r.get("session_id") or "—",
        )
    console.print(tbl)
    console.print(
        "[dim]Detalle: `memo resume <id|prefix>`  ·  "
        "Retomar: `claude --resume <session_id>` (copy desde la tabla).[/dim]",
    )
