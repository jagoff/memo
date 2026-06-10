"""`memo session` command group — session checkpoints + autosave.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(session_group)`.
"""

from __future__ import annotations

import sys
from typing import Any

import click

from memo.cli_common import console
from memo.config import Config
from memo.flags import flag_bool, flag_int


@click.group(name="session")
def session_group() -> None:
    """Internal session-snapshot ops — hook targets, not user-facing.

    User-facing entry is `memo resume` (list / inspect). This group
    holds the wiring the hooks call: `checkpoint` (Stop hook), `recent`
    (SessionStart additionalContext), and `prune` (LRU cleanup). Stays
    namespaced so `memo --help` doesn't surface plumbing as if it were
    everyday CLI.
    """


@session_group.command(name="checkpoint")
@click.option(
    "--session-id", default=None, help="Override session_id (default: read from stdin payload)."
)
@click.option(
    "--cwd",
    default=None,
    help="Override cwd (default: read from stdin payload, fallback os.getcwd).",
)
@click.option("--transcript-path", default=None, help="Override transcript path.")
@click.option("--lru-cap", default=50, type=int, show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Print the persisted snapshot as JSON.")
def session_checkpoint(
    session_id: str | None,
    cwd: str | None,
    transcript_path: str | None,
    lru_cap: int,
    as_json: bool,
) -> None:
    """Stop hook entrypoint — upsert a session snapshot from stdin JSON.

    Reads the Stop hook payload from stdin (Claude Code passes
    `{"session_id", "transcript_path", "cwd", ...}`). Falls back to
    flags/cwd if stdin is empty (lets you run it manually for testing).

    Always exits 0 — like the other hooks, a checkpoint failure must
    not block Claude Code. On any exception we swallow + print `{}`.
    """
    import json as _json
    import os as _os
    import sys as _sys

    if flag_bool("MEMO_SESSION_DISABLE"):
        if as_json:
            click.echo("{}")
        sys.exit(0)

    payload: dict[str, Any] = {}
    # Stdin is a TTY when run interactively → don't block on read.
    if not _sys.stdin.isatty():
        try:
            raw = _sys.stdin.read()
            if raw.strip():
                payload = _json.loads(raw)
        except _json.JSONDecodeError:
            payload = {}

    sid = session_id or payload.get("session_id")
    cwd_resolved = cwd or payload.get("cwd") or _os.getcwd()
    transcript = transcript_path or payload.get("transcript_path")
    prompt_text = payload.get("prompt")  # present in UserPromptSubmit events

    if not sid:
        # Without a session_id we can't key the snapshot. Fail silently
        # so the hook still exits 0.
        if as_json:
            click.echo("{}")
        sys.exit(0)

    try:
        from memo.session import checkpoint as _checkpoint

        cfg = Config.from_env()
        cfg.ensure_dirs()
        snap = _checkpoint(
            cfg.state_dir,
            session_id=sid,
            cwd=cwd_resolved,
            transcript_path=transcript,
            prompt=prompt_text,
            lru_cap=lru_cap,
        )
    except Exception as exc:
        if flag_bool("MEMO_SESSION_DEBUG"):
            print(f"# memo session checkpoint failed: {exc}", file=_sys.stderr)
        if as_json:
            click.echo("{}")
        sys.exit(0)

    if as_json:
        click.echo(_json.dumps(snap, ensure_ascii=False, indent=2))


@session_group.command(name="autosave")
@click.option(
    "--threshold-kb",
    default=1024,
    type=int,
    show_default=True,
    help="Transcript size (KB) that triggers an autosave.",
)
@click.option(
    "--cooldown",
    default=300,
    type=int,
    show_default=True,
    help="Minimum seconds between autosaves for the same session.",
)
def session_autosave(threshold_kb: int, cooldown: int) -> None:
    """UserPromptSubmit hook — proactive save when context approaches limits.

    Stats the transcript file (fast, O(1)). If it exceeds threshold_kb
    and the per-session cooldown has elapsed, spawns ``memo capture-stop``
    in a detached background process and emits an additionalContext warning.

    Always exits 0 — never blocks the prompt.
    """
    import json as _json
    import os as _os
    import subprocess as _sp
    import sys as _sys

    if flag_bool("MEMO_SESSION_DISABLE"):
        print("{}")
        _sys.exit(0)

    payload: dict[str, Any] = {}
    if not _sys.stdin.isatty():
        try:
            raw = _sys.stdin.read()
            if raw.strip():
                payload = _json.loads(raw)
        except _json.JSONDecodeError:
            payload = {}

    sid = payload.get("session_id")
    transcript = payload.get("transcript_path")
    cwd = payload.get("cwd") or _os.getcwd()

    if not sid or not transcript:
        print("{}")
        _sys.exit(0)

    try:
        from memo.session import check_autosave, mark_autosaved

        cfg = Config.from_env()
        should_save, size_kb = check_autosave(
            cfg.state_dir,
            session_id=sid,
            transcript_path=transcript,
            threshold_kb=threshold_kb,
            cooldown_secs=cooldown,
        )
    except Exception as exc:
        if flag_bool("MEMO_SESSION_DEBUG"):
            print(f"# session autosave check failed: {exc}", file=_sys.stderr)
        print("{}")
        _sys.exit(0)

    if not should_save:
        print("{}")
        _sys.exit(0)

    # Spawn capture-stop detached, passing the hook payload via stdin.
    try:
        env = {**_os.environ, "MEMO_NONINTERACTIVE": "1"}
        capture_payload = _json.dumps(
            {
                "session_id": sid,
                "transcript_path": transcript,
                "cwd": cwd,
            }
        ).encode()
        proc = _sp.Popen(
            ["memo", "capture-stop"],
            stdin=_sp.PIPE,
            stdout=_sp.DEVNULL,
            stderr=_sp.DEVNULL,
            start_new_session=True,
            env=env,
            cwd=cwd,
        )
        if proc.stdin is not None:
            proc.stdin.write(capture_payload)
            proc.stdin.close()
        mark_autosaved(cfg.state_dir, sid)
    except Exception as exc:
        if flag_bool("MEMO_SESSION_DEBUG"):
            print(f"# session autosave spawn failed: {exc}", file=_sys.stderr)
        print("{}")
        _sys.exit(0)

    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": (
                "## ⚠️ Snapshot automático\n\n"
                f"El transcript de esta sesión supera los {size_kb} KB "
                f"(umbral: {threshold_kb} KB) — contexto próximo al límite. "
                "Lancé `memo capture-stop` en segundo plano para preservar "
                "los insights más importantes. Podés continuar con normalidad; "
                "si hay compactación automática, lo esencial ya está en memo.\n"
            ),
        }
    }
    print(_json.dumps(output, ensure_ascii=False))


@session_group.command(name="refresh-summary")
def session_refresh_summary() -> None:
    """Stop hook entrypoint — generate/update running_summary for the active session.

    Reads the Stop hook payload from stdin, runs the 3B LLM helper to
    summarize the session arc, and stores it in the snapshot. Throttled:
    skips if fewer than 3 new turns since the last summary. Always exits 0.
    """
    import json as _json
    import sys as _sys

    if flag_bool("MEMO_SESSION_DISABLE"):
        _sys.exit(0)

    payload: dict[str, Any] = {}
    if not _sys.stdin.isatty():
        try:
            raw = _sys.stdin.read()
            if raw.strip():
                payload = _json.loads(raw)
        except _json.JSONDecodeError:
            payload = {}

    sid = payload.get("session_id")
    if not sid:
        _sys.exit(0)

    try:
        from memo.session import refresh_summary as _refresh_summary

        cfg = Config.from_env()
        cfg.ensure_dirs()
        _refresh_summary(cfg.state_dir, sid)
    except Exception as exc:
        if flag_bool("MEMO_SESSION_DEBUG"):
            print(f"# memo session refresh-summary failed: {exc}", file=_sys.stderr)
    _sys.exit(0)


@session_group.command(name="recent")
@click.option("--limit", default=None, type=int, show_default=True)
def session_recent(limit: int | None) -> None:
    """SessionStart hook entrypoint — emit `additionalContext` markdown
    listing recent sessions. Same exit-0-silent contract as recall-hook."""
    import json as _json
    import os as _os
    import sys as _sys

    if flag_bool("MEMO_SESSION_DISABLE"):
        print("{}")
        _sys.exit(0)

    if limit is None:
        limit = flag_int("MEMO_SESSION_RECENT_LIMIT") or 12

    try:
        from memo.session import (
            _strip_command_wrappers,
            format_relative,
            is_command_noise,
            list_sessions,
        )

        cfg = Config.from_env()
        rows = list_sessions(cfg.state_dir, limit=limit)
    except Exception as exc:
        if flag_bool("MEMO_SESSION_DEBUG"):
            print(f"# memo session recent failed: {exc}", file=_sys.stderr)
        print("{}")
        _sys.exit(0)

    if not rows:
        print("{}")
        _sys.exit(0)

    from pathlib import Path as _Path

    cur_cwd = str(_Path(_os.getcwd()).resolve())
    same_cwd = [r for r in rows if (r.get("cwd") or "") == cur_cwd]
    top = same_cwd[0] if same_cwd else None

    def _clean_summary(r: dict, width: int) -> str:
        """First non-noise candidate among stored summary / last prompt /
        running-summary — heals junk persisted by pre-fix checkpoints."""
        for cand in (r.get("summary"), r.get("last_user_msg")):
            if not is_command_noise(cand):
                return _strip_command_wrappers(cand or "").replace("\n", " ")[:width]
        rs = r.get("running_summary")
        if rs and not is_command_noise(rs):
            return rs.strip().replace("\n", " ")[:width]
        return "—"

    lines: list[str] = []

    if top:
        sid = top.get("session_id") or ""
        when = format_relative(top.get("updated"))
        branch = top.get("branch") or "—"
        turns = top.get("turn_count") or 0
        # Prefer running_summary (LLM-generated arc) over plain last_user_msg.
        running_summary = top.get("running_summary")
        summary = _clean_summary(top, 120)
        lines.extend(
            [
                "## Sesión anterior detectada — ¿continuar?",
                "",
                f"Había una sesión activa en este directorio ({when}):",
                f"- **Resumen**: {summary}",
                f"- **Branch**: `{branch}`  |  **Turnos**: {turns}",
                f"- **Para retomar** (en una nueva terminal): `claude --resume {sid}`",
                "",
            ]
        )
        if running_summary:
            lines.extend(
                [
                    "### El Briefing",
                    "",
                    running_summary.strip(),
                    "",
                ]
            )
        prompt_trail = top.get("prompt_trail") or []
        if prompt_trail:
            lines.append("### Loops abiertos (últimos 7 días)")
            lines.append("")
            for i, p in enumerate(reversed(prompt_trail[-3:]), 1):
                lines.append(f"{i}. {p.strip()}")
            lines.append("")
        lines.extend(
            [
                "> **Acción**: Al iniciar esta conversación, pregunta al usuario si quiere "
                "retomar la sesión anterior (ejecutando el comando de arriba en la terminal) "
                "o continuar con esta sesión nueva.",
                "",
            ]
        )

    def _render_table(title: str, items: list[dict]) -> None:
        if not items:
            return
        lines.extend(
            [
                f"### {title}",
                "",
                "| cuándo | proyecto | branch | resumen | id |",
                "|--------|----------|--------|---------|----|",
            ]
        )
        for r in items:
            s = _clean_summary(r, 55).replace("|", "·")
            lines.append(
                f"| {format_relative(r.get('updated'))} | "
                f"{(r.get('project') or '—')[:18]} | "
                f"{(r.get('branch') or '—')[:14]} | "
                f"{s} | "
                f"`{(r.get('session_id') or '')[:8]}` |"
            )
        lines.append("")

    remaining = [r for r in rows if r is not top]
    if remaining:
        cur_project = _Path(cur_cwd).name
        this_project = [
            r
            for r in remaining
            if (r.get("cwd") or "") == cur_cwd or (r.get("project") or "") == cur_project
        ]
        others = [r for r in remaining if r not in this_project][:8]
        _render_table("Sesiones recientes en este proyecto", this_project)
        _render_table("Otros proyectos", others)
        lines.append("_`memo resume <id>` para ver detalles. `claude --resume <id>` para retomar._")

    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": "\n".join(lines),
        }
    }
    print(_json.dumps(output, ensure_ascii=False))


@session_group.command(name="prune")
@click.option("--cap", default=50, type=int, show_default=True)
def session_prune(cap: int) -> None:
    """Delete oldest sessions beyond `cap`. Idempotent."""
    from memo.session import prune_lru

    cfg = Config.from_env()
    n = prune_lru(cfg.state_dir, cap=cap)
    console.print(f"[green]✓[/green] pruned {n} session(s); cap={cap}")
